#!/usr/bin/env python
"""O1 -- Decay-value analysis: does KDA USE its per-channel forget gate?

QUESTION
--------
KDA (R1, R3) has one forget gate per key channel: 128 decay values per head,
2,560 per layer and token.  GDN (R2, R4) has one per head: 20 per layer and
token.  KDA is closer to the teacher in every comparison (RQ3).  If the 128
channels of a KDA head decay at about the same rate, KDA behaves like GDN in
this respect, and the per-channel gate cannot be what explains its lead.  If
they spread over orders of magnitude, the freedom is used.

No training.  One forward pass per model over held-out documents; nothing is
changed in any model.

WHAT IS MEASURED
----------------
Both fla layers hand their RAW gate projection to the kernel and let the kernel
form the decay (use_gate_in_kernel=True).  fla 0.5.2, kda.py / gated_deltanet.py:

    KDA   g = -exp(A_log[h]) * softplus(f_proj(x)[h, k] + dt_bias[h, k])   [T, H, K]
    GDN   g = -exp(A_log[h]) * softplus(a_proj(x)[h]    + dt_bias[h])      [T, H]

    per-step decay   alpha = exp(g)            (0 < alpha < 1)
    half-life        h     = ln 2 / (-g)       tokens until a stored item is halved
                                               by the forget gate alone

This script wraps the kernel function inside fla's own module, reads exactly
the g / A_log / dt_bias the layer passes, and recomputes g with the formula
above.  The formula is VERIFIED, not assumed: once per model and mechanism the
kernel is re-run with our precomputed g (use_gate_in_kernel=False) and its
output compared with the original call, next to deliberately wrong formulas
(fp32 where the kernel allows it; see verify()).  A mismatch stops the run.

Per linear layer and head, with y = log10(half-life):
  median half-life                      (from a fine histogram of y)
  total variance of y over tokens x channels, split exactly into
      within-token spread across the head's channels  E_t Var_k(y)
      token-to-token variation of the head mean       Var_t(mean_k y)
  channel_fraction = E_t Var_k(y) / Var(y)          0 for GDN by construction;
                     ~0 for KDA = its channels behave like one gate per head
  static share     = Var_k(mean_t y) / E_t Var_k(y) how much of the channel
                     spread is a fixed per-channel property (dt_bias, f_proj
                     bias direction) rather than input-dependent
  channel ratio p90/p10 of the channels' typical half-lives (10^mean_t y)
  write strength beta = sigmoid(b_proj(x)) (the kernel's own transform), mean
  "static" half-life at f_proj(x) = 0, i.e. from A_log and dt_bias alone

Caveat for the text: the half-life is the forget gate's contribution only; the
delta rule also erases by overwriting (beta), so it is an upper bound on how
long an item survives, not a measured memory span.

DATA
----
LAYOUT=docs exactly as heldout_kl.py: the first N_DOC documents of each
held-out set, each one sequence [BOS] text [EOS] cut at 2048 (the tokenizer
adds BOS).  T = 2048 > 64, so fla runs its chunk kernel -- the training kernel.

USAGE
-----
    python decay_analysis.py --selftest            # CPU, no model: checks the statistics
    python decay_analysis.py --dry-run --set all   # lists models and paths, loads nothing
    sbatch --export=ALL,SET=a decay_analysis.sbatch    # init + stage 1   (8 models)
    sbatch --export=ALL,SET=b decay_analysis.sbatch    # stage 2 web + maths (8 models)
    python decay_analysis.py --table               # print the table from saved JSONs

One JSON per model in $WORK_LUKAS/eval_results/decay_analysis/; a model whose
JSON exists is skipped, so a job killed at the 30-minute debug limit is
finished by resubmitting the same command.
"""
import argparse
import gc
import json
import math
import os
import sys
import time

import torch

W = os.environ.get("WORK_LUKAS", "/leonardo_work/EUHPC_D34_049/WORK_LUKAS")
O = f"{W}/output"
S = f"{W}/students"
TEACHER = f"{W}/hf_cache/hub/models--arcee-ai--AFM-4.5B-Base/snapshots/dab7922f9c868d479b365410304f466b007b1c5a"
CORPORA = {"web": f"{W}/datasets/fineweb_edu_3B_heldout", "math": f"{W}/datasets/finemath_4plus_heldout"}
OUTDIR = os.environ.get("OUTDIR", f"{W}/eval_results/decay_analysis")
SEQ, N_DOC = 2048, int(os.environ.get("N_DOC", 64))
DEV, DTYPE, IMPL = "cuda", torch.bfloat16, "sdpa"

# y = log10(half-life in tokens); bins of 0.05 decades from 0.1 to 10^7 tokens,
# plus one underflow and one overflow bin.
Y_LO, Y_HI, Y_STEP = -1.0, 7.0, 0.05
EDGES = torch.arange(Y_LO, Y_HI + Y_STEP / 2, Y_STEP, dtype=torch.float64)
NEG_G_MIN = 1e-9          # -g below this (half-life > 6.9e8 tokens) is clipped and counted

# name -> path.  Init students: `ls -d $WORK_LUKAS/students/*_init` -- the two
# GDN names marked CHECK were not recorded in the project docs.  A missing path
# is skipped with a message, never fatal.
SETS = {
    "a": [
        ("r1_init_kda_hyb", f"{S}/afm_kda_hybrid31_init"),
        ("r2_init_gdn_hyb", f"{S}/afm_gdn_hybrid31_init"),          # CHECK
        ("r3_init_kda_only", f"{S}/afm_kda_only_init"),
        ("r4_init_gdn_only", f"{S}/afm_gdn_only_init"),             # CHECK
        ("r1_s1_kda_hyb", f"{O}/r1_stage1_hybrid31/checkpoint-4580"),  # top level overwritten 09-29
        ("r2_s1_gdn_hyb", f"{O}/r2_stage1_gdn_hybrid31/checkpoint-4580"),
        ("r3_s1_kda_only", f"{O}/r3_stage1_kda_only/checkpoint-4580"),
        ("r4_s1_gdn_only", f"{O}/r4_stage1_gdn_only/checkpoint-4580"),
    ],
    "b": [
        ("r1_s2_kda_hyb", f"{O}/r1_stage2_kda_hybrid31_unfreeze/checkpoint-15260"),
        ("r2_s2_gdn_hyb", f"{O}/r2_stage2_gdn_hybrid31_unfreeze/checkpoint-15260"),
        ("r3_s2_kda_only", f"{O}/r3_stage2_kda_only_unfreeze/checkpoint-15260"),
        ("r4_s2_gdn_only", f"{O}/r4_stage2_gdn_only_unfreeze/checkpoint-15260"),
        ("r1_s2math_kda_hyb", f"{O}/r1_stage2_kda_hybrid31_finemath/checkpoint-15260"),
        ("r2_s2math_gdn_hyb", f"{O}/r2_stage2_gdn_hybrid31_finemath/checkpoint-15260"),
        ("r3_s2math_kda_only", f"{O}/r3_stage2_kda_only_finemath/checkpoint-15260"),
        ("r4_s2math_gdn_only", f"{O}/r4_stage2_gdn_only_finemath/checkpoint-15260"),
    ],
}
SETS["all"] = SETS["a"] + SETS["b"]

# fla class -> (mechanism, kernel function names looked up in the class's own module)
KERNELS = {
    "KimiDeltaAttention": ("kda", ("chunk_kda", "fused_recurrent_kda")),
    "GatedDeltaNet": ("gdn", ("chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule")),
}


# --------------------------------------------------------------------------
# the gate, recomputed exactly as the fla 0.5.2 layers ask the kernel to
# --------------------------------------------------------------------------
def log_decay(mech, g_raw, A_log, dt_bias):
    """g in log space, fp32.  KDA: [B,T,H,K] -> [B,T,H,K];  GDN: [B,T,H] -> [B,T,H,1]."""
    g_raw = g_raw.float()
    A = torch.exp(A_log.float())
    if mech == "kda":
        H, K = g_raw.shape[-2], g_raw.shape[-1]
        return -A.view(H, 1) * torch.nn.functional.softplus(g_raw + dt_bias.float().view(H, K))
    return (-A * torch.nn.functional.softplus(g_raw + dt_bias.float())).unsqueeze(-1)


def to_y(g):
    """log10 half-life in tokens, and the number of clipped entries."""
    neg = -g
    clipped = int((neg < NEG_G_MIN).sum())
    return torch.log10(math.log(2.0) / neg.clamp(min=NEG_G_MIN)), clipped


# --------------------------------------------------------------------------
# statistics -- pure tensor code, checked by --selftest
# --------------------------------------------------------------------------
class LayerStats:
    """Running sums for one layer.  y: [T, H, K] (K = 1 for GDN), beta: [T, H]."""

    def __init__(self, H, K, device):
        f64 = dict(dtype=torch.float64, device=device)
        self.H, self.K, self.n = H, K, 0
        self.ch_sum = torch.zeros(H, K, **f64)
        self.ch_sq = torch.zeros(H, K, **f64)
        self.within = torch.zeros(H, **f64)          # sum_t Var_k(y_t)
        self.hm_sum = torch.zeros(H, **f64)          # sum_t mean_k(y_t)
        self.hm_sq = torch.zeros(H, **f64)
        self.hist = torch.zeros(H, len(EDGES) + 1, dtype=torch.int64, device=device)
        self.b_sum = torch.zeros(H, **f64)
        self.b_sq = torch.zeros(H, **f64)
        self.clipped = 0

    def add(self, y, beta, clipped=0):
        y = y.double()
        T = y.shape[0]
        self.n += T
        self.ch_sum += y.sum(0)
        self.ch_sq += (y * y).sum(0)
        self.within += y.var(dim=-1, correction=0).sum(0)
        hm = y.mean(-1)
        self.hm_sum += hm.sum(0)
        self.hm_sq += (hm * hm).sum(0)
        idx = torch.bucketize(y.permute(1, 0, 2).reshape(self.H, -1).contiguous(), EDGES.to(y.device))
        self.hist.scatter_add_(1, idx, torch.ones_like(idx))
        b = beta.double()
        self.b_sum += b.sum(0)
        self.b_sq += (b * b).sum(0)
        self.clipped += clipped

    def finalize(self):
        n, K = self.n, self.K
        ch_mean = self.ch_sum / n                                   # [H, K]
        mean = self.ch_sum.sum(-1) / (n * K)                        # [H]
        total = self.ch_sq.sum(-1) / (n * K) - mean ** 2
        within = self.within / n                                    # E_t Var_k
        tokvar = self.hm_sq / n - (self.hm_sum / n) ** 2            # Var_t mean_k
        between = ch_mean.var(dim=-1, correction=0)               # Var_k mean_t
        safe = lambda a, b: torch.where(b > 1e-12, a / b.clamp(min=1e-12), torch.zeros_like(a))
        hl_ch = 10 ** ch_mean
        p90 = torch.quantile(hl_ch, 0.9, dim=-1)
        p10 = torch.quantile(hl_ch, 0.1, dim=-1)
        b_mean = self.b_sum / n
        return {
            "n_tokens": n,
            "median_y": [median_from_hist(h) for h in self.hist.cpu()],
            "mean_y": mean.tolist(),
            "var_total": total.tolist(),
            "var_within_token": within.tolist(),
            "var_token_headmean": tokvar.tolist(),
            "identity_residual": (total - within - tokvar).abs().max().item(),
            "channel_fraction": safe(within, total).tolist(),
            "static_share": safe(between, within).tolist(),
            "channel_ratio_p90_p10": (p90 / p10).tolist(),
            "channel_mean_y": ch_mean.tolist(),                     # [H][K]; K = 1 for GDN
            "hist": self.hist.cpu().tolist(),
            "beta_mean": b_mean.tolist(),
            "beta_sd": (self.b_sq / n - b_mean ** 2).clamp(min=0).sqrt().tolist(),
            "clipped": self.clipped,
        }


def median_from_hist(h):
    """Median of y from the binned counts (bin centre; resolution 0.05 decades)."""
    h = h.double()
    c = torch.cumsum(h, 0)
    i = int(torch.searchsorted(c, c[-1] / 2))
    if i == 0:
        return Y_LO - Y_STEP / 2
    if i >= len(EDGES):
        return Y_HI + Y_STEP / 2
    return float((EDGES[i - 1] + EDGES[i]) / 2)


def hist_fraction(hist, lo_tokens=None, hi_tokens=None):
    """Fraction of all entries whose half-life lies in [lo, hi) tokens (bin resolution)."""
    h = torch.tensor(hist, dtype=torch.float64).sum(0)
    # bucketize(right=False): index i holds EDGES[i-1] < y <= EDGES[i]; index 0 = underflow
    lo_i = 0 if lo_tokens is None else int(torch.bucketize(torch.tensor(math.log10(lo_tokens), dtype=torch.float64), EDGES))
    hi_i = len(h) if hi_tokens is None else int(torch.bucketize(torch.tensor(math.log10(hi_tokens), dtype=torch.float64), EDGES))
    return float(h[lo_i:hi_i].sum() / h.sum())


# --------------------------------------------------------------------------
# kernel interception
# --------------------------------------------------------------------------
class Run:
    """What the kernel wrapper writes into; one per model and corpus."""
    active = None            # the current Run, or None (wrapper passes through)
    layer = None             # set by a forward pre-hook on every fla layer
    mech = None
    checks = {}              # mechanism -> gate-formula check result, reset per model


def install_wrappers(model):
    """Wrap the kernels in fla's own modules (once per process) and hook every layer."""
    found = {}
    for _, m in model.named_modules():
        cls = type(m).__name__
        if cls in KERNELS:
            found.setdefault(cls, type(m))
            if not getattr(m, "_o1_hooked", False):
                def pre(mod, args, _cls=cls):
                    Run.layer, Run.mech = mod.layer_idx, KERNELS[_cls][0]
                m.register_forward_pre_hook(pre)
                m._o1_hooked = True
    for cls, typ in found.items():
        pymod = sys.modules[typ.__module__]
        for fname in KERNELS[cls][1]:
            orig = getattr(pymod, fname)
            if getattr(orig, "_o1_orig", None) is not None:
                continue                                    # already wrapped by an earlier model
            setattr(pymod, fname, make_wrapper(orig, fname))
            print(f"  wrapped {typ.__module__}.{fname}", flush=True)
    return {KERNELS[c][0] for c in found}


def make_wrapper(orig, fname):
    def wrapper(*args, **kw):
        run = Run.active
        if run is None:
            return orig(*args, **kw)
        assert not args, f"{fname} called with positional arguments -- the wrapper reads kwargs only"
        assert kw.get("use_gate_in_kernel") is True, f"{fname}: gate not formed in the kernel"
        assert kw.get("lower_bound") is None, f"{fname}: lower_bound set -- formula would differ"
        assert not kw.get("allow_neg_eigval", False), f"{fname}: allow_neg_eigval changes beta"
        mech, layer = Run.mech, Run.layer
        assert layer is not None, f"{fname}: fla layer has no layer_idx -- layers would be merged"
        assert kw["g"].shape[0] == 1, f"{fname}: batch {kw['g'].shape[0]} -- this script feeds one sequence"
        g = log_decay(mech, kw["g"], kw["A_log"], kw["dt_bias"])          # [1, T, H, K]
        y, clipped = to_y(g[0])
        beta = torch.sigmoid(kw["beta"].float())[0]                         # [T, H]
        run.add(layer, mech, y, beta, clipped, fname)
        if mech not in Run.checks:
            Run.checks[mech] = verify(orig, kw, g, mech)
        return orig(**kw)
    wrapper._o1_orig = orig
    return wrapper


@torch.no_grad()
def verify(orig, kw, g, mech):
    """Re-run the kernel with OUR precomputed g and with deliberately WRONG alternatives.

    v1 of this check compared max-abs differences of bf16 outputs against a 1 % perturbation of g.
    Both landed on ~one bf16 step of the largest output element (2^-8 = 3.9e-3), so it could not
    tell right from wrong (job 62949679/85: 6.9e-3 vs 1.03e-2, 5.08e-3 vs 5.08e-3).  v2:
      * runs q/k/v/beta/g in fp32 where the kernel allows it (fp32 output, no bf16 floor),
        falling back to bf16;
      * uses the relative Frobenius norm ||o - o_ref|| / ||o_ref|| over all outputs, not one max;
      * compares OUR g with plausible WRONG formulas (decay x0.9 / x1.1, no dt_bias, a different A,
        and for KDA one gate per head).  It passes only if ours is at least 5x closer to the
        kernel's own output than every wrong one.
    """
    g_shape = kw["g"].shape
    fit = lambda x: x.reshape(g_shape) if kw["g"].dim() == 3 else x       # GDN: [1,T,H,1] -> [1,T,H]
    A_log, dt = kw["A_log"], kw["dt_bias"]
    cands = {
        "ours": g,
        "decay x0.9": g * 0.9,
        "decay x1.1": g * 1.1,
        "no dt_bias": log_decay(mech, kw["g"], A_log, torch.zeros_like(dt)),
        "A -> log(1+A)": log_decay(mech, kw["g"], torch.log(torch.exp(A_log.float()).log1p()), dt),
    }
    if mech == "kda":
        cands["head-mean gate (GDN-like)"] = g.mean(-1, keepdim=True).expand_as(g)
    last = None
    for f32 in (True, False):
        cast = (lambda t: t.float()) if f32 else (lambda t: t)
        try:
            k32 = {k: (cast(v) if k in ("q", "k", "v", "beta", "g") else v) for k, v in kw.items()}
            ref = orig(**k32)[0].float()
            base = dict(k32, use_gate_in_kernel=False, A_log=None, dt_bias=None)
            err = {}
            for n, gg in cands.items():
                o = orig(**dict(base, g=fit(gg.float()).contiguous()))[0].float()
                err[n] = ((o - ref).norm() / ref.norm()).item()
            break
        except Exception as e:
            last = e
            print(f"  (check in {'fp32' if f32 else 'bf16'} not possible: {type(e).__name__}: {str(e)[:120]})", flush=True)
    else:
        print(f"  !! CHECK could not run ({type(last).__name__}: {last}).  The formula is fla's documented "
              f"one but UNVERIFIED for this run -- say so before using the numbers.", flush=True)
        return {"ok": None, "error": f"{type(last).__name__}: {last}"}
    # A "wrong" formula that coincides with ours for THIS model is not wrong here: for KDA with
    # GDN's forget gate (O2) the head-mean gate IS the gate.  Compare only real alternatives.
    same = {n for n, gg in cands.items() if n != "ours" and torch.allclose(gg, g, rtol=1e-5, atol=0)}
    wrong = min(v for n, v in err.items() if n != "ours" and n not in same)
    for n in same:
        err[n + " (= ours for this model)"] = err.pop(n)
    ok = err["ours"] < 0.2 * wrong
    print(f"  CHECK gate formula ({mech}, {'fp32' if f32 else 'bf16'}): relative output error  "
          + "  ".join(f"{n} {v:.1e}" for n, v in err.items()) + f"  -> {'OK' if ok else 'MISMATCH'}", flush=True)
    if not ok:
        raise SystemExit("FATAL: recomputed gate does not reproduce the kernel -- results would be wrong")
    return {"ok": True, "precision": "fp32" if f32 else "bf16", "rel_err": err}


class Collector:
    def __init__(self):
        self.layers, self.calls = {}, {}

    def add(self, layer, mech, y, beta, clipped, fname):
        T, H, K = y.shape
        if layer not in self.layers:
            self.layers[layer] = (mech, LayerStats(H, K, y.device))
        self.layers[layer][1].add(y, beta, clipped)
        self.calls[fname] = self.calls.get(fname, 0) + 1

    def finalize(self):
        return {"calls": self.calls,
                "layers": {str(l): dict(mech=m, **st.finalize()) for l, (m, st) in sorted(self.layers.items())}}


def static_halflife(model):
    """Half-life at f_proj(x) = 0: A_log and dt_bias alone, per layer x head (x channel)."""
    out = {}
    for _, m in model.named_modules():
        cls = type(m).__name__
        if cls not in KERNELS:
            continue
        mech = KERNELS[cls][0]
        H = m.A_log.numel()
        K = m.dt_bias.numel() // H
        zero = torch.zeros(H, K) if mech == "kda" else torch.zeros(H)
        g = log_decay(mech, zero, m.A_log.detach().cpu(), m.dt_bias.detach().cpu())   # [H, K]
        y, _ = to_y(g)
        out[str(m.layer_idx)] = {"mech": mech, "A": torch.exp(m.A_log.detach().float().cpu()).tolist(),
                                 "head_mean_y": y.mean(-1).tolist(), "channel_y": y.tolist()}
    return out


# --------------------------------------------------------------------------
# data and models
# --------------------------------------------------------------------------
def build_docs(tok, path):
    from datasets import load_from_disk
    ds, seqs = load_from_disk(path), []
    for row in ds:
        ids = (tok(row["text"]).input_ids + [tok.eos_token_id])[:SEQ]
        if len(ids) >= 64:                         # as heldout_kl.py: skip near-empty documents
            seqs.append(torch.tensor(ids))
        if len(seqs) == N_DOC:
            break
    assert len(seqs) == N_DOC, f"{path}: only {len(seqs)} documents"
    assert seqs[0][0].item() == tok.bos_token_id, "tokenizer did not add BOS -- not the training format"
    return seqs


@torch.no_grad()
def measure(name, path, docs):
    from transformers import AutoModelForCausalLM
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(path, dtype=DTYPE, attn_implementation=IMPL,
                                                 trust_remote_code=True).to(DEV).eval()
    cfg = model.config
    Run.checks = {}                                  # verify the gate once per model and mechanism
    mechs = install_wrappers(model)
    lin = sorted(getattr(cfg, "kda_layers", []) or [])
    print(f"  {name}: mechanism {getattr(cfg, 'mechanism', '?')}, {len(lin)} linear layers, "
          f"loaded in {time.time() - t0:.0f} s", flush=True)
    res = {"name": name, "path": path, "mechanism": getattr(cfg, "mechanism", None),
           "linear_layers": lin, "num_layers": cfg.num_hidden_layers, "n_doc": N_DOC,
           "static": static_halflife(model)}
    for corpus, seqs in docs.items():
        Run.active = Collector()
        n_tok = 0
        for ids in seqs:
            model.model(input_ids=ids.view(1, -1).to(DEV), use_cache=False)
            n_tok += ids.numel()
        col, Run.active = Run.active, None
        out = col.finalize()
        n_layers_seen = len(out["layers"])
        assert n_layers_seen == len(lin), f"{name}: saw {n_layers_seen} linear layers, config says {len(lin)}"
        assert all(v["n_tokens"] == n_tok for v in out["layers"].values()), "token count differs across layers"
        res[corpus] = out
        print(f"    {corpus}: {n_tok} tokens, kernel calls {out['calls']}  "
              + summary_line(out), flush=True)
    res["gate_check"] = {m: Run.checks.get(m) for m in mechs}
    del model
    gc.collect(); torch.cuda.empty_cache()
    return res


# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------
def summarize(corpus_res):
    L = corpus_res["layers"].values()
    hist = [h for v in L for h in v["hist"]]
    med = sorted(m for v in L for m in v["median_y"])
    cf = [c for v in L for c in v["channel_fraction"]]
    ss = [c for v in L for c, w in zip(v["static_share"], v["var_within_token"]) if w > 1e-12]
    ratio = sorted(r for v in L for r in v["channel_ratio_p90_p10"])
    beta = [b for v in L for b in v["beta_mean"]]
    return {
        "median_head_halflife": 10 ** med[len(med) // 2],
        "frac_lt16": hist_fraction(hist, None, 16),
        "frac_16_256": hist_fraction(hist, 16, 256),
        "frac_256_2048": hist_fraction(hist, 256, 2048),
        "frac_gt2048": hist_fraction(hist, 2048, None),
        "channel_fraction": sum(cf) / len(cf),
        "static_share": sum(ss) / len(ss) if ss else float("nan"),
        "median_channel_ratio": ratio[len(ratio) // 2],
        "beta_mean": sum(beta) / len(beta),
    }


def summary_line(corpus_res):
    s = summarize(corpus_res)
    return (f"median head half-life {s['median_head_halflife']:.1f} tok, channel fraction "
            f"{s['channel_fraction']:.3f}, p90/p10 {s['median_channel_ratio']:.1f}")


def table():
    rows = [json.load(open(os.path.join(OUTDIR, f))) for f in sorted(os.listdir(OUTDIR)) if f.endswith(".json")]
    order = [n for n, _ in SETS["all"]]
    rows.sort(key=lambda r: order.index(r["name"]) if r["name"] in order else 99)
    for corpus in CORPORA:
        print(f"\n=== {corpus} held-out, {N_DOC} documents, training format ===")
        print(f"{'model':22}{'med hl':>8}{'<16':>7}{'16-256':>8}{'256-2k':>8}{'>2k':>7}"
              f"{'ch.frac':>9}{'static':>8}{'p90/p10':>9}{'beta':>7}  check")
        for r in rows:
            if corpus not in r:
                continue
            s = summarize(r[corpus])
            chk = ",".join(f"{m}:{'ok' if (c or {}).get('ok') else '??'}" for m, c in r.get("gate_check", {}).items())
            print(f"{r['name'][:21]:22}{s['median_head_halflife']:8.1f}{s['frac_lt16']:7.2f}{s['frac_16_256']:8.2f}"
                  f"{s['frac_256_2048']:8.2f}{s['frac_gt2048']:7.2f}{s['channel_fraction']:9.3f}"
                  f"{s['static_share']:8.2f}{s['median_channel_ratio']:9.1f}{s['beta_mean']:7.2f}  {chk}")
    print("\nmed hl = median over layers x heads of the head's median half-life (tokens).  <16 ... >2k = share of all\n"
          "decay values (tokens x channels) by half-life.  ch.frac = E_t Var_k(log hl) / Var(log hl), mean over heads:\n"
          "0 = the head's channels share one decay (GDN by construction).  static = share of that channel spread that\n"
          "is fixed per channel.  p90/p10 = ratio of the 90th to 10th percentile channel half-life within a head (median\n"
          "over heads; 1 for GDN).  beta = mean write strength sigmoid(b).  Half-life = forget gate only (upper bound).")


# --------------------------------------------------------------------------
# self-test (CPU, no model, no fla)
# --------------------------------------------------------------------------
def selftest():
    torch.manual_seed(0)
    T, H, K = 300, 3, 8
    # head 0: channels identical; head 1: fixed per-channel offsets; head 2: random per token
    base = torch.randn(T, 1, 1).expand(T, H, K).clone()
    base[:, 1] += torch.linspace(-2, 2, K)
    base[:, 2] += torch.randn(T, K)
    st = LayerStats(H, K, "cpu")
    st.add(base[:100], torch.rand(100, H))
    st.add(base[100:], torch.rand(200, H))                 # two calls must equal one
    r = st.finalize()
    y = base.double()
    tot = y.permute(1, 0, 2).reshape(H, -1).var(-1, correction=0)
    assert torch.allclose(torch.tensor(r["var_total"], dtype=torch.float64), tot, atol=1e-9), "total variance"
    assert r["identity_residual"] < 1e-9, "Var = E Var_k + Var mean_k must hold exactly"
    cf = r["channel_fraction"]
    assert cf[0] < 1e-9 and 0.3 < cf[1] < 1 and 0.3 < cf[2] < 1, cf
    ss = r["static_share"]
    assert ss[1] > 0.99 and ss[2] < 0.2, ss                 # fixed offsets are static, noise is not
    assert sum(sum(h) for h in r["hist"]) == T * H * K, "histogram loses entries"
    med = r["median_y"][0]
    true_med = float(y[:, 0].reshape(-1).median())
    assert abs(med - true_med) <= Y_STEP, (med, true_med)
    # GDN shape: K = 1 -> channel fraction 0, ratio 1
    st1 = LayerStats(H, 1, "cpu")
    st1.add(torch.randn(T, H, 1), torch.rand(T, H))
    r1 = st1.finalize()
    assert max(r1["channel_fraction"]) == 0 and all(abs(x - 1) < 1e-9 for x in r1["channel_ratio_p90_p10"])
    # gate formula shapes and sign
    gk = log_decay("kda", torch.randn(1, 5, H, K), torch.log(torch.rand(H) * 15 + 1), torch.randn(H * K))
    gg = log_decay("gdn", torch.randn(1, 5, H), torch.log(torch.rand(H) * 16), torch.randn(H))
    assert gk.shape == (1, 5, H, K) and gg.shape == (1, 5, H, 1) and (gk < 0).all() and (gg <= 0).all()
    # half-life: alpha = 0.5 per step -> exactly 1 token
    yy, _ = to_y(torch.tensor([math.log(0.5)]))
    assert abs(yy.item()) < 1e-6
    f = hist_fraction(r["hist"], 10 ** -0.5, None) + hist_fraction(r["hist"], None, 10 ** -0.5)
    assert abs(f - 1) < 1e-12
    print("selftest OK: variance identity, channel fraction, static share, histogram, median, GDN case, gate shapes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default=os.environ.get("SET", "all"), choices=sorted(SETS))
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--model", action="append", default=[],
                    help="name=path, repeatable; replaces --set (used by the O2 eval chain)")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.table:
        return table()
    os.makedirs(OUTDIR, exist_ok=True)
    todo = []
    models = [tuple(m.split("=", 1)) for m in a.model] if a.model else SETS[a.set]
    for n, p in models:
        if os.path.exists(f"{OUTDIR}/{n}.json"):
            print(f"done  {n}")
        elif not os.path.isdir(p):
            print(f"SKIP  {n}: no directory {p}")
        else:
            todo.append((n, p)); print(f"todo  {n}: {p}")
    if a.dry_run:
        return
    if not todo:
        return table()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TEACHER)
    docs = {c: build_docs(tok, p) for c, p in CORPORA.items()}
    print(f"{N_DOC} documents per corpus; tokens: " + ", ".join(f"{c} {sum(x.numel() for x in s)}" for c, s in docs.items()))
    for n, p in todo:
        res = measure(n, p, docs)
        tmp = f"{OUTDIR}/{n}.json.tmp"
        json.dump(res, open(tmp, "w"))
        os.replace(tmp, f"{OUTDIR}/{n}.json")         # never leave a half-written JSON that would be skipped
    table()


if __name__ == "__main__":
    main()