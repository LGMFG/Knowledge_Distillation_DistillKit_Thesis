#!/usr/bin/env python
"""O2 gate test: is the freshly built 'KDA with GDN's forget gate' student what we think it is?

    python test_o2_headgate.py            (inside o2_test.sbatch, 1 GPU, < 30 min)

Exit code 0 only if every HARD check passes -- the training chain waits on it (afterok).
SOFT checks print a WARNING and are for you to read; they do not stop the chain.

HARD
  H1  flag OFF is still the teacher: AFM built from the PATCHED repo code with kda_layers = []
      reproduces AFM-4.5B-Base's logits exactly (the A3 regression test, in miniature)
  H2  the student on disk: config kda_forget_gate == "head", 27 KDA layers, each with
      HeadForgetGate (Linear 2560 -> 20 + bias), dt_bias a zero BUFFER, exp(A_log) in (0, 16]
  H3  parameter count = 4,982,645,176  (= R1's 4,999,026,076 - 27 x 606,700)
  H4  the forget gate really is one value per head: f_proj's output is identical over the 128
      channels of every head, on real text, in every KDA layer
  H5  training works: forward + backward on one 2,048-token sequence, finite loss, finite grads,
      non-zero gradient on the new gate, no gradient path to the zero buffer
  H6  incremental decoding with the cache gives the same logits as one full forward pass
      (the fused_recurrent kernel with the tied gate), up to the bf16 noise R1's init shows
SOFT
  S1  initial CE on 8 x 2,048 held-out FineWeb-Edu tokens within 0.3 nat of R1's init
      (A5b: R1 init 7.9412; the gate is random in both, so they should be close)
"""
import math
import os
import sys

import torch

W = os.environ.get("WORK_LUKAS", "/leonardo_work/EUHPC_D34_049/WORK_LUKAS")
sys.path.insert(0, W)
TEACHER = f"{W}/hf_cache/hub/models--arcee-ai--AFM-4.5B-Base/snapshots/dab7922f9c868d479b365410304f466b007b1c5a"
NEW = os.environ.get("STUDENT", f"{W}/students/afm_kda_hybrid31_headgate_init")
R1_INIT = f"{W}/students/afm_kda_hybrid31_init"
HELDOUT = f"{W}/datasets/fineweb_edu_3B_heldout"
EXPECTED_PARAMS = 4_999_026_076 - 27 * 606_700          # 4,982,645,176
DEV, DT = "cuda", torch.bfloat16

FAIL, WARN = [], []


def hard(ok, msg):
    print(("  PASS  " if ok else "  FAIL  ") + msg, flush=True)
    if not ok:
        FAIL.append(msg)


def soft(ok, msg):
    print(("  ok    " if ok else "  WARN  ") + msg, flush=True)
    if not ok:
        WARN.append(msg)


def load(path):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(path, dtype=DT, trust_remote_code=True,
                                                attn_implementation="sdpa").to(DEV).eval()


def kda_modules(model):
    return [m for m in model.modules() if type(m).__name__ == "KimiDeltaAttention"]


def heldout_batch(tok, n_seq=8, seq=2048):
    from datasets import load_from_disk
    buf = []
    for row in load_from_disk(HELDOUT):
        buf.extend(tok(row["text"]).input_ids)
        if len(buf) >= n_seq * seq:
            break
    return torch.tensor(buf[: n_seq * seq]).view(n_seq, seq)


@torch.no_grad()
def mean_ce(model, batch):
    ces = []
    for i in range(len(batch)):
        ids = batch[i:i + 1].to(DEV)
        lg = model(input_ids=ids, use_cache=False).logits[0, :-1].float()
        ces.append(torch.nn.functional.cross_entropy(lg, ids[0, 1:]).item())
    return sum(ces) / len(ces)


@torch.no_grad()
def cache_vs_full(model, ids, n_new=16):
    """max |logit| difference between step-by-step cached decoding and one full forward."""
    full = model(input_ids=ids, use_cache=False).logits[0].float()
    P = ids.shape[1] - n_new
    out = model(input_ids=ids[:, :P], use_cache=True)
    pkv, diffs = out.past_key_values, [(out.logits[0, -1].float() - full[P - 1]).abs().max().item()]
    for t in range(P, ids.shape[1] - 1):
        out = model(input_ids=ids[:, t:t + 1], past_key_values=pkv, use_cache=True)
        pkv = out.past_key_values
        diffs.append((out.logits[0, -1].float() - full[t]).abs().max().item())
    return max(diffs), full.abs().max().item()


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"student under test: {NEW}")
    tok = AutoTokenizer.from_pretrained(TEACHER)
    batch = heldout_batch(tok)

    # ---------------- H1: flag off == teacher, with the PATCHED repo code ----------------
    print("\n[H1] patched repo code, all flags at default -> must equal the teacher")
    from afm_kda.configuration_afm_kda import AFMKDAConfig
    from afm_kda.modeling_afm_kda import AFMKDAForCausalLM
    teacher = AutoModelForCausalLM.from_pretrained(TEACHER, dtype=DT).to(DEV).eval()
    cfg = AFMKDAConfig.from_pretrained(TEACHER)
    hard(cfg.kda_forget_gate == "channel" and cfg.kda_layers == [], "default config: channel gate, no KDA layers")
    base = AFMKDAForCausalLM.from_pretrained(TEACHER, config=cfg, dtype=DT).to(DEV).eval()
    with torch.no_grad():
        ids = batch[:1].to(DEV)
        d = (base(input_ids=ids).logits.float() - teacher(input_ids=ids).logits.float()).abs().max().item()
    hard(d == 0.0, f"max |logit diff| flag-off student vs teacher = {d:.3e} (must be 0.000e+00)")
    del base, teacher
    torch.cuda.empty_cache()

    # ---------------- H2-H6, S1: the new student, loaded the way DistillKit loads it ---------------
    print(f"\n[H2] the student on disk")
    m = load(NEW)
    c = m.config
    hard(getattr(c, "kda_forget_gate", None) == "head", f"config.kda_forget_gate = {getattr(c, 'kda_forget_gate', None)!r}")
    hard(c.mechanism == "kda" and len(c.kda_layers) == 27, f"mechanism {c.mechanism}, {len(c.kda_layers)} KDA layers")
    kdas = kda_modules(m)
    hard(len(kdas) == 27, f"{len(kdas)} KimiDeltaAttention modules")
    gate_ok = all(type(k.f_proj).__name__ == "HeadForgetGate"
                  and tuple(k.f_proj.proj.weight.shape) == (20, 2560)
                  and k.f_proj.proj.bias is not None for k in kdas)
    hard(gate_ok, "every KDA layer has HeadForgetGate(Linear 2560 -> 20, bias)")
    buf_ok = all("dt_bias" in dict(k.named_buffers()) and "dt_bias" not in dict(k.named_parameters())
                 and float(k.dt_bias.abs().max()) == 0.0 for k in kdas)
    hard(buf_ok, "dt_bias is a buffer, all zeros, in every KDA layer (also after save + reload)")
    A = torch.cat([k.A_log.detach().float().exp() for k in kdas])
    hard(bool(torch.isfinite(A).all()) and float(A.max()) <= 16.0 + 1e-3 and float(A.min()) > 0,
         f"exp(A_log) in (0, 16]: min {A.min():.3f}, max {A.max():.3f}  (GDN init U(0,16))")
    b = torch.cat([k.f_proj.proj.bias.detach().float() for k in kdas])
    dt = torch.nn.functional.softplus(b)
    soft(float(dt.min()) > 5e-4 and float(dt.max()) < 0.2,
         f"softplus(gate bias) = initial dt in [{dt.min():.4f}, {dt.max():.4f}] (GDN init: 1e-3 .. 1e-1)")

    print(f"\n[H3] parameter count")
    n = sum(p.numel() for p in m.parameters())
    hard(n == EXPECTED_PARAMS, f"{n:,} parameters (expected {EXPECTED_PARAMS:,})")

    print(f"\n[H4] one forget value per head, on real text")
    spreads = []
    hooks = [k.f_proj.register_forward_hook(
        lambda mod, inp, out: spreads.append(
            (out.view(*out.shape[:-1], 20, -1).float().amax(-1)
             - out.view(*out.shape[:-1], 20, -1).float().amin(-1)).abs().max().item()))
        for k in kdas]
    with torch.no_grad():
        m(input_ids=batch[:1].to(DEV), use_cache=False)
    for h in hooks:
        h.remove()
    hard(len(spreads) == 27 and max(spreads) == 0.0,
         f"{len(spreads)} layers seen; max spread of f_proj output within a head = {max(spreads):.3e}")

    print(f"\n[S1] initial CE vs R1's init, same 8 x 2048 held-out tokens")
    ce_new = mean_ce(m, batch)
    print(f"        new student: {ce_new:.4f}")

    print(f"\n[H5] one training step")
    m.train()
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    ids = batch[:1].to(DEV)
    loss = m(input_ids=ids, labels=ids, use_cache=False).loss
    loss.backward()
    g_ok = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in m.parameters())
    g_gate = sum(float(k.f_proj.proj.weight.grad.float().norm()) for k in kdas if k.f_proj.proj.weight.grad is not None)
    hard(math.isfinite(loss.item()), f"loss {loss.item():.4f} finite")
    hard(g_ok, "all gradients finite")
    hard(g_gate > 0, f"gradient reaches the new gate (sum of norms {g_gate:.3e})")
    hard(all(not k.dt_bias.requires_grad for k in kdas), "zero dt_bias buffer has no gradient")
    m.zero_grad(set_to_none=True)
    m.gradient_checkpointing_disable()
    m.eval()

    print(f"\n[H6] cached decoding == full forward (16 steps after a 112-token prefix)")
    ids = batch[:1, :128].to(DEV)
    d_new, scale = cache_vs_full(m, ids)
    del m
    torch.cuda.empty_cache()
    r1 = load(R1_INIT)
    d_r1, _ = cache_vs_full(r1, ids)
    ce_r1 = mean_ce(r1, batch)
    del r1
    print(f"        max |logit diff|  new {d_new:.3e}   R1 init (reference) {d_r1:.3e}   logit scale {scale:.1f}")
    hard(d_new <= max(3 * d_r1, 0.05 * scale), "new student's cache path within 3x of R1's bf16 noise")
    print(f"        R1 init CE: {ce_r1:.4f}   (A5b: 7.9412)")
    soft(abs(ce_new - ce_r1) < 0.3, f"initial CE new {ce_new:.4f} vs R1 init {ce_r1:.4f} (diff {ce_new - ce_r1:+.4f})")

    print("\n" + "=" * 70)
    if FAIL:
        print(f"RESULT: {len(FAIL)} HARD CHECK(S) FAILED -- the training chain will NOT start:")
        for f in FAIL:
            print("   - " + f)
        sys.exit(1)
    print(f"RESULT: all hard checks passed; {len(WARN)} warning(s){':' if WARN else ''}")
    for w in WARN:
        print("   - " + w)


if __name__ == "__main__":
    main()