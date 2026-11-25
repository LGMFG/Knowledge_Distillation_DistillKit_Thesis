#!/usr/bin/env python
"""Held-out KL by POSITION in the document -- is a student weak only at the start of a sequence?

    sbatch --export=ALL,SCRIPT=heldout_kl_positions.py heldout_kl_positions.sbatch     (1 GPU, debug, ~10 min)
    python heldout_kl_positions.py --table                                            (login node: print only)

Why (2026-11-05): R6-S2 looks much worse than R1-S2 / R2-S2 in heldout_kl.py's training-format layout
(LAYOUT=docs: every document is its own sequence, starting from an EMPTY recurrent state and an empty
KV cache), worst on short documents, while DistillKit's own stage-2 eval (packed rows: most documents
start after other documents, so with a non-empty state) puts R6-S2 between R1-S2 and R2-S2.  If the
excess sits in the first tokens of each document, the two evals agree once the start is excluded and
the explanation is a start-of-sequence weakness (as for R2-S2-math and run B).  If R6-S2 is worse at
every position, something else is going on.

Same documents, tokenisation and KL definition as heldout_kl.py with LAYOUT=docs (imported from it):
[BOS] text [EOS] cut at 2048, the first 64 documents with >= 64 tokens per corpus, KL(teacher || student)
at T = 2 on fp32 log-softmax, x T^2.  Position p = index of the PREDICTED token minus 1 (p = 0 is the
prediction made from the BOS token alone).  Per bucket: token-weighted mean KL over all documents.

Results: $WORK_LUKAS/eval_results/heldout_kl_positions/<name>.json; models already done are skipped.
POS_LAYOUT=concat (v2, 11-05): the same profile for heldout_kl.py's concatenated layout (64 windows of
2048 tokens that start mid-text without BOS) -> eval_results/heldout_kl_positions_concat/.
"""
import argparse
import json
import os

POS_LAYOUT = os.environ.get("POS_LAYOUT", "docs")   # docs (training format) or concat (windows start mid-text, no BOS)
assert POS_LAYOUT in ("docs", "concat"), POS_LAYOUT
os.environ["LAYOUT"] = POS_LAYOUT                  # must be set before importing heldout_kl

import torch                                       # noqa: E402

import heldout_kl as H                             # noqa: E402

W, O = H.W, H.O
OUTDIR = f"{W}/eval_results/heldout_kl_positions" + ("" if POS_LAYOUT == "docs" else f"_{POS_LAYOUT}")
EDGES = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]      # bucket b = [EDGES[b], EDGES[b+1])
DEFAULT = [
    ("r1_s2_kda_hyb", f"{O}/r1_stage2_kda_hybrid31_unfreeze/checkpoint-15260"),
    ("r2_s2_gdn_hyb", f"{O}/r2_stage2_gdn_hybrid31_unfreeze/checkpoint-15260"),
    ("r6_s2_kdahg_hyb", f"{O}/r6_stage2_kda_headgate_hybrid31_unfreeze/checkpoint-15260"),
    ("r6_s1_kdahg_hyb", f"{O}/r6_stage1_kda_headgate_hybrid31/checkpoint-4580"),
    ("r2_s1_gdn_hyb", f"{O}/r2_stage1_gdn_hybrid31/checkpoint-4580"),
]


def bucket_of(p):
    for b in range(len(EDGES) - 1):
        if EDGES[b] <= p < EDGES[b + 1]:
            return b
    raise ValueError(p)


def per_position_kl(ls, lt, T=H.T):
    """KL(teacher || student) per position, x T^2; ls / lt: [L, V] fp32 logits."""
    out = []
    for a in range(0, ls.shape[0], 512):
        s = torch.log_softmax(ls[a:a + 512] / T, -1)
        t = torch.log_softmax(lt[a:a + 512] / T, -1)
        out.append((t.exp() * (t - s)).sum(-1) * T * T)
    return torch.cat(out)


@torch.no_grad()
def measure(student, teacher, docs):
    nb = len(EDGES) - 1
    tot, cnt = [0.0] * nb, [0] * nb
    for ids in docs:
        ids = ids.view(1, -1).to(H.DEV)
        ls = student(input_ids=ids, use_cache=False).logits[0, :-1].float()
        lt = teacher(input_ids=ids, use_cache=False).logits[0, :-1].float()
        kl = per_position_kl(ls, lt).cpu()
        for p, v in enumerate(kl.tolist()):
            b = bucket_of(p)
            tot[b] += v
            cnt[b] += 1
        del ls, lt
    return {"mean_kl": [t / c if c else float("nan") for t, c in zip(tot, cnt)], "n_tokens": cnt}


def table():
    rows = [json.load(open(os.path.join(OUTDIR, f))) for f in sorted(os.listdir(OUTDIR)) if f.endswith(".json")]
    if not rows:
        return print(f"no results in {OUTDIR}")
    labels = [f"{EDGES[b]}-{EDGES[b + 1] - 1}" for b in range(len(EDGES) - 1)]
    for c in ("web", "math"):
        print(f"\n{c}: token-weighted mean KL per position bucket (position of the predicting token)")
        print(f"   {'model':20}" + "".join(f"{l:>10}" for l in labels) + f"{'all':>9}{'>=64':>9}")
        for r in rows:
            m, n = r[c]["mean_kl"], r[c]["n_tokens"]
            allv = sum(a * b for a, b in zip(m, n)) / sum(n)
            late = [(a, b) for a, b, e in zip(m, n, EDGES) if e >= 64]
            latev = sum(a * b for a, b in late) / sum(b for _, b in late)
            print(f"   {r['name'][:19]:20}" + "".join(f"{x:10.3f}" for x in m) + f"{allv:9.4f}{latev:9.4f}")
        print(f"   {'tokens':20}" + "".join(f"{x:10d}" for x in rows[0][c]["n_tokens"]))
    print("\nRead: if a model's excess over the others sits in the first buckets and '>=64' matches the order of\n"
          "DistillKit's own eval, the docs-layout difference is a start-of-sequence effect.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    if a.table:
        return table()
    models = DEFAULT
    if os.environ.get("MODELS_FILE"):
        models = [tuple(l.strip().split("=", 1)) for l in open(os.environ["MODELS_FILE"]) if "=" in l]
    todo = [(n, p) for n, p in models
            if not os.path.exists(f"{OUTDIR}/{n}.json") and (os.path.isdir(p) or print(f"skip {n}: no {p}"))]
    print(f"{len(todo)} models to measure: {[n for n, _ in todo]}", flush=True)
    if todo:
        tok = H.AutoTokenizer.from_pretrained(H.TEACHER)
        docs = {c: H.build(tok, p) for c, p in H.CORPORA.items()}
        nobos = sum(int(d[0]) != tok.bos_token_id for ds in docs.values() for d in ds)
        print(f"layout {POS_LAYOUT}: sequences not starting with BOS: {nobos} (expected 0 for docs, 128 for concat)", flush=True)
        teacher = H.load(H.TEACHER)
        for name, path in todo:
            m = H.load(path)
            res = {"name": name, "path": path, "edges": EDGES}
            for c, ds in docs.items():
                res[c] = measure(m, teacher, ds)
                print(f"  {name:22} {c:4} " + " ".join(f"{x:.3f}" for x in res[c]["mean_kl"]), flush=True)
            json.dump(res, open(f"{OUTDIR}/{name}.json", "w"))
            del m
            torch.cuda.empty_cache()
    table()


if __name__ == "__main__":
    main()