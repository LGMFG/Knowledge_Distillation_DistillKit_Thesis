#!/usr/bin/env python
"""Paired arithmetic comparisons for R6 (or any pair of models) -- login node, no GPU.

    python r6_compare.py $WORK_LUKAS/eval_results/arith_probe_v2 \
        r1_kda_hybrid:r6_s1_kdahg_hyb r6_s1_kdahg_hyb:r2_gdn_hybrid

Each A:B prints pooled accuracy of A and B and mean(B - A) in points with the 95 % CI of the
paired bootstrap stratified by condition -- the same function and constants (10,000 resamples,
seed 1) as arith_compare.py, which must sit in the same directory.  Names are the "model"
field of the result files; run without pairs to list them.
"""
import sys

from arith_compare import acc, load, paired_boot


def main():
    d, pairs = sys.argv[1], sys.argv[2:]
    data, _ = load(d)
    if not pairs:
        print("models found:", ", ".join(sorted(data)))
        return
    for p in pairs:
        A, B = p.split(":")
        for m in (A, B):
            if m not in data:
                sys.exit(f"{m} not in {d}; found: {', '.join(sorted(data))}")
        keys = [k for k in data[A] if data[A][k]["op"] != "copy" and k in data[B]]
        pa = 100 * acc([data[A][k] for k in keys])[0]
        pb = 100 * acc([data[B][k] for k in keys])[0]
        pt, lo, hi = paired_boot(data[A], data[B], keys)
        print(f"{B:24} - {A:24} pooled {pb:5.1f} - {pa:5.1f} = {pt:+5.1f}  [{lo:+.1f}, {hi:+.1f}]  (n = {len(keys)})")


if __name__ == "__main__":
    main()