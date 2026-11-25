#!/usr/bin/env python
"""Paired standard errors for every arithmetic-probe difference in the thesis.  (11-10)
Login node, CPU only, a few seconds.  Must sit next to arith_compare.py.

    python arith_se.py                       # default: $WORK_LUKAS/eval_results/arith_probe_v2
    python arith_se.py <results dir>
    python arith_se.py <results dir> --list  # only print the model names it finds
    python arith_se.py <results dir> --r6s2 <name>   # if R6's stage-2 file has another name

For each comparison B - A (pooled over all non-copy conditions, as in the thesis) it prints
    pooled A, pooled B, the paired difference, its standard error and the old 95 % CI
  * SE      = analytic, conditions as strata: sqrt(sum_c n_c * var_c) / N, var_c the sample
              variance (ddof 1) of the per-problem differences in condition c   <- use this
  * SE_boot = SD of the paired bootstrap (arith_compare's resampling: 10,000, within
              condition, seed 1) -- a check, should agree with SE to ~0.05
  * CI      = arith_compare.paired_boot's percentile interval, to match the old numbers
and, per model, the pooled accuracy with its binomial SE sqrt(p(1-p)/(n-1)) and n
(for the arithmetic column of tab:results-b-mathcode).
"""
import os
import sys

import numpy as np

from arith_compare import TEACHER, R1, R2, R3, R4, R5, N_BOOT, SEED, acc, load, paired_boot

# (A, B, where the thesis uses B - A)
PAIRS = [
    (R1, R5, "4.2.1  one-epoch stage 1 - stage 1 (KDA hybrid)        old +7.6 [6.1, 9.2]"),
    (R2, R1, "4.2.3  KDA - GDN, hybrid, stage 1                       old +13.6 [11.9, 15.3]"),
    (R4, R3, "4.2.3  KDA - GDN, all-linear, stage 1                   old +10.0 [8.2, 11.7]"),
    ("r2_s2_gdn_hyb", "r1_s2_kda_hyb", "4.2.3  KDA - GDN, hybrid, S2 FineWeb    old +17.9 [16.2, 19.7]"),
    ("r4_s2_gdn_only", "r3_s2_kda_only", "4.2.3  KDA - GDN, all-linear, S2 FineWeb  old +9.3 [MISSING]"),
    ("r2_s2math_gdn_hyb", "r1_s2math_kda_hyb", "4.2.3  KDA - GDN, hybrid, S2 FineMath   old +9.5 [7.6, 11.4]"),
    ("r4_s2math_gdn_only", "r3_s2math_kda_only", "4.2.3  KDA - GDN, all-linear, S2 FineMath old +5.5 [3.7, 7.3]"),
    ("r1_s2math_kda_hyb", "r1fm_s2math_kda_hyb", "4.2.2  FineMath both stages - FineWeb S1 + FineMath S2  old +0.6 [-1.1, 2.2]"),
    ("r3_s2math_kda_only", "r1_s2math_kda_hyb", "4.2.4  hybrid - all-linear, KDA, S2 FineMath  old +4.7 [2.9, 6.5]"),
    ("r4_s2math_gdn_only", "r2_s2math_gdn_hyb", "4.2.4  hybrid - all-linear, GDN, S2 FineMath  old +0.7 [-1.3, 2.6]"),
    (R2, "r6_s1_kdahg_hyb", "Tab 4.9 output-gate step, stage 1 (GDN -> GDN+KDA gate)  old +4.3 [2.7, 6.0]"),
    ("r6_s1_kdahg_hyb", R1, "Tab 4.9 forget-gate step, stage 1 (GDN+KDA gate -> KDA)  old +9.2 [7.7, 10.8]"),
    ("r2_s2_gdn_hyb", "R6S2", "Tab 4.9 output-gate step, stage 2  old +6.6 [5.0, 8.2]"),
    ("R6S2", "r1_s2_kda_hyb", "Tab 4.9 forget-gate step, stage 2  old +11.4 [9.7, 13.0]"),
]
R6S2_DEFAULT = "r6_s2_kdahg_hyb"   # guess; check with --list


def strat_se(A, B, keys):
    by = {}
    for k in keys:
        by.setdefault(k[0], []).append(int(B[k]["correct"]) - int(A[k]["correct"]))
    N = sum(len(v) for v in by.values())
    var = sum(len(v) * np.var(v, ddof=1) for v in by.values() if len(v) > 1)
    return 100 * np.sqrt(var) / N


def boot_sd(A, B, keys, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    by = {}
    for k in keys:
        by.setdefault(k[0], []).append(int(B[k]["correct"]) - int(A[k]["correct"]))
    N = sum(len(v) for v in by.values())
    sums = np.zeros(n)
    for v in by.values():
        v = np.asarray(v)
        sums += v[rng.integers(0, len(v), size=(n, len(v)))].sum(1)
    return float(np.std(100 * sums / N, ddof=1))


def main():
    args = [a for a in sys.argv[1:]]
    r6s2 = R6S2_DEFAULT
    if "--r6s2" in args:
        i = args.index("--r6s2")
        r6s2 = args[i + 1]
        del args[i:i + 2]
    only_list = "--list" in args
    args = [a for a in args if a != "--list"]
    d = args[0] if args else os.path.join(
        os.environ.get("WORK_LUKAS", "/leonardo_work/EUHPC_D34_049/WORK_LUKAS"), "eval_results", "arith_probe_v2")
    data, _ = load(d)
    if only_list or TEACHER not in data:
        print(f"results: {os.path.abspath(d)}\nmodels: " + ", ".join(sorted(data)))
        if TEACHER not in data:
            sys.exit("no teacher file here -- pass the arith_probe_v2 results directory")
        return
    keys = [k for k, r in data[TEACHER].items() if r["op"] != "copy"]

    def pooled(m):
        return acc([data[m][k] for k in keys])

    print(f"results: {os.path.abspath(d)}   pooled over {len(keys)} non-copy problems "
          f"({len({k[0] for k in keys})} conditions)\n")
    print("B - A in points: diff, SE (use this), SE_boot (check), old-style 95 % CI\n")
    for a, b, why in PAIRS:
        a = r6s2 if a == "R6S2" else a
        b = r6s2 if b == "R6S2" else b
        print(why)
        miss = [m for m in (a, b) if m not in data]
        if miss:
            print(f"    -- missing {miss}  (see --list)\n")
            continue
        ks = [k for k in keys if k in data[a] and k in data[b]]
        p, lo, hi = paired_boot(data[a], data[b], ks)
        se, sb = strat_se(data[a], data[b], ks), boot_sd(data[a], data[b], ks)
        print(f"    {b} - {a}:  {100 * pooled(b)[0]:.1f} - {100 * pooled(a)[0]:.1f}   "
              f"diff {p:+.2f}   SE {se:.2f}   SE_boot {sb:.2f}   CI [{lo:+.1f}, {hi:+.1f}]   n = {len(ks)}\n")

    print("\npooled accuracy per model, binomial SE sqrt(p(1-p)/(n-1)) in points")
    for m in sorted(data):
        p, n = pooled(m)
        if n:
            print(f"    {m:28} {100 * p:5.1f} +- {100 * np.sqrt(p * (1 - p) / (n - 1)):.2f}   (n = {n})")


if __name__ == "__main__":
    main()