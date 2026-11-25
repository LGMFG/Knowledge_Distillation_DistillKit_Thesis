#!/usr/bin/env python
"""Decay figure for Results 4.2.3 (v2, 11-10).  CPU only, login node.

    cp thesis_style.py plot_decay_thesis.py $WORK_LUKAS/afm_kda/      # next to each other
    python plot_decay_thesis.py                 # reads eval_results/decay_analysis/*.json
    python plot_decay_thesis.py --list          # only print the result names it finds
    python plot_decay_thesis.py --a ratio       # panel (a) as the p90/p10 ratio instead of the band

Writes figures/fig_b_decay.pdf (+ .png).  Web held-out, after stage 1, hybrids only.
  (a) time scales inside a KDA head: per layer, the 10th to 90th percentile of the 128 channel
      half-lives of a head (shaded) and their median (line), each the median over the 20 heads;
      grey: the same at initialization.  Channel half-life = 10^(mean over tokens of log10 hl).
  (b) change of a head's decay from token to token: SD over tokens of the head's mean log10
      half-life (in decades, i.e. powers of ten), median over heads, for the KDA hybrid, GDN with
      KDA's output gate and the GDN hybrid.
v2 vs v1: no titles inside the figure (thesis_style rule), legends below the panels, all-linear
student and the init lines for it dropped (numbers stay in the text), (a) shows the band of
half-lives instead of only their ratio.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import thesis_style as ts

W = os.environ.get("WORK_LUKAS", "/leonardo_work/EUHPC_D34_049/WORK_LUKAS")


def load(d):
    out = {}
    for f in sorted(os.listdir(d)):
        if f.endswith(".json"):
            r = json.load(open(os.path.join(d, f)))
            out[r["name"]] = r
    return out


def layers(r, corpus):
    L = r[corpus]["layers"]
    return sorted(int(k) for k in L), L


def channel_band(r, corpus):
    """per layer: median over heads of the p10 / p50 / p90 channel half-life (tokens)."""
    ls, L = layers(r, corpus)
    lo, mid, hi = [], [], []
    for l in ls:
        hl = 10 ** np.asarray(L[str(l)]["channel_mean_y"], dtype=float)      # [H, K]
        q = np.percentile(hl, [10, 50, 90], axis=1)                         # [3, H]
        lo.append(np.median(q[0])); mid.append(np.median(q[1])); hi.append(np.median(q[2]))
    return ls, np.array(lo), np.array(mid), np.array(hi)


def per_layer(r, corpus, key, f=lambda x: x):
    ls, L = layers(r, corpus)
    return ls, [float(np.median([f(x) for x in L[str(l)][key]])) for l in ls]


def panel_label(ax, s):
    ax.text(-0.13, 1.03, s, transform=ax.transAxes, fontweight="bold", va="bottom", ha="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=f"{W}/eval_results/decay_analysis")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--corpus", default="web", choices=["web", "math"])
    ap.add_argument("--r1", default="r1_s1_kda_hyb")
    ap.add_argument("--r1init", default="r1_init_kda_hyb")
    ap.add_argument("--r2", default="r2_s1_gdn_hyb")
    ap.add_argument("--r6", default="r6_s1_kdahg_hyb")
    ap.add_argument("--r3", default="r3_s1_kda_only")
    ap.add_argument("--r3init", default="r3_init_kda_only")
    ap.add_argument("--a", default="ratio", choices=["ratio", "band"])
    ap.add_argument("--b", default="factor", choices=["factor", "decades"])
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    R = load(a.dir)
    if a.list:
        return print("\n".join(sorted(R)))
    missing = [n for n in (a.r1, a.r1init, a.r2, a.r6) if n not in R]
    if missing:
        raise SystemExit(f"missing {missing}; available: {sorted(R)}  (pass names with --r1/--r1init/--r2/--r6)")

    ts.use()
    os.makedirs(a.out, exist_ok=True)
    fig, (p1, p2) = plt.subplots(1, 2, figsize=ts.figsize("full", ratio=0.58))
    grey = "#9a9a94"

    # (a) time scales inside a KDA head
    if a.a == "band":
        x, lo, mid, hi = channel_band(R[a.r1init], a.corpus)
        p1.plot(x, lo, color=grey, ls=(0, (4, 2)), lw=1.1, zorder=3, label="initialization: 10th and 90th percentile")
        p1.plot(x, hi, color=grey, ls=(0, (4, 2)), lw=1.1, zorder=3)
        x, lo, mid, hi = channel_band(R[a.r1], a.corpus)
        p1.fill_between(x, lo, hi, color=ts.C["kda"], alpha=0.18, lw=0, zorder=1,
                        label="after stage 1: 10th to 90th percentile")
        p1.plot(x, mid, color=ts.C["kda"], lw=1.4, marker="o", ms=2.5, zorder=2, label="after stage 1: median")
        p1.set_ylabel("channel half-life (tokens)")
    else:
        # initialization: R1 and R3 draw dt_bias from the same distribution, so one grey line
        # stands for both layouts; the medians of both are printed as a check
        x, y = per_layer(R[a.r1init], a.corpus, "channel_ratio_p90_p10")
        p1.plot(x, y, color=grey, ls=(0, (4, 2)), lw=1.2, label="KDA, initialization (both layouts)")
        if a.r3init in R:
            y3 = per_layer(R[a.r3init], a.corpus, "channel_ratio_p90_p10")[1]
            print(f"init p90/p10, median over layers: hybrid {np.median(y):.1f}, all-linear {np.median(y3):.1f}")
        x, y = per_layer(R[a.r1], a.corpus, "channel_ratio_p90_p10")
        p1.plot(x, y, color=ts.C["kda"], marker="o", ms=2.5, lw=1.4, label="KDA hybrid, after stage 1")
        if a.r3 in R:
            x, y = per_layer(R[a.r3], a.corpus, "channel_ratio_p90_p10")
            p1.plot(x, y, color=ts.C["kda"], ls=(0, (4, 2)), marker="s", ms=2.8, mfc="white", lw=1.2,
                    label="KDA all-linear, after stage 1")
        else:
            print(f"skip {a.r3} (not found)")
        p1.hlines(1.0, min(x), max(x), color=ts.C["gdn"], lw=1.4, label="GDN: one decay per head (ratio 1)")
        p1.set_ylabel("spread of channel half-lives\nwithin a head (p90 / p10)")
    p1.set_yscale("log")
    p1.set_xlabel("layer")
    panel_label(p1, "(a)")
    p1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=1, frameon=False)

    # (b) token-to-token variation of a head's decay
    for name, label, c in ((a.r1, "KDA hybrid", ts.C["kda"]), (a.r6, "GDN with KDA's output gate", ts.C["r6"]),
                           (a.r2, "GDN hybrid", ts.C["gdn"])):
        x, y = per_layer(R[name], a.corpus, "var_token_headmean", lambda v: np.sqrt(max(v, 0.0)))
        if a.b == "factor":            # geometric SD: 10^(SD of log10 half-life)
            y = [10 ** v for v in y]
        p2.plot(x, y, color=c, marker="o", ms=2.5, lw=1.4, label=label)
    p2.set_xlabel("layer")
    if a.b == "factor":
        p2.set_ylim(bottom=1)
        p2.set_ylabel("variation of a head's half-life\nover tokens (factor)")
    else:
        p2.set_ylim(bottom=0)
        p2.set_ylabel("SD over tokens (decades)")
    panel_label(p2, "(b)")
    p2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=1, frameon=False)

    fig.tight_layout(w_pad=3.5)
    ts.save(fig, os.path.join(a.out, "fig_b_decay.pdf"), png=True)
    print("wrote", os.path.join(a.out, "fig_b_decay.pdf"))


if __name__ == "__main__":
    main()