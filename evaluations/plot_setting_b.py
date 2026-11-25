"""Setting-B figures in the thesis style (draft 11-09).

    python plot_setting_b.py   -> figures/fig_b_tokens.pdf, figures/fig_b_speed.pdf (+ .png previews)

Numbers copied from claude/results.md:
  token curve  : §7 "Token curve" (FineWeb-Edu stage 2) and "Arithmetic token curve, FineMath vs FineWeb"
                 (arithmetic probe pooled, KDA hybrid, checkpoints every 1,526 steps = 100 M tokens)
  speed        : §8 prefill table (batch 1) and claude/speed_table.md (decode, batch 1/8/32/64, cache MiB);
                 panel (b) shows batch 8 only (11-11; the other batch sizes are quoted in the text)
The decay figure needs the per-layer JSON files on the cluster -> plot_decay_thesis.py.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

import thesis_style as ts

OUT = "figures"
os.makedirs(OUT, exist_ok=True)
ts.use()

# ------------------------------------------------------------------ token curve (KDA hybrid, arithmetic pooled)
tok = np.arange(0, 11) * 100            # million tokens of stage 2
fineweb = [38.3, 38.1, 38.8, 39.8, 42.5, 42.6, 40.6, 44.6, 42.2, 45.4, 46.4]
finemath = [38.3, 52.9, 61.3, 59.4, 67.2, 68.2, 71.5, 71.6, 74.0, 76.4, 77.0]
fig, ax = plt.subplots(figsize=ts.figsize(9.5, ratio=0.62))
ax.plot(tok, finemath, color=ts.C["kda"], marker="o", label="stage 2 on FineMath")
ax.plot(tok, fineweb, color=ts.C["kda"], marker="o", ls=(0, (4, 2)), mfc="white", label="stage 2 on FineWeb-Edu")
ts.reference_line(ax, 91.0, label="teacher")
ax.set_xlabel("stage-2 tokens (millions)")
ax.set_ylabel("arithmetic probe, pooled (%)")
ax.set_ylim(30, 100)
ax.set_xlim(-20, 1020)
ax.set_xticks(range(0, 1001, 200))
ts.legend(ax, ncol=2)
ts.save(fig, f"{OUT}/fig_b_tokens.pdf", png=True)

# ------------------------------------------------------------------ speed
N = [128, 2048, 8192, 16384, 32768, 65536, 131072]
prefill = {  # ms, batch 1
    "teacher": [31, 98, 396, 850, 2057, 5756, 18366],
    "KDA hybrid": [77, 111, 417, 822, 1710, 3828, 9371],
    "GDN hybrid": [62, 109, 411, 810, 1686, 3777, 9275],
    "KDA all-linear": [92, 115, 424, 813, 1595, 3184, 6342],
    "GDN all-linear": [73, 113, 416, 798, 1565, 3127, 6228],
}
STYLE = {"KDA hybrid": (ts.C["kda"], "-"), "GDN hybrid": (ts.C["gdn"], "-"),
         "KDA all-linear": (ts.C["kda"], (0, (4, 2))), "GDN all-linear": (ts.C["gdn"], (0, (4, 2)))}
# decode: (ms per token, cache MiB) for batch 1/8/32/64 at 2k/8k/32k/64k/128k (speed_table.md); None = OOM
dec = {
    "teacher": {1: [(31.0, 148), (31.0, 576), (31.0, 2304), (34.2, 4608), (41.5, 9216)],
                8: [(29.6, 1156), (29.5, 4612), (56.3, 18436), (97.6, 36868), None],
                32: [(30.2, 4624), (56.9, 18448), None, None, None],
                64: [(39.3, 9247), (110.7, 36895), None, None, None]},
    "KDA hybrid": {1: [(46.8, 79), (46.8, 187), (46.4, 619), (47.6, 1195), (49.0, 2347)],
                   8: [(48.0, 575), (48.1, 1439), (48.2, 4895), (48.2, 9503), (56.1, 18719)],
                   32: [(47.6, 2298), (47.4, 5754), (57.4, 19579), (98.9, 38019), None],
                   64: [(47.0, 4618), (47.2, 11529), (113.7, 39178), None, None]},
    "GDN hybrid": {1: [(44.7, 79), (44.6, 187), (44.6, 619), (45.3, 1195), (47.1, 2347)],
                   8: [(45.8, 575), (45.5, 1439), (45.8, 4895), (45.7, 9503), (56.0, 18719)],
                   32: [(47.4, 2299), (46.4, 5754), (57.2, 19579), (98.6, 38019), None],
                   64: [(45.9, 4618), (45.7, 11529), (113.5, 39178), None, None]},
    "KDA all-linear": {1: [(51.7, 57), (51.5, 57), (51.9, 57), (51.8, 57), (51.3, 57)],
                       8: [(54.4, 381), (54.2, 381), (54.1, 381), (54.1, 381), (54.2, 381)],
                       32: [(53.9, 1523), (54.1, 1523), (53.6, 1523), (53.5, 1523), (53.8, 1523)],
                       64: [(53.8, 3073), (53.5, 3073), (53.6, 3073), (53.5, 3073), (53.6, 3073)]},
    "GDN all-linear": {1: [(49.5, 57), (49.3, 57), (49.3, 57), (49.4, 57), (49.5, 57)],
                       8: [(51.5, 381), (51.4, 381), (51.6, 381), (51.2, 381), (51.2, 381)],
                       32: [(52.1, 1524), (51.8, 1523), (51.5, 1523), (51.3, 1523), (52.0, 1523)],
                       64: [(51.6, 3073), (51.6, 3073), (51.8, 3073), (51.7, 3073), (51.3, 3073)]},
}
fig, (a1, a2) = plt.subplots(1, 2, figsize=ts.figsize("full", ratio=0.42))


def panel_label(ax, s):
    ax.text(-0.14, 1.03, s, transform=ax.transAxes, fontweight="bold", va="bottom", ha="left")


# (a) prefill speed-up at batch size 1
for m, (c, ls) in STYLE.items():
    lin = "all-linear" in m
    a1.plot(N, np.array(prefill["teacher"]) / np.array(prefill[m]), color=c, ls=ls, marker="s" if lin else "o",
            ms=3.5, mfc="white" if lin else c, label=m)
a1.axhline(1.0, color=ts.C["teacher"], ls=(0, (4, 2)), lw=0.9, zorder=1)
a1.text(140, 1.05, "teacher", fontsize=7, color=ts.TEXT)
a1.set_xscale("log", base=2)
a1.set_xticks([128, 1024, 8192, 65536], ["128", "1k", "8k", "64k"])
a1.set_xlabel("context length $N$ (tokens)")
a1.set_ylabel("prefill speed-up over the teacher")
panel_label(a1, "(a)")

# (b) decode at batch size 8: time per generated token vs context length; x = out of memory (11-11)
CTX = [2048, 8192, 32768, 65536, 131072]
B = 8
for m in ["teacher", *STYLE]:
    c, ls = (ts.C["teacher"], "-") if m == "teacher" else STYLE[m]
    lin = "all-linear" in m
    pts = [(n, p[0]) for n, p in zip(CTX, dec[m][B]) if p is not None]
    a2.plot([p[0] for p in pts], [p[1] for p in pts], color=c, ls=ls, marker="s" if lin else "o", ms=3.5,
            mfc="white" if lin else c)
    if len(pts) < len(CTX):                      # first context length that did not fit
        n_oom = CTX[len(pts)]
        a2.plot([n_oom], [pts[-1][1]], marker="x", color=c, ms=6, ls="none")
        a2.annotate("out of memory", (n_oom, pts[-1][1]), xytext=(4, 7), textcoords="offset points",
                    ha="right", fontsize=7, color=ts.TEXT)
a2.set_xscale("log", base=2)
a2.set_xticks([2048, 8192, 32768, 131072], ["2k", "8k", "32k", "128k"])
a2.set_xlabel("context length $N$ (tokens)")
a2.set_ylabel("time per generated token (ms)")
panel_label(a2, "(b)")

from matplotlib.lines import Line2D
h, l = a1.get_legend_handles_labels()
th = Line2D([], [], color=ts.C["teacher"], marker="o", ls="-", ms=3.5)
fig.legend([th] + h, ["teacher"] + l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False)
fig.tight_layout()
ts.save(fig, f"{OUT}/fig_b_speed.pdf", png=True)
print("ok")