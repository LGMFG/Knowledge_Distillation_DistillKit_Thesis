#!/usr/bin/env python
"""Figures and a markdown table for O1 (decay-value analysis).  CPU only, numpy + matplotlib.

    python plot_decay.py                         # reads $WORK_LUKAS/eval_results/decay_analysis/*.json
    python plot_decay.py --dir DIR --out FIGDIR  # any input / output directory
    python plot_decay.py --corpus math           # the FineMath held-out set instead of web

Writes into FIGDIR (default DIR/figs), v2 (10-30, after the first real results):
  decay_heatmap.{pdf,png}      layer x head TYPICAL half-life (geometric mean over tokens and channels; exact,
                               not binned), one panel per student, stage 2 web; colour range from the data
  decay_cdf.{pdf,png}          cumulative share of decay values with half-life <= x, per student, one line per
                               stage (v1 plotted a density, which is spiky because KDA's decays are mostly a
                               fixed per-channel spectrum -- correct but unreadable)
  decay_spread.{pdf,png}       per layer: (a) spread of half-lives ACROSS the channels of a head (p90/p10;
                               GDN = 1 by construction), (b) how much a head's decay changes FROM TOKEN TO
                               TOKEN (SD over tokens of the head-mean log10 half-life) -- comparable for KDA and GDN
  decay_kda_channels.{pdf,png} R1-S2 / R3-S2: every channel's typical half-life, sorted within its head
  decay_table.md               summary table + how much each training stage changed the gates

Missing models are skipped with a message, so the script runs on a partial result directory.
"""
import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

W = os.environ.get("WORK_LUKAS", "/leonardo_work/EUHPC_D34_049/WORK_LUKAS")
Y_LO, Y_HI, Y_STEP = -1.0, 7.0, 0.05                      # must match decay_analysis.py
EDGES = np.arange(Y_LO, Y_HI + Y_STEP / 2, Y_STEP)

# validated categorical slots 1 / 2 (dataviz reference palette): mechanism = hue
KDA, GDN = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"
# single-hue sequential ramp (blue 100 -> 700) for magnitude
SEQ = ListedColormap(["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
                      "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"])

STUDENTS = [("r1", "R1 KDA hybrid", "kda_hyb", KDA), ("r2", "R2 GDN hybrid", "gdn_hyb", GDN),
            ("r3", "R3 KDA all-linear", "kda_only", KDA), ("r4", "R4 GDN all-linear", "gdn_only", GDN)]
STAGES = [("init", "initialisation", ":"), ("s1", "stage 1", "--"), ("s2", "stage 2 web", "-"),
          ("s2math", "stage 2 FineMath", "-.")]
# stage = line style AND lightness of the mechanism's hue (ordinal: later stage = darker)
SHADE = {"init": 0.35, "s1": 0.6, "s2": 1.0, "s2math": 1.0}


def shade(hex_col, stage):
    """Mix the hue with white; s2math keeps full colour but is told apart by its line style."""
    rgb = np.array([int(hex_col[i:i + 2], 16) for i in (1, 3, 5)]) / 255
    a = SHADE[stage]
    return tuple(a * rgb + (1 - a) * np.ones(3))


def stage_legend(ax):
    from matplotlib.lines import Line2D
    ax.legend([Line2D([], [], color=INK, ls=ls, lw=2) for _, _, ls in STAGES], [l for _, l, _ in STAGES],
              fontsize=7, frameon=False)


def name(rid, stage, tag):
    return f"{rid}_{stage}_{tag}"


def load(d):
    out = {}
    for f in sorted(os.listdir(d)):
        if f.endswith(".json"):
            r = json.load(open(os.path.join(d, f)))
            out[r["name"]] = r
    return out


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.label.set_color(INK); ax.xaxis.label.set_color(INK)


def save(fig, out, stem):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{stem}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.pdf/.png")


def cdf(layers):
    """Cumulative share of all decay values with log10 half-life <= upper bin edge."""
    h = np.sum([np.sum(v["hist"], axis=0) for v in layers.values()], axis=0).astype(float)
    c = np.cumsum(h) / h.sum()
    # entry i covers (EDGES[i-1], EDGES[i]]; index 0 = underflow (<= EDGES[0]), last = overflow
    x = np.concatenate([EDGES, [Y_HI + Y_STEP]])
    return x, c


def per_layer(r, corpus, key, agg=np.median):
    L = r[corpus]["layers"]
    ls = sorted(int(k) for k in L)
    return ls, [agg(L[str(l)][key]) for l in ls]


def decade_ticks(ax, lo, hi, axis="x"):
    t = list(range(int(lo), int(hi) + 1))
    lab = [{0: "1", 1: "10", 2: "100", 3: "1k", 4: "10k", 5: "100k", 6: "1M"}.get(i, f"1e{i}") for i in t]
    (ax.set_xticks if axis == "x" else ax.set_yticks)(t)
    (ax.set_xticklabels if axis == "x" else ax.set_yticklabels)(lab)


def colour_range(vals):
    v = np.asarray([x for x in vals if np.isfinite(x)])
    lo, hi = np.floor(np.percentile(v, 2) * 2) / 2, np.ceil(np.percentile(v, 98) * 2) / 2
    return lo, max(hi, lo + 1.0)


def add_cbar(fig, axes, im, lo, hi, label):
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    t = np.arange(np.ceil(lo), np.floor(hi) + 1)
    cb.set_ticks(t); cb.set_ticklabels([f"{10 ** x:,.0f}" if x >= 0 else f"{10 ** x:g}" for x in t])
    cb.set_label(label, fontsize=8); cb.ax.tick_params(labelsize=8)


# ---------------------------------------------------------------- figures
def fig_heatmap(R, corpus, stage, out):
    have = [(s, R.get(name(s[0], stage, s[2]))) for s in STUDENTS]
    have = [(s, r) for s, r in have if r]
    if not have:
        return print(f"heatmap: no {stage} models")
    imgs = []
    for _, r in have:
        L = r["num_layers"]
        H = len(next(iter(r[corpus]["layers"].values()))["mean_y"])
        img = np.full((L, H), np.nan)
        for k, v in r[corpus]["layers"].items():
            img[int(k)] = v["mean_y"]
        imgs.append(img)
    lo, hi = colour_range(np.concatenate([i.ravel() for i in imgs]))
    fig, axes = plt.subplots(1, len(have), figsize=(3.0 * len(have), 4.4), sharey=True, squeeze=False)
    cm = SEQ.copy(); cm.set_bad("#f3f2ee")
    for ax, ((rid, label, _, _), _), img in zip(axes[0], have, imgs):
        im = ax.imshow(img, aspect="auto", cmap=cm, vmin=lo, vmax=hi, interpolation="nearest")
        ax.set_title(label, fontsize=9, color=INK)
        ax.set_xlabel("head", fontsize=8)
        style(ax)
    axes[0][0].set_ylabel("layer  (empty rows = softmax attention)", fontsize=8)
    add_cbar(fig, axes[0].tolist(), im, lo, hi, "typical half-life (tokens)")
    st = dict((s[0], s[1]) for s in STAGES)[stage]
    fig.suptitle(f"How long each head's forget gate keeps information ({st}, {corpus} held-out)\n"
                 "one square = one head; colour = its typical half-life (geometric mean over tokens); darker = longer memory.\n"
                 "KDA squares average the head's 128 channels, which hides their spread -- see decay_kda_channels",
                 fontsize=9, color=INK, y=1.07)
    save(fig, out, "decay_heatmap")


def fig_cdf(R, corpus, out):
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.1), sharey=True)
    any_ = False
    for ax, (rid, label, tag, col) in zip(axes, STUDENTS):
        # stage 1 and stage 2 FineMath lie on top of stage 2 web (|Δ| <= 0.09 decades per channel, see the
        # stage-change table); drawing all four made the legend promise lines nobody could see
        for st, ls in (("init", ":"), ("s2", "-")):
            r = R.get(name(rid, st, tag))
            if not r:
                continue
            any_ = True
            x, c = cdf(r[corpus]["layers"])
            ax.step(x, c, where="post", ls=ls, lw=2, color=shade(col, st))
        ax.axvline(math.log10(2048), color=MUTED, lw=1, ls=(0, (1, 2)))
        ax.text(math.log10(2048) + 0.05, 0.04, "2,048 =\ntraining length", fontsize=6.5, color=MUTED)
        ax.set_xlim(-0.5, 5); ax.set_ylim(0, 1.0)
        decade_ticks(ax, 0, 5)
        ax.set_title(label, fontsize=9, color=INK); ax.set_xlabel("half-life (tokens)", fontsize=8)
        ax.grid(color=GRID, lw=0.6); style(ax)
    if not any_:
        plt.close(fig); return print("cdf: no models")
    axes[0].set_ylabel("share of decay values with half-life ≤ x", fontsize=8)
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], color=MUTED, ls=":", lw=2), Line2D([], [], color=INK, ls="-", lw=2)],
               ["initialisation", "after training (stage 2 web; stage 1 and stage 2 FineMath lie on top of it)"],
               fontsize=8, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.12))
    fig.suptitle(f"How long the forget gates keep information: all decay values (tokens × channels), {corpus} held-out.\n"
                 "Read: at x = 100, the curve gives the share of decay values that halve a stored item within 100 tokens.",
                 fontsize=9, color=INK, y=1.07)
    save(fig, out, "decay_cdf")


def fig_spread(R, corpus, out):
    from matplotlib.lines import Line2D
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.4))
    for rid, label, tag, col in STUDENTS:
        for st, ls in (("init", ":"), ("s2", "-")):
            r = R.get(name(rid, st, tag))
            if not r:
                continue
            if tag.startswith("kda"):
                ls_, v = per_layer(r, corpus, "channel_ratio_p90_p10")
                a1.plot(ls_, np.log10(v), ls=ls, lw=2, color=shade(col, st), marker="o" if "only" in tag else "s",
                        ms=3.5, mfc="white" if "only" in tag else None)
            L = r[corpus]["layers"]
            ls_ = sorted(int(k) for k in L)
            sd = [np.median(np.sqrt(np.clip(L[str(l)]["var_token_headmean"], 0, None))) for l in ls_]
            a2.plot(ls_, sd, ls=ls, lw=2, color=shade(col, st), marker="o" if "only" in tag else "s", ms=3.5,
                    mfc="white" if "only" in tag else None)
    a1.axhline(0, color=GDN, lw=1.5)
    a1.text(0.5, 0.05, "GDN: one gate per head → ratio 1", color=GDN, fontsize=7)
    a1.set_ylim(bottom=-0.1)
    yt = range(0, int(np.ceil(a1.get_ylim()[1])) + 1)
    a1.set_yticks(list(yt)); a1.set_yticklabels([f"{10 ** t:,}×" for t in yt])
    a1.set_title("(a) spread ACROSS the 128 channels of a KDA head\n(p90/p10 of channel half-lives, median over heads)",
                 fontsize=9, color=INK)
    a2.set_title("(b) change FROM TOKEN TO TOKEN of a head's decay\n(SD over tokens of log10 half-life, median over heads)",
                 fontsize=9, color=INK)
    a2.set_ylabel("decades", fontsize=8)
    for ax in (a1, a2):
        ax.set_xlabel("layer", fontsize=8); ax.grid(axis="y", color=GRID, lw=0.6); style(ax)
    h = [Line2D([], [], color=KDA, marker="s", lw=2), Line2D([], [], color=KDA, marker="o", mfc="white", lw=2),
         Line2D([], [], color=GDN, marker="s", lw=2), Line2D([], [], color=GDN, marker="o", mfc="white", lw=2),
         Line2D([], [], color=INK, ls=":", lw=2), Line2D([], [], color=INK, ls="-", lw=2)]
    fig.legend(h, ["KDA hybrid", "KDA all-linear", "GDN hybrid", "GDN all-linear", "initialisation", "stage 2 web"],
               fontsize=8, frameon=False, ncol=6, loc="lower center", bbox_to_anchor=(0.5, -0.1))
    fig.suptitle(f"Where the decay variation lives ({corpus} held-out)", fontsize=10, color=INK, y=1.06)
    save(fig, out, "decay_spread")


def fig_kda_channels(R, corpus, out):
    have = [(lab, R.get(name(rid, "s2", tag))) for rid, lab, tag, _ in STUDENTS if tag.startswith("kda")]
    have = [(lab, r) for lab, r in have if r]
    if not have:
        return print("kda_channels: no KDA stage-2 models")
    imgs = []
    for lab, r in have:
        L = r["num_layers"]
        rows = [None] * L
        for k, v in r[corpus]["layers"].items():
            rows[int(k)] = np.sort(np.array(v["channel_mean_y"]), axis=1).reshape(-1)
        width = max(len(x) for x in rows if x is not None)
        img = np.full((L, width), np.nan)
        for l, x in enumerate(rows):
            if x is not None:
                img[l] = x
        imgs.append(img)
    lo, hi = colour_range(np.concatenate([i.ravel() for i in imgs]))
    fig, axes = plt.subplots(1, len(have), figsize=(6 * len(have), 4.2), squeeze=False)
    cmap = SEQ.copy(); cmap.set_bad("#f3f2ee")
    for ax, (lab, r), img in zip(axes[0], have, imgs):
        im = ax.imshow(img, aspect="auto", cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
        width = img.shape[1]
        if width % 20 == 0:
            K = width // 20
            for h in range(1, 20):
                ax.axvline(h * K - 0.5, color="white", lw=0.4)
            ax.set_xticks([h * K + K / 2 for h in range(0, 20, 4)]); ax.set_xticklabels(range(0, 20, 4))
        ax.set_xlabel("head (its 128 channels sorted from short to long half-life)", fontsize=8)
        ax.set_ylabel("layer", fontsize=8); ax.set_title(f"{lab}, stage 2 web", fontsize=9, color=INK)
        style(ax)
    add_cbar(fig, axes[0].tolist(), im, lo, hi, "channel's typical half-life (tokens)")
    fig.suptitle("Each KDA head holds a spectrum of time scales: light = forgets within a few tokens, "
                 "dark = remembers long", fontsize=9, color=INK, y=1.03)
    save(fig, out, "decay_kda_channels")


# ---------------------------------------------------------------- table
def frac(layers, lo, hi):
    h = np.sum([np.sum(v["hist"], axis=0) for v in layers.values()], axis=0).astype(float)
    lo_i = 0 if lo is None else int(np.searchsorted(EDGES, math.log10(lo), side="left"))
    hi_i = len(h) if hi is None else int(np.searchsorted(EDGES, math.log10(hi), side="left"))
    return h[lo_i:hi_i].sum() / h.sum()


def stage_change(a, b, corpus):
    """Mean |difference| in decades between two checkpoints: per head (mean_y) and per channel."""
    La, Lb = a[corpus]["layers"], b[corpus]["layers"]
    dh = [abs(x - y) for k in La for x, y in zip(La[k]["mean_y"], Lb[k]["mean_y"])]
    dc = [abs(x - y) for k in La for ra, rb in zip(La[k]["channel_mean_y"], Lb[k]["channel_mean_y"])
          for x, y in zip(ra, rb)]
    return np.mean(dh), np.mean(dc)


def table(R, corpus, out):
    lines = ["| model | typical half-life (geo. mean) | median head half-life | < 16 | 16–256 | 256–2k | > 2k "
             "| channel p90/p10 | channel share | static share | token SD (decades) | β | gate check |",
             "|---" * 13 + "|"]
    order = [name(rid, st, tag) for rid, _, tag, _ in STUDENTS for st, _, _ in STAGES]
    for n in sorted(R, key=lambda n: order.index(n) if n in order else 99):
        r = R[n]
        if corpus not in r:
            continue
        L = r[corpus]["layers"]
        geo = np.mean([m for v in L.values() for m in v["mean_y"]])
        med = np.median([m for v in L.values() for m in v["median_y"]])
        cf = np.mean([c for v in L.values() for c in v["channel_fraction"]])
        ss = [c for v in L.values() for c, w in zip(v["static_share"], v["var_within_token"]) if w > 1e-12]
        rat = np.median([x for v in L.values() for x in v["channel_ratio_p90_p10"]])
        tsd = np.median([math.sqrt(max(x, 0)) for v in L.values() for x in v["var_token_headmean"]])
        beta = np.mean([b for v in L.values() for b in v["beta_mean"]])
        chk = ", ".join(f"{m} {'✅' if (c or {}).get('ok') else '⚠️'}" for m, c in r.get("gate_check", {}).items())
        lines.append(f"| {n} | {10 ** geo:.1f} | {10 ** med:.1f} | {frac(L, None, 16):.2f} | {frac(L, 16, 256):.2f} | "
                     f"{frac(L, 256, 2048):.2f} | {frac(L, 2048, None):.2f} | {rat:.0f} | {cf:.3f} | "
                     f"{np.mean(ss) if ss else float('nan'):.2f} | {tsd:.2f} | {beta:.2f} | {chk} |")
    ch = ["", "**How much each stage changed the gates** (mean |Δ log10 half-life| in decades; 0.1 decade = ×1.26):", "",
          "| student | init → stage 1, per head | per channel | stage 1 → stage 2 web, per head | per channel "
          "| stage 1 → stage 2 FineMath, per head | per channel |", "|---" * 7 + "|"]
    for rid, label, tag, _ in STUDENTS:
        g = {st: R.get(name(rid, st, tag)) for st, _, _ in STAGES}
        cells = []
        for a, b in (("init", "s1"), ("s1", "s2"), ("s1", "s2math")):
            if g[a] and g[b]:
                h, c = stage_change(g[a], g[b], corpus)
                cells += [f"{h:.3f}", f"{c:.3f}"]
            else:
                cells += ["–", "–"]
        ch.append(f"| {label} | " + " | ".join(cells) + " |")
    note = ("\n*Half-life = ln 2 / (−g) tokens, forget gate only (an upper bound on memory: the delta rule also "
            "overwrites). Typical half-life = 10^(mean log10 half-life) over layers, heads, tokens and channels (exact). "
            "Median head half-life is read from 0.05-decade bins, so it moves in steps of ×1.12 (8.4, 9.4, 10.6, 11.9, 13.3 …). "
            "< 16 … > 2k = share of all decay values (tokens × channels) per band. Channel p90/p10 = ratio of the 90th "
            "to the 10th percentile channel half-life within a head (median over heads; 1 for GDN). Channel share = "
            "E_t Var_k(log h) / Var(log h): how much of a head's decay variation lies between its channels rather than "
            "between tokens. Static share = fraction of the channel spread that is fixed per channel. Token SD = SD over "
            "tokens of the head-mean log10 half-life (median over heads): how input-dependent the gate is. "
            "β = mean write strength sigmoid(b).*\n")
    path = os.path.join(out, "decay_table.md")
    open(path, "w").write(f"### O1 decay analysis — {corpus} held-out\n\n" + "\n".join(lines) + "\n" + note
                          + "\n".join(ch) + "\n")
    print(f"wrote decay_table.md ({len(lines) - 2} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=f"{W}/eval_results/decay_analysis")
    ap.add_argument("--out", default=None)
    ap.add_argument("--corpus", default="web", choices=["web", "math"])
    ap.add_argument("--heatmap-stage", default="s2", choices=[s[0] for s in STAGES])
    a = ap.parse_args()
    out = a.out or os.path.join(a.dir, "figs" if a.corpus == "web" else "figs_math")
    os.makedirs(out, exist_ok=True)
    R = load(a.dir)
    print(f"{len(R)} result files: {sorted(R)}")
    fig_heatmap(R, a.corpus, a.heatmap_stage, out)
    fig_cdf(R, a.corpus, out)
    fig_spread(R, a.corpus, out)
    fig_kda_channels(R, a.corpus, out)
    table(R, a.corpus, out)


if __name__ == "__main__":
    main()