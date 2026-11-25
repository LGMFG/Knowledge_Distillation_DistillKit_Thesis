"""thesis_style.py -- one figure style for every plot in the thesis.
 
    import thesis_style as ts
    ts.use()                                   # rcParams: white, light y-grid, 9 pt, serif, PDF fonts
    fig, ax = plt.subplots(figsize=ts.figsize("half"))
    ts.bars(ax, ["KDA-H", "GDN-H"], [46.4, 28.4], errors=[1.1, 1.0], colors=[ts.C["kda"], ts.C["gdn"]])
    ts.save(fig, "figures/arith_s2.pdf")       # PDF for LaTeX (+ PNG preview with png=True)
 
Rules (decided 11-05, see claude/results_notes.md "Figure style"):
  * no title inside a figure (the LaTeX caption carries it), no claim boxes
  * error bars +-1 SE (or the 95 % CI where the caption says so), in the axis ink colour
  * colour = entity, fixed across the whole thesis (C below); a second factor is a secondary
    encoding (hatch or tint), never a new hue; if a panel needs both layout and corpus, use
    small multiples instead
  * value labels on every bar (author 11-06), placed above the error bar; legend whenever there are >= 2 series
  * one y-axis per panel; different units -> separate panels
 
Palette (convention 11-06: teacher = orange tone, students = blue tones; checked with the dataviz
palette validator against white):
  Teacher (both settings): amber #e0a13c.
  Setting A: base student grey #6b7075 (not distilled), Tulu-3 student light blue #5aa9e6, Dolci
             student strong blue #1f6feb.  (The report's grey #9aa0a6 was too close to the light
             blue: delta-E 10.8, below the 15 floor; #6b7075 gives >= 17 for every pair.)
  Setting B: KDA navy #163f8c, R6 = GDN with KDA's output gate strong blue #1f6feb, GDN light blue
             #5aa9e6 -- dark -> light follows the two one-factor steps KDA -> R6 -> GDN; every pair
             passes the colour-blindness and normal-vision checks (worst delta-E 17).
  Amber and the light blue have < 3:1 contrast on white, so values are printed on every bar
  (value labels, default on) and every figure with >= 2 series has a legend or x-axis labels.
  The same blues mean different students in the two settings; they never share a figure.
"""
from __future__ import annotations
 
import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np
 
# ---------------------------------------------------------------------------- palette
C = {
    # Setting B (students: blue tones, dark -> light = KDA -> R6 -> GDN)
    "kda": "#163f8c",
    "r6": "#1f6feb",
    "gdn": "#5aa9e6",
    # Setting A
    "base": "#6b7075",
    "tulu": "#5aa9e6",
    "dolci": "#1f6feb",
    # both: teacher = orange tone
    "teacher": "#e0a13c",
}
INK = "#6b6a64"        # axes, ticks, error bars
TEXT = "#1f1f1e"       # labels and value labels (text never wears a series colour)
GRID = "#e4e3dd"
SURFACE = "#ffffff"
 
# secondary encodings (a second factor on top of the entity colour)
HATCH = {
    "all_linear": "////",   # layout: solid = hybrid, hatched = all-linear
    "finemath": "xxxx",     # corpus: solid = FineWeb-Edu, hatched = FineMath
    "variant": "....",      # e.g. base student with chat template
}
 
# ---------------------------------------------------------------------------- sizes
TEXTWIDTH_CM = 15.0    # TODO check against the thesis template (\the\textwidth in pt / 28.45)
CM = 1 / 2.54
 
 
def figsize(width: str | float = "full", ratio: float = 0.55) -> tuple[float, float]:
    """Figure size in inches. width = "full" | "half" | width in cm; height = ratio * width."""
    if width == "full":
        w = TEXTWIDTH_CM * CM
    elif width == "half":
        w = (TEXTWIDTH_CM / 2 - 0.4) * CM
    else:
        w = float(width) * CM
    return (w, w * ratio)
 
 
def tint(color: str, amount: float = 0.45) -> str:
    """Mix a colour with white (amount 0 = colour, 1 = white) -- for a light variant of an entity."""
    r, g, b = mcolors.to_rgb(color)
    return mcolors.to_hex((r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount))
 
 
# ---------------------------------------------------------------------------- rcParams
def use(font: str = "serif", size: float = 9) -> None:
    """Apply the thesis style. font = "serif" (matches a LaTeX body) or "sans"."""
    serif = ["Latin Modern Roman", "CMU Serif", "Computer Modern Roman", "DejaVu Serif"]
    sans = ["Latin Modern Sans", "CMU Sans Serif", "Helvetica", "Arial", "DejaVu Sans"]
    mpl.rcParams.update({
        "font.family": "serif" if font == "serif" else "sans-serif",
        "font.serif": serif,
        "font.sans-serif": sans,
        "mathtext.fontset": "cm" if font == "serif" else "dejavusans",
        "font.size": size,
        "axes.labelsize": size,
        "axes.titlesize": size,           # only for panel letters / small-multiple headers
        "xtick.labelsize": size - 1,
        "ytick.labelsize": size - 1,
        "legend.fontsize": size - 1,
        "text.color": TEXT,
        "axes.labelcolor": TEXT,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelcolor": TEXT,
        "ytick.labelcolor": TEXT,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.6,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "lines.linewidth": 1.5,
        "lines.markersize": 4,
        "errorbar.capsize": 2,
        "hatch.linewidth": 0.6,
        "hatch.color": SURFACE,
        "legend.frameon": False,
        "legend.handlelength": 1.2,
        "pdf.fonttype": 42,               # embed TrueType: text stays text in the PDF
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "figure.dpi": 150,
    })
 
 
# ---------------------------------------------------------------------------- marks
def _err_kw():
    return dict(ecolor=INK, elinewidth=0.8, capsize=2, capthick=0.8)
 
 
def bars(ax, labels, values, errors=None, colors=None, hatches=None, width=0.62,
         value_labels: bool | None = None, fmt="{:.1f}", label_offset=0.6):
    """One bar per label, optional +-errors. Value labels are on by default for <= 8 bars."""
    x = np.arange(len(labels))
    colors = colors or [C["teacher"]] * len(labels)
    hatches = hatches or [None] * len(labels)
    for xi, v, c, h in zip(x, values, colors, hatches):
        ax.bar(xi, v, width=width, color=c, hatch=h, linewidth=0, zorder=2)
    if errors is not None:
        ax.errorbar(x, values, yerr=errors, fmt="none", zorder=3, **_err_kw())
    if value_labels is None:
        value_labels = True
    if value_labels:
        errs = errors if errors is not None else [0] * len(labels)
        for xi, v, e in zip(x, values, errs):
            ax.text(xi, v + e + label_offset, fmt.format(v + 1e-9), ha="center", va="bottom",
                    fontsize=mpl.rcParams["xtick.labelsize"], color=TEXT)
    ax.set_xticks(x, labels)
    ax.tick_params(axis="x", length=0)
    return x
 
 
def grouped_bars(ax, groups, series, gap=0.04, group_width=0.8, value_labels=True, fmt="{:.1f}",
                 label_offset=0.6, label_size=None):
    """groups: x categories (e.g. tasks); series: list of dicts with keys
    label, values, [errors], color, [hatch]. Bars inside a group are separated by a small gap."""
    n = len(series)
    x = np.arange(len(groups))
    w = group_width / n
    for i, s in enumerate(series):
        xs = x - group_width / 2 + w * (i + 0.5)
        ax.bar(xs, s["values"], width=w - gap, color=s["color"], hatch=s.get("hatch"),
               linewidth=0, label=s["label"], zorder=2)
        if s.get("errors") is not None:
            ax.errorbar(xs, s["values"], yerr=s["errors"], fmt="none", zorder=3, **_err_kw())
        if value_labels:
            errs = s.get("errors") if s.get("errors") is not None else [0] * len(xs)
            size = label_size or mpl.rcParams["xtick.labelsize"] - 1.5
            for xi, v, e in zip(xs, s["values"], errs):
                ax.text(xi, v + e + label_offset, fmt.format(v + 1e-9), ha="center", va="bottom",
                        fontsize=size, color=TEXT)
    ax.set_xticks(x, groups)
    ax.tick_params(axis="x", length=0)
    return x
 
 
def reference_line(ax, y, label=None, color=None, x_text=None):
    """Dashed horizontal line (teacher ceiling, chance level); label at the right end."""
    color = color or C["teacher"]
    ax.axhline(y, ls=(0, (4, 2)), lw=0.9, color=color, zorder=1)
    if label:
        xlim = ax.get_xlim()
        ax.text(x_text if x_text is not None else xlim[1], y, f" {label}", ha="left" if x_text is None else "center",
                va="center", fontsize=mpl.rcParams["xtick.labelsize"], color=TEXT)
 
 
def legend(ax, **kw):
    """Legend above the plot area, in one row."""
    kw.setdefault("loc", "lower left")
    kw.setdefault("bbox_to_anchor", (0, 1.01))
    kw.setdefault("ncol", 4)
    kw.setdefault("borderaxespad", 0)
    return ax.legend(**kw)
 
 
def save(fig, path: str, png: bool = False):
    """Save as vector PDF (for LaTeX); png=True also writes a 200 dpi preview next to it."""
    fig.savefig(path)
    if png:
        fig.savefig(path.rsplit(".", 1)[0] + ".png", dpi=200)
 
