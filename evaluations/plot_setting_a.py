"""Setting-A figures in the thesis style (replace ifeval_bar.pdf and base_eval_bar.pdf of the report).
 
    python plot_setting_a.py            -> figures/fig_a_ifeval.pdf, figures/fig_a_benchmarks.pdf (+ .png previews)
 
Numbers: copied from plot_eval_corrected.ipynb (harness .out files of the practical work), value and SE.
  IFEval prompt-level strict; base student in its native format only (author 11-07: the chat-template
  run of the base student, 0.2680 +- 0.0191, is not shown or discussed in the thesis).
  ARC-C / HellaSwag acc_norm, WinoGrande acc (0-shot, no chat template), MMLU acc (5-shot).
  Tulu MMLU = 0.4784 from the rerun ..._fewshot_mmlu_correct-55398969.out (author 11-07), not the notebook's 0.4660.
TODO before submission: the Dolci IFEval SE (0.0211) is marked "TODO: full-run stderr from .out" in the
notebook -- confirm it from the full Dolci run's .out file.
"""
import os
 
import matplotlib.pyplot as plt
 
import thesis_style as ts
 
OUT = "figures"
os.makedirs(OUT, exist_ok=True)
ts.use()
 
# ---------------------------------------------------------------- IFEval (value, SE), fractions
IFEVAL = [
    ("Base", 0.1553, 0.0156, ts.C["base"], None),
    ("Tulu-3", 0.5009, 0.0215, ts.C["tulu"], None),
    ("Dolci", 0.5970, 0.0211, ts.C["dolci"], None),
    ("Teacher", 0.8373, 0.0159, ts.C["teacher"], None),
]
fig, ax = plt.subplots(figsize=ts.figsize(7.5, ratio=0.72))
ts.bars(ax, [r[0] for r in IFEVAL], [100 * r[1] for r in IFEVAL], errors=[100 * r[2] for r in IFEVAL],
        colors=[r[3] for r in IFEVAL], hatches=[r[4] for r in IFEVAL])
ax.set_ylabel("IFEval prompt-level strict (%)")
ax.set_ylim(0, 100)
ts.save(fig, f"{OUT}/fig_a_ifeval.pdf", png=True)
 
# ---------------------------------------------------------------- general benchmarks
TASKS = ["ARC-C", "HellaSwag", "WinoGrande", "MMLU"]
MODELS = [  # label, colour, [(value, SE) per task]
    ("Base student", ts.C["base"], [(0.3848, 0.0142), (0.5385, 0.0050), (0.5912, 0.0138), (0.5245, 0.0041)]),
    ("Tulu-3 student", ts.C["tulu"], [(0.3558, 0.0140), (0.4895, 0.0050), (0.5667, 0.0139), (0.4784, 0.0041)]),
    ("Dolci student", ts.C["dolci"], [(0.3703, 0.0141), (0.4864, 0.0050), (0.5722, 0.0139), (0.4793, 0.0041)]),
    ("Teacher", ts.C["teacher"], [(0.5853, 0.0144), (0.6904, 0.0046), (0.6796, 0.0131), (0.7219, 0.0036)]),
]
fig, ax = plt.subplots(figsize=ts.figsize("full", ratio=0.42))
ts.grouped_bars(ax, TASKS, [dict(label=m, color=c, values=[100 * v for v, _ in r], errors=[100 * e for _, e in r])
                            for m, c, r in MODELS])
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 85)
ts.legend(ax)
ts.save(fig, f"{OUT}/fig_a_benchmarks.pdf", png=True)
print("wrote", sorted(os.listdir(OUT)))