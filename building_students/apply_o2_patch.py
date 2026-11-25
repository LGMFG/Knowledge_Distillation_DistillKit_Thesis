#!/usr/bin/env python
"""O2 -- add "KDA with GDN's forget gate" to the afm_kda repo, as a config flag.

    cd $WORK_LUKAS/afm_kda
    python apply_o2_patch.py            # edits the three files in place, keeps *.bak_o2 copies
    python apply_o2_patch.py --check    # only reports whether every anchor is found

What changes (and why nothing else does)
----------------------------------------
configuration_afm_kda.py   new field  kda_forget_gate: "channel" (default = KDA as before) | "head"
modeling_afm_kda.py        HeadForgetGate + make_forget_gate_per_head(): when the flag is "head",
                           the KDA layer's forget-gate projection f_proj (2560 -> 128 -> 2560, one
                           value per CHANNEL) is replaced by GDN's gate projection (2560 -> 20, one
                           value per HEAD, plus a per-head bias that plays GDN's dt_bias), repeated
                           over the 128 channels of each head; KDA's own per-channel dt_bias becomes
                           a frozen all-zero buffer.  The kernel then computes
                               g = -exp(A_log[h]) * softplus(a_proj(x)[h] + dt[h])
                           identically for the 128 channels of head h -- exactly GDN's forget gate,
                           inside an otherwise unchanged KDA layer (KDA kernel, KDA output gate).
                           init_new_parameters(): GDN's init for this gate -- A_log ~ log U(0,16),
                           dt ~ logU(1e-3, 1e-1) per head (KDA's own layers keep U(1,16)).
build_student.py           new variant hybrid31_headgate (= hybrid31 + the flag), parameter
                           arithmetic, flag recorded in build_info.json.

With the flag at its default every code path is the one R1-R4 ran.  Trained checkpoints are
untouched anyway: each one carries its own copy of the .py files.

Every edit is an exact-string replacement whose anchor must occur exactly once; if your file
differs from the version this was written against, the script stops and changes nothing in
that file.  Already-patched files are detected and skipped.
"""
import argparse
import os
import py_compile
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------ configuration_afm_kda.py
CFG = [
    ("""    gdn_use_gate: bool = True       # GDN-only; KDA's output gate is not optional
""",
     """    gdn_use_gate: bool = True       # GDN-only; KDA's output gate is not optional

    # --- O2 (2026-10-30): KDA with GDN's forget gate.  "channel" = KDA as published (one decay
    # per key channel, 2560 per layer); "head" = one decay per head (20 per layer), computed
    # exactly as GDN computes it.  Default keeps every existing config.json meaning what it meant.
    kda_forget_gate: str = "channel"   # "channel" | "head"
"""),
    ("""        if self.mechanism not in ("kda", "gdn"):
            raise ValueError(
                f"mechanism must be 'kda' or 'gdn', got {self.mechanism!r}")
""",
     """        if self.mechanism not in ("kda", "gdn"):
            raise ValueError(
                f"mechanism must be 'kda' or 'gdn', got {self.mechanism!r}")
        # (also checked in the model: from_pretrained kwargs arrive AFTER __post_init__)
        if self.kda_forget_gate not in ("channel", "head"):
            raise ValueError(
                f"kda_forget_gate must be 'channel' or 'head', got {self.kda_forget_gate!r}")
"""),
]

# ------------------------------------------------------------------ modeling_afm_kda.py
MOD = [
    # 1. the gate module, inserted before the KDA adapter
    ("""class AFMKDALinearAttention(nn.Module):
""",
     '''class HeadForgetGate(nn.Module):
    """O2: GDN's forget-gate projection inside a KDA layer.

    GDN:  g[h]    = -exp(A_log[h]) * softplus(a_proj(x)[h] + dt_bias[h])        one value per head
    KDA:  g[h, k] = -exp(A_log[h]) * softplus(f_proj(x)[h, k] + dt_bias[h, k])  one per channel

    This module stands in for KDA's f_proj: a full-rank Linear(hidden -> H) whose bias plays GDN's
    per-head dt_bias, repeated over the K channels of each head.  KDA's own dt_bias is replaced by
    an all-zero buffer (see make_forget_gate_per_head), so the kernel computes GDN's gate for all
    K channels of a head.  Parameters: hidden*H + H = 51,220 per layer -- exactly GDN's a_proj +
    dt_bias -- instead of KDA's 655,360 + 2,560.
    """

    def __init__(self, hidden_size: int, num_heads: int, head_k_dim: int):
        super().__init__()
        self.head_k_dim = head_k_dim
        self.proj = nn.Linear(hidden_size, num_heads, bias=True)

    def forward(self, x):
        # [..., H] -> [..., H*K] in "(h d)" order, which is how KDA rearranges f_proj's output
        return self.proj(x).repeat_interleave(self.head_k_dim, dim=-1)


def make_forget_gate_per_head(kda: "KimiDeltaAttention") -> None:
    """Turn one fla KimiDeltaAttention into KDA-with-GDN's-forget-gate, in place."""
    H, K = kda.num_v_heads, kda.head_k_dim
    assert kda.gate_dim == H * K, (kda.gate_dim, H, K)
    kda.f_proj = HeadForgetGate(kda.hidden_size, H, K)
    # KDA adds dt_bias[h, k] inside the kernel.  A per-channel PARAMETER would let training untie
    # the channels again, so it becomes a frozen zero BUFFER (persistent: saved and reloaded, and
    # checked by test_o2_headgate.py); the per-head bias lives in HeadForgetGate.proj.bias.
    del kda.dt_bias
    kda.register_buffer("dt_bias", torch.zeros(H * K), persistent=True)


def is_head_gate(module) -> bool:
    return isinstance(getattr(module, "f_proj", None), HeadForgetGate)


class AFMKDALinearAttention(nn.Module):
'''),
    # 2. the swap, right after the KDA layer is built
    ("""            mode="chunk",          # fla asserts chunk mode during training
            layer_idx=layer_idx,
        )
""",
     """            mode="chunk",          # fla asserts chunk mode during training
            layer_idx=layer_idx,
        )
        # O2.  Checked HERE because a from_pretrained kwarg bypasses the config's __post_init__.
        forget_gate = getattr(config, "kda_forget_gate", "channel")
        if forget_gate == "head":
            make_forget_gate_per_head(self.attn)
        elif forget_gate != "channel":
            raise ValueError(f"kda_forget_gate must be 'channel' or 'head', got {forget_gate!r}")
"""),
    # 3. init: KDA's scheme must not touch the head-gate layers ...
    ("""        if isinstance(module, KimiDeltaAttention):
            if f"{name}.A_log" in missing:
""",
     """        if isinstance(module, KimiDeltaAttention):
            if is_head_gate(module):
                continue        # O2: GDN's scheme, in the block at the end of this function
            if f"{name}.A_log" in missing:
"""),
    # 4. ... which get GDN's scheme after the generic loop (which zeroes every new bias)
    ("""        else:                            # 1-D norm weights (o_norm)
            nn.init.ones_(p)
""",
     """        else:                            # 1-D norm weights (o_norm)
            nn.init.ones_(p)

    # ---- O2: KDA layers with GDN's forget gate get GDN's initialisation for that gate.
    # Runs AFTER the generic loop, which has drawn proj.weight ~ N(0, std) and zeroed proj.bias.
    for name, module in model.named_modules():
        if not (isinstance(module, KimiDeltaAttention) and is_head_gate(module)):
            continue
        if f"{name}.f_proj.proj.weight" not in missing:
            continue                     # loaded from a checkpoint: leave it alone
        module.A_log.copy_(torch.empty_like(module.A_log).uniform_(0, 16).log())   # GDN: U(0,16)
        b = module.f_proj.proj.bias                                                # = GDN dt_bias
        dt = torch.exp(
            torch.empty(b.shape, dtype=torch.float32, device=b.device).uniform_(0, 1)
            * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        b.copy_((dt + torch.log(-torch.expm1(-dt))).to(b.dtype))
        module.dt_bias.zero_()           # buffer; may hold uninitialised memory after a meta init
"""),
]

# ------------------------------------------------------------------ build_student.py
BLD = [
    ("""GATE_PARAMS = 6_553_600       # gate_proj, 2560 x 2560
""",
     """GATE_PARAMS = 6_553_600       # gate_proj, 2560 x 2560
# O2: KDA layer with GDN's forget gate.  Removed: f_proj 2560*128 + 128*2560 = 655,360 and
# dt_bias 2,560 (becomes a buffer); added: proj 2560*20 + 20 = 51,220.  27,612,308 -> 27,005,608.
HEADGATE_SAVING = 655_360 + 2_560 - 51_220      # = 606,700 per KDA layer
"""),
    ("""    # ---- Track C: the GDN sibling.""",
     """    # ---- O2 (2026-10-30): R1 with GDN's forget gate -- one decay per head, GDN's projection,
    # ---- GDN's init.  Everything else (layout, KDA kernel, KDA output gate, transfer) = hybrid31.
    "hybrid31_headgate": dict(
        dirname="afm_kda_hybrid31_headgate_init",
        kda_forget_gate="head",
        full_attn=list(range(3, N_LAYERS, 4)),          # 3,7,11,...,35
        use_rope=False, use_attn_gate=True,
    ),
    # ---- Track C: the GDN sibling."""),
    ("""    mechanism = spec.get("mechanism", "kda")
""",
     """    mechanism = spec.get("mechanism", "kda")
    forget_gate = spec.get("kda_forget_gate", "channel")
    assert forget_gate == "channel" or mechanism == "kda", "kda_forget_gate applies to KDA only"
"""),
    ("""    if args.variant in ("hybrid31", "gdn_hybrid31") and args.full_attn is None:""",
     """    if args.variant in ("hybrid31", "gdn_hybrid31", "hybrid31_headgate") and args.full_attn is None:"""),
    ("""    print(f"transfer q/k/v/o : {not args.no_transfer}")
""",
     """    print(f"transfer q/k/v/o : {not args.no_transfer}")
    print(f"kda_forget_gate  : {forget_gate}")
"""),
    ("""        mechanism=mechanism,                   # <-- new
""",
     """        mechanism=mechanism,                   # <-- new
        kda_forget_gate=forget_gate,           # O2
"""),
    ("""    assert cfg.mechanism == mechanism, cfg.mechanism
""",
     """    assert cfg.mechanism == mechanism, cfg.mechanism
    assert cfg.kda_forget_gate == forget_gate, cfg.kda_forget_gate
"""),
    ("""    expected = (teacher_params + n_kda * LINEAR_DELTA[mechanism]
""",
     """    expected = (teacher_params
                + n_kda * (LINEAR_DELTA[mechanism] - (HEADGATE_SAVING if forget_gate == "head" else 0))
"""),
    ("""            "use_attn_gate": spec["use_attn_gate"],
        },
""",
     """            "use_attn_gate": spec["use_attn_gate"],
            "kda_forget_gate": forget_gate,
        },
"""),
]

FILES = [("configuration_afm_kda.py", CFG, "kda_forget_gate"),
         ("modeling_afm_kda.py", MOD, "class HeadForgetGate"),
         ("build_student.py", BLD, "HEADGATE_SAVING")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.environ.get("AFM_KDA_SRC", HERE))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    ok = True
    plans = []
    for fname, edits, marker in FILES:
        path = os.path.join(a.repo, fname)
        if not os.path.isfile(path):
            print(f"!! {path} not found"); ok = False; continue
        src = open(path).read()
        if marker in src:
            print(f"   {fname}: already patched (found {marker!r}) -- skipped"); continue
        new = src
        for i, (old, rep) in enumerate(edits, 1):
            n = new.count(old)
            if n != 1:
                print(f"!! {fname}: anchor {i}/{len(edits)} found {n} times (need exactly 1):\n"
                      + "\n".join("     | " + l for l in old.splitlines()[:4]))
                ok = False
                break
            new = new.replace(old, rep)
        else:
            plans.append((path, src, new))
            print(f"   {fname}: {len(edits)} edits ready")
    if not ok:
        sys.exit("NOTHING WRITTEN: send the file(s) named above, the anchors do not match your version")
    if a.check:
        print("check only: nothing written"); return
    for path, src, new in plans:
        shutil.copy(path, path + ".bak_o2")
        open(path, "w").write(new)
        py_compile.compile(path, doraise=True)
        print(f"   wrote {path}  (backup {os.path.basename(path)}.bak_o2; compiles)")
    print("done.  Next: git diff, git commit, then the build + test jobs (o2_submit_chain.sh)")


if __name__ == "__main__":
    main()