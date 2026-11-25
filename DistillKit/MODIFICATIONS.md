# Modifications to DistillKit

This folder contains [DistillKit](https://github.com/arcee-ai/DistillKit)
(Apache License 2.0) as of upstream commit `a61c2e7d25c2`, with changes by
Lukas Rosenberger made for the thesis experiments. Only the library files are
included; the run configurations are in `../run_configurations/`. Each modified file carries
a one-line notice at the top.

## Modified files

- `distillkit/main.py`
  - `_load_dataset`: conversations are formatted (`_format_row`) with
    `num_proc=os.cpu_count()` instead of a single process.
  - `load_student_model`: receives the tokenizer and sets the student's
    `pad_token_id`, `eos_token_id` and `bos_token_id` (in `config` and
    `generation_config`) to the tokenizer's, so that student and tokenizer
    agree on the special tokens. `do_distill` passes the tokenizer.
  - A comment on the `attn_implementation` line (no change in behavior).
- `distillkit/monkey_patch_packing.py`: `AutoConfig.from_pretrained(...,
  trust_remote_code=True)`, so that the packing patch can read the
  configuration of a remote-code student (the `afm_kda` model).

The upstream `.gitignore` is included unchanged.
