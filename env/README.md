# Environments

Two Python 3.11.7 virtual environments on Leonardo (A100 GPUs, CUDA 12.6):

| file | used for |
|---|---|
| `requirements_train.txt` | training (DistillKit), building the students, own evaluation and analysis scripts |
| `requirements_eval.txt` | benchmarks with the LM Evaluation Harness |

Installation, from the repository root (PyTorch first, because FlashAttention
is compiled against it):

```bash
python3.11 -m venv distill_env && source distill_env/bin/activate
pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu126
pip install flash-attn==2.8.3 --no-build-isolation
pip install -r env/requirements_train.txt
```

The evaluation environment is installed the same way with
`requirements_eval.txt`. Keep `flash-linear-attention` at 0.5.2: later versions
handle `cu_seqlens` differently.
