# Environment Setup — replicating the exact `GenAI_FA` conda environment

This guide reproduces the **exact** environment this fork was built and tested on,
so anyone with a comparable NVIDIA GPU gets identical behaviour. Package **versions
matter** — some components (bitsandbytes 4-bit quantization, the CUDA build of
PyTorch) only work together at specific versions, so do **not** "just upgrade to
latest".

## Tested hardware / target

| | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop (8 GB VRAM), Ada / compute capability 8.9 |
| Minimum | Any NVIDIA GPU with **≥ 8 GB VRAM** |
| NVIDIA driver | **581.57** (any driver that supports CUDA 12.8+ works) |
| OS | Windows 11 (the `environment.yml` is win-64; see notes for Linux) |

> You do **not** need to install the CUDA Toolkit separately. PyTorch ships its own
> CUDA 12.8 runtime; you only need a recent **NVIDIA GPU driver**.

## Exact versions (the ones that matter)

| Component | Version |
|---|---|
| Python | **3.12.12** |
| PyTorch | **2.9.1+cu128** (CUDA 12.8 runtime, cuDNN 9.10.2) |
| transformers | **4.57.6** |
| accelerate | **1.11.0** |
| bitsandbytes | **0.48.2** (4-bit / NF4 backend) |
| tokenizers | **0.22.2** |
| safetensors | **0.6.2** |
| huggingface_hub | **0.36.2** |
| numpy | **2.3.3** |
| python-docx | **1.2.0** (.docx text extraction) |

The complete pins are in [`environment.yml`](../environment.yml) (conda) and
[`requirements-lock.txt`](../requirements-lock.txt) (pip).

---

## Method A — one command (recommended, exact)

Reproduces the full environment, name and all, from the committed lockfile:

```bash
conda env create -f environment.yml
conda activate GenAI_FA
```

`environment.yml` already points pip at the PyTorch CUDA 12.8 index, so
`torch==2.9.1+cu128` resolves automatically.

Then install Binoculars itself into the env (without touching the pinned deps):

```bash
pip install -e . --no-deps
```

---

## Method B — step by step (more portable, e.g. Linux)

```bash
# 1) Python 3.12.12, exact
conda create -n GenAI_FA python=3.12.12 -y
conda activate GenAI_FA

# 2) PyTorch with CUDA 12.8 — MUST come from the PyTorch index
pip install torch==2.9.1+cu128 --index-url https://download.pytorch.org/whl/cu128

# 3) Everything else, pinned (also via the cu128 index for torch-adjacent wheels)
pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu128

# 4) Binoculars itself, without re-resolving/downgrading the pinned deps
pip install -e . --no-deps
```

---

## Verify the environment

```bash
python -c "import torch; print('torch', torch.__version__); print('CUDA ok:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
python -c "import bitsandbytes, transformers, docx; print('bnb', bitsandbytes.__version__, '| transformers', transformers.__version__)"
```

Expected:
```
torch 2.9.1+cu128
CUDA ok: True
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
bnb 0.48.2 | transformers 4.57.6
```

## Notes & gotchas

- **Don't downgrade** anything afterwards — the set above is internally consistent.
- **bitsandbytes on Windows**: 0.48.2 ships working CUDA wheels; no manual build needed.
- **Model download**: set `HF_HUB_DISABLE_XET=1` before first run. The Hugging Face
  Xet transfer backend stalled / corrupted shards on some connections; the classic
  HTTP downloader with resume is reliable. The launcher sets this for you.
- **Linux users**: prefer Method B. `environment.yml` lists some win-64-only conda
  packages; Method B avoids them.
