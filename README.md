# <img src="./assets/bino-logo.svg" width=40 style="padding-top: 0px"/> Binoculars (8 GB GPU edition) — Local AI-Generated Text Detector

> A fork of [**Binoculars**](https://github.com/ahans30/Binoculars) by Hans et al.,
> extended to run **fully on a single consumer NVIDIA GPU with as little as 8 GB VRAM**
> (e.g. an RTX 4060) via 4-bit quantization — and to score real documents
> (`.docx`) out of the box, with a one-command launcher.

<p align="center">
  <img src="assets/binoculars.jpg" width="260" height="260" alt="Binoculars with a Falcon on top">
</p>

## Why this matters

The internet is now flooded with AI-generated text, and telling machine-written
content from human writing is increasingly important — for **educators marking
student work, editors, reviewers, hiring teams, journalists, and anyone verifying a
document's authenticity**.

Binoculars is a strong, **training-free** detector (see the [paper](https://arxiv.org/abs/2401.12070)).
But the original needs to hold **two 7B models in GPU memory at once** (~28 GB) —
out of reach for most people's hardware. **This fork brings it back into everyone's
hands:** it runs locally on an ordinary 8 GB gaming laptop GPU, so checking a
document is **simple, fast, free, and fully private** — your file never leaves your
machine (no upload, no API, no cloud).

## What's different in this fork

- **4-bit (NF4) quantization** + **sequential model loading** — only one 7B model is
  resident at a time, so the canonical `falcon-7b` / `falcon-7b-instruct` pair fits in
  **~7.8 GB of VRAM** instead of ~28 GB.
- **Document support**: extracts text from **`.docx`** (paragraphs *and* tables)
  and **`.pdf`** (digital PDFs, via pypdf), including files currently open in
  Word / synced by OneDrive.
- **One-command launcher** (`ai_generated_checked.ps1`) and a CLI runner
  (`run_detection.py`) that splits long documents into windows, scores each, and
  reports a length-weighted verdict.
- **Reproducible environment**: exact, pinned `environment.yml` / `requirements-lock.txt`
  (see [docs/ENVIRONMENT_SETUP.md](docs/ENVIRONMENT_SETUP.md)).
- The detection method, models, and thresholds are **unchanged** from upstream — this
  fork only changes *how* the models are loaded so they fit a small GPU.

## Requirements

- An NVIDIA GPU with **≥ 8 GB VRAM** (tested on RTX 4060 Laptop, 8 GB).
- A recent NVIDIA driver (CUDA 12.8-capable). No separate CUDA Toolkit needed.
- ~28 GB free disk for the model weights (downloaded once, on first run).

## Installation

1. **Create the exact environment** — full step-by-step in
   [docs/ENVIRONMENT_SETUP.md](docs/ENVIRONMENT_SETUP.md):
   ```bash
   conda env create -f environment.yml
   conda activate GenAI_FA
   ```
2. **Install Binoculars** into that env without changing the pinned deps:
   ```bash
   pip install -e . --no-deps
   ```

## Usage

### Easiest — the launcher (Windows / PowerShell)

```powershell
# Show help (supported file types, options):
.\ai_generated_checked.ps1

# Check a document:
.\ai_generated_checked.ps1 "C:\path\to\my essay.docx"
```

**Supported input for extraction:** `.docx` (Word 2007+) and `.pdf` (digital
PDFs). Plain `.txt` / `.md` are read as-is. Legacy `.doc` / `.odt` / `.rtf` and
**scanned/image-only PDFs** (no text layer — they'd need OCR) should be converted
to `.docx` or a text-based `.pdf` first.

### CLI runner (any OS)

```bash
# .docx / .pdf -> text, then score:
python scripts/extract_docx.py "my essay.docx" extracted_text.txt
python scripts/extract_pdf.py  "my paper.pdf"  extracted_text.txt
python run_detection.py extracted_text.txt --quant 4bit
```

### Library (the original API still works)

```python
from binoculars import Binoculars

# 4-bit, one model on the GPU at a time -> fits 8 GB:
bino = Binoculars(quantization="4bit", sequential=True)
print(bino.compute_score("Some text to test..."))
print(bino.predict("Some text to test..."))
```

### Reading the result

Each scoring window (≤384 tokens by default, tuned to stay under 8 GB VRAM; the
script also sets `PYTORCH_ALLOC_CONF=expandable_segments:True`) gets a
**Binoculars score**; the tool reports a length-weighted
aggregate and a verdict against two thresholds (`0.8536` low-false-positive /
`0.9015` accuracy). **Lower = more AI-like.** Below the threshold → *"Most likely
AI-generated"*; above → *"Most likely human-generated"*.

## Limitations

No detector is perfect. Binoculars works best on **English** prose and is intended as
**decision support, not proof** — always keep a human in the loop. A high score means
the text *doesn't look machine-generated*; it is not a guarantee of authorship. 4-bit
quantization adds tiny score noise versus the bf16-calibrated thresholds (negligible
unless a result is already borderline).

## Credits & license

This is a community fork. **All credit for the method and the original implementation
goes to the Binoculars authors.** Original repository:
[ahans30/Binoculars](https://github.com/ahans30/Binoculars) · Paper:
[arXiv:2401.12070](https://arxiv.org/abs/2401.12070). Licensed under the terms in
[LICENSE.md](LICENSE.md) (unchanged from upstream).

If you use this work, please cite the original paper:

```bibtex
@misc{hans2024spotting,
      title={Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text},
      author={Abhimanyu Hans and Avi Schwarzschild and Valeriia Cherepanova and Hamid Kazemi and Aniruddha Saha and Micah Goldblum and Jonas Geiping and Tom Goldstein},
      year={2024},
      eprint={2401.12070},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
