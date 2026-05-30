"""Run Binoculars AI-text detection on a text file, fitted for an 8 GB GPU.

Uses 4-bit (NF4) quantization + sequential model loading (one 7B model on the
GPU at a time). Long documents are split into <=max_token windows and each
window is scored; we report per-chunk scores plus a length-weighted aggregate
and a verdict against both Binoculars thresholds.

Usage:
    python run_detection.py extracted_text.txt [--quant 4bit] [--max-token 512]
"""
import argparse
import sys
import time

import numpy as np
import torch

from binoculars.detector import (
    Binoculars,
    BINOCULARS_ACCURACY_THRESHOLD,
    BINOCULARS_FPR_THRESHOLD,
)


def chunk_by_tokens(tokenizer, text, max_tokens):
    """Split text into a list of strings, each <= max_tokens tokens."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    chunks = []
    for i in range(0, len(ids), max_tokens):
        window = ids[i:i + max_tokens]
        chunks.append(tokenizer.decode(window, skip_special_tokens=True).strip())
    return [c for c in chunks if c]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text_file")
    ap.add_argument("--quant", default="4bit", choices=["4bit", "8bit", "none"])
    ap.add_argument("--max-token", type=int, default=512)
    ap.add_argument("--observer", default="tiiuae/falcon-7b")
    ap.add_argument("--performer", default="tiiuae/falcon-7b-instruct")
    args = ap.parse_args()

    with open(args.text_file, "r", encoding="utf-8") as f:
        text = f.read().strip()

    print(f"[run] loaded text: {len(text)} chars, {len(text.split())} words")
    print(f"[run] quantization={args.quant} sequential=True max_token={args.max_token}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        print(f"[run] GPU: {torch.cuda.get_device_name(0)}")

    t0 = time.time()
    quant = None if args.quant == "none" else args.quant
    bino = Binoculars(
        observer_name_or_path=args.observer,
        performer_name_or_path=args.performer,
        quantization=quant,
        sequential=True,
        trust_remote_code=False,
        max_token_observed=args.max_token,
        mode="low-fpr",
    )
    print(f"[run] detector constructed in {time.time()-t0:.1f}s (tokenizer only; models load on demand)")

    chunks = chunk_by_tokens(bino.tokenizer, text, args.max_token)
    print(f"[run] document split into {len(chunks)} chunk(s) of <= {args.max_token} tokens")

    t1 = time.time()
    scores = bino.compute_score(chunks)  # single sequential load of each model
    if not isinstance(scores, list):
        scores = [scores]
    elapsed = time.time() - t1

    # length-weighted aggregate (weight by chunk token count)
    weights = [len(bino.tokenizer(c, add_special_tokens=False)["input_ids"]) for c in chunks]
    agg = float(np.average(scores, weights=weights))

    print("\n================ RESULTS ================")
    for i, (s, w) in enumerate(zip(scores, weights)):
        verdict = "AI" if s < BINOCULARS_FPR_THRESHOLD else "human"
        print(f"  chunk {i:02d} | tokens={w:4d} | score={s:.4f} | {verdict}")
    print("-----------------------------------------")
    print(f"  aggregate (length-weighted) score: {agg:.4f}")
    print(f"  low-fpr threshold   = {BINOCULARS_FPR_THRESHOLD:.4f}  -> "
          f"{'Most likely AI-generated' if agg < BINOCULARS_FPR_THRESHOLD else 'Most likely human-generated'}")
    print(f"  accuracy threshold  = {BINOCULARS_ACCURACY_THRESHOLD:.4f}  -> "
          f"{'Most likely AI-generated' if agg < BINOCULARS_ACCURACY_THRESHOLD else 'Most likely human-generated'}")
    print(f"  (lower score = more AI-like)")
    print("-----------------------------------------")
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"  peak VRAM allocated: {peak:.2f} GiB")
    print(f"  scoring time: {elapsed:.1f}s")
    print("=========================================")


if __name__ == "__main__":
    sys.exit(main())
