"""Run Binoculars AI-text detection on a text file, fitted for an 8 GB GPU.

Uses 4-bit (NF4) quantization + sequential model loading (one 7B model on the
GPU at a time). Long documents are split into <=max_token windows and each
window is scored; the result is shown as a colored table, a headline verdict,
and a visual position on the AI<->human classification scale.

Usage:
    python run_detection.py extracted_text.txt [--quant 4bit] [--max-token 384] [--no-color]
"""
import argparse
import os
import sys
import time

# Give CUDA room near the 8 GB ceiling: expandable segments cut fragmentation and
# OOM risk. Must be set before torch initialises CUDA (override-able by the env).
# torch >= 2.6 renamed PYTORCH_CUDA_ALLOC_CONF -> PYTORCH_ALLOC_CONF.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

from binoculars.detector import (
    Binoculars,
    BINOCULARS_ACCURACY_THRESHOLD,   # 0.9015 - "accuracy" decision line
    BINOCULARS_FPR_THRESHOLD,        # 0.8536 - conservative low-false-positive line
)

# Make non-ASCII output (box drawing, icons) safe on any console code page,
# and enable ANSI colour processing on Windows terminals.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
if os.name == "nt":
    os.system("")  # turns on VT/ANSI escape handling in modern Windows consoles

# Interpretive band edges (the two thresholds above are the official ones).
AI_STRONG_EDGE = 0.75
HUMAN_STRONG_EDGE = 1.00
SCALE_LO, SCALE_HI = 0.70, 1.10


class C:
    """ANSI colour codes (blanked out when colour is disabled)."""
    RESET = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
    RED = "\033[91m"; ORANGE = "\033[38;5;208m"; YELLOW = "\033[93m"
    GREEN = "\033[32m"; GREEN_B = "\033[1;92m"; CYAN = "\033[96m"; GREY = "\033[90m"

    @classmethod
    def disable(cls):
        for k in vars(cls):
            if k.isupper():
                setattr(cls, k, "")


def classify(score):
    """Return (icon, label, colour, action) for a Binoculars score."""
    if score < AI_STRONG_EDGE:
        return "🔴", "Very likely AI", C.RED, "Treat as AI-written"
    if score < BINOCULARS_FPR_THRESHOLD:
        return "🟠", "Likely AI", C.ORANGE, "Probably AI - verify if stakes are high"
    if score < BINOCULARS_ACCURACY_THRESHOLD:
        return "🟡", "Borderline / uncertain", C.YELLOW, "Human review recommended"
    if score < HUMAN_STRONG_EDGE:
        return "🟢", "Likely human", C.GREEN, "Treat as human-written"
    return "✅", "Very likely human", C.GREEN_B, "Strong human signal"


def _band_colour(value):
    if value < AI_STRONG_EDGE:                 return C.RED
    if value < BINOCULARS_FPR_THRESHOLD:       return C.ORANGE
    if value < BINOCULARS_ACCURACY_THRESHOLD:  return C.YELLOW
    if value < HUMAN_STRONG_EDGE:              return C.GREEN
    return C.GREEN_B


def _pos(score, width):
    frac = (score - SCALE_LO) / (SCALE_HI - SCALE_LO)
    return max(0, min(width - 1, round(frac * (width - 1))))


def mini_bar(score, width=12):
    """A small colour-banded gauge with a marker at the score position."""
    p = _pos(score, width)
    out = []
    for i in range(width):
        v = SCALE_LO + (i / (width - 1)) * (SCALE_HI - SCALE_LO)
        ch = "┃" if i == p else "─"
        out.append(_band_colour(v) + ch + C.RESET)
    return "".join(out)


def render_scale(agg, width=46):
    """A wide colour gradient bar with a ▲ marker under the aggregate score."""
    bar = "".join(
        _band_colour(SCALE_LO + (i / (width - 1)) * (SCALE_HI - SCALE_LO)) + "█" + C.RESET
        for i in range(width)
    )
    p = _pos(agg, width)
    icon, label, colour, _ = classify(agg)
    pad = " " * 5  # aligns under the bar (after the "0.70 " prefix)
    marker = pad + " " * p + C.BOLD + "▲" + C.RESET
    caption = pad + " " * p + f"{colour}{C.BOLD}{agg:.4f}  {icon} {label}{C.RESET}"
    return (f"{C.GREY}0.70 {C.RESET}{bar}{C.GREY} 1.10{C.RESET}\n"
            f"{marker}\n{caption}")


def rule(width=58, colour=None):
    return (colour or C.GREY) + "─" * width + C.RESET


def print_results(scores, weights, agg, *, quant, peak_gib, elapsed):
    icon, label, colour, action = classify(agg)

    # ---- Headline verdict ----
    print()
    print(rule(58, C.CYAN))
    print(f"  {icon}  {colour}{C.BOLD}{label.upper()}{C.RESET}"
          f"    {C.DIM}aggregate score{C.RESET} {C.BOLD}{agg:.4f}{C.RESET}")
    print(f"      {C.DIM}{action}.  Lower score = more AI-like.{C.RESET}")
    print(rule(58, C.CYAN))

    # ---- Per-chunk table ----
    print(f"\n {C.BOLD}{'Chunk':>5}  {'Tokens':>6}  {'Score':>8}   {'Gauge':<12}   Classification{C.RESET}")
    print(f" {C.GREY}{'-'*5}  {'-'*6}  {'-'*8}   {'-'*12}   {'-'*22}{C.RESET}")
    for i, (s, w) in enumerate(zip(scores, weights)):
        ic, lab, col, _ = classify(s)
        print(f" {i:>5}  {w:>6}  {col}{s:8.4f}{C.RESET}   {mini_bar(s)}   {ic} {col}{lab}{C.RESET}")

    # ---- Visual scale ----
    print(f"\n {C.BOLD}Where this document sits on the scale{C.RESET}"
          f"  {C.GREY}(AI  <───────────>  human){C.RESET}")
    print(render_scale(agg))

    # ---- Legend / classification bands ----
    print(f"\n {C.BOLD}Classification bands{C.RESET}")
    bands = [
        ("🔴", C.RED,     f"score < {AI_STRONG_EDGE:.2f}", "Very likely AI"),
        ("🟠", C.ORANGE,  f"{AI_STRONG_EDGE:.2f} - {BINOCULARS_FPR_THRESHOLD:.3f}", "Likely AI"),
        ("🟡", C.YELLOW,  f"{BINOCULARS_FPR_THRESHOLD:.3f} - {BINOCULARS_ACCURACY_THRESHOLD:.3f}", "Borderline (review)"),
        ("🟢", C.GREEN,   f"{BINOCULARS_ACCURACY_THRESHOLD:.3f} - {HUMAN_STRONG_EDGE:.2f}", "Likely human"),
        ("✅", C.GREEN_B, f"score >= {HUMAN_STRONG_EDGE:.2f}", "Very likely human"),
    ]
    for ic, col, rng, desc in bands:
        print(f"   {ic} {col}{rng:<16}{C.RESET} {col}{desc}{C.RESET}")
    print(f"   {C.GREY}official thresholds: {BINOCULARS_FPR_THRESHOLD:.4f} (low-fpr) / "
          f"{BINOCULARS_ACCURACY_THRESHOLD:.4f} (accuracy){C.RESET}")

    # ---- Footer / run info ----
    print(f"\n {C.DIM}quant={quant} | sequential | "
          f"{'peak VRAM %.2f GiB | ' % peak_gib if peak_gib is not None else ''}"
          f"scored in {elapsed:.1f}s{C.RESET}")
    print(f" {C.DIM}Not proof of authorship - decision support only; best on English text.{C.RESET}\n")


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
    ap.add_argument("--max-token", type=int, default=384)  # 8 GB-VRAM-friendly default
    ap.add_argument("--observer", default="tiiuae/falcon-7b")
    ap.add_argument("--performer", default="tiiuae/falcon-7b-instruct")
    ap.add_argument("--no-color", action="store_true", help="disable ANSI colours")
    args = ap.parse_args()

    if args.no_color or not sys.stdout.isatty():
        # keep colours when piped to a real terminal via the launcher; users can
        # force-disable with --no-color.
        if args.no_color:
            C.disable()

    with open(args.text_file, "r", encoding="utf-8") as f:
        text = f.read().strip()

    print(f"{C.DIM}[run] text: {len(text)} chars, {len(text.split())} words | "
          f"quant={args.quant} | sequential | max_token={args.max_token}{C.RESET}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        print(f"{C.DIM}[run] GPU: {torch.cuda.get_device_name(0)}{C.RESET}")

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

    chunks = chunk_by_tokens(bino.tokenizer, text, args.max_token)
    print(f"{C.DIM}[run] scoring {len(chunks)} chunk(s) of <= {args.max_token} tokens "
          f"(loading models, please wait)...{C.RESET}")

    t1 = time.time()
    scores = bino.compute_score(chunks)
    if not isinstance(scores, list):
        scores = [scores]
    elapsed = time.time() - t1

    weights = [len(bino.tokenizer(c, add_special_tokens=False)["input_ids"]) for c in chunks]
    agg = float(np.average(scores, weights=weights))
    peak = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else None

    print_results(scores, weights, agg, quant=args.quant, peak_gib=peak, elapsed=elapsed)


if __name__ == "__main__":
    sys.exit(main())
