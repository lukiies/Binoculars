"""Extract readable text from a .pdf for Binoculars scoring.

Uses pypdf (pure-Python). Handles files locked by another app / OneDrive via a
Win32 shared read. Note: this extracts text from *digital* PDFs — scanned/image
PDFs contain no text layer and would need OCR (out of scope here).

Usage:
    python extract_pdf.py input.pdf [output.txt]
"""
import io
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _winio import read_file_shared

from pypdf import PdfReader

# Make console output safe regardless of the active Windows code page.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _open_reader(path: str) -> PdfReader:
    """Open a PDF, falling back to a shared read if the file is locked."""
    try:
        return PdfReader(path)
    except (PermissionError, OSError):
        return PdfReader(io.BytesIO(read_file_shared(path)))


def extract(path: str) -> str:
    reader = _open_reader(path)

    # Some PDFs are encrypted with an empty user password; try to unlock.
    if getattr(reader, "is_encrypted", False):
        try:
            reader.decrypt("")
        except Exception:
            raise RuntimeError(
                f"'{path}' is password-protected. Remove the password and retry."
            )

    parts = []
    for page in reader.pages:
        try:
            t = (page.extract_text() or "").strip()
        except Exception:
            t = ""
        if t:
            parts.append(t)
    return "\n".join(parts)


if __name__ == "__main__":
    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "extracted_text.txt"
    text = extract(src)
    Path(out).write_text(text, encoding="utf-8")
    words = len(text.split())
    print(f"[extract] source: {src}")
    print(f"[extract] -> {out}")
    print(f"[extract] chars={len(text)} words={words} approx_tokens~={int(words*1.3)}")
    if words < 20:
        print("[extract] WARNING: almost no text found. This may be a SCANNED / "
              "image-only PDF (no text layer). Such files need OCR, which this "
              "tool does not do. Try a text-based PDF or paste the text into a .txt.")
    print("[extract] --- first 1200 chars ---")
    print(text[:1200])
