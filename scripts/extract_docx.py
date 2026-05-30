"""Extract readable text from the target .docx (paragraphs + table cells).

Writes the extracted text next to the script and prints simple stats so we know
how much text Binoculars will see.

Robust to files that are currently OPEN in Word / held by OneDrive: such files
are locked for direct opening but still allow a shared-read copy, so we fall back
to extracting from a temporary copy.
"""
import io
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _winio import read_file_shared

from docx import Document
from docx.document import Document as _Doc
from docx.opc.exceptions import PackageNotFoundError
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph

# Make console output safe regardless of the active Windows code page
# (e.g. en-dashes / smart quotes in the preview must not crash extraction).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _load_document(path: str):
    """Open a .docx, reading via shared access when the original is locked
    (open in Word) or held by OneDrive."""
    try:
        return Document(path)
    except (PackageNotFoundError, PermissionError, OSError):
        try:
            data = read_file_shared(path)
            return Document(io.BytesIO(data))
        except Exception as e:
            raise RuntimeError(
                f"Could not read '{path}'. It may be open in Word or not fully "
                f"downloaded by OneDrive. Close it (or right-click > 'Always keep "
                f"on this device') and try again. Underlying error: {e}"
            )


def iter_block_items(parent):
    """Yield paragraphs and tables in document order."""
    if isinstance(parent, _Doc):
        parent_elm = parent.element.body
    else:
        parent_elm = parent._tc
    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)


def table_text(table):
    rows = []
    for row in table.rows:
        cells = [c.text.strip() for c in row.cells]
        # collapse duplicate merged cells
        dedup = []
        for c in cells:
            if not dedup or dedup[-1] != c:
                dedup.append(c)
        rows.append(" | ".join(dedup))
    return "\n".join(rows)


def extract(path: str) -> str:
    doc = _load_document(path)
    parts = []
    for block in iter_block_items(doc):
        if isinstance(block, Paragraph):
            t = block.text.strip()
            if t:
                parts.append(t)
        elif isinstance(block, Table):
            t = table_text(block).strip()
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
    print("[extract] --- first 1200 chars ---")
    print(text[:1200])
