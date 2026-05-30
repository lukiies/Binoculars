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


def _read_file_shared(path: str) -> bytes:
    """Read a file even when another process (Word) holds it open.

    Word opens documents with FILE_SHARE_DELETE in its share mode, so a reader
    must also request read+write+delete sharing. Python's open() does not, which
    is why it fails with PermissionError while Windows CopyFile succeeds. We open
    via CreateFileW with all three share flags. Non-Windows falls back to open().
    """
    if os.name != "nt":
        with open(path, "rb") as f:
            return f.read()

    import ctypes
    from ctypes import wintypes

    GENERIC_READ = 0x80000000
    FILE_SHARE_RWD = 0x1 | 0x2 | 0x4  # READ | WRITE | DELETE
    OPEN_EXISTING = 3
    FILE_ATTRIBUTE_NORMAL = 0x80
    INVALID_HANDLE = ctypes.c_void_p(-1).value

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateFileW.restype = wintypes.HANDLE
    k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD,
                                wintypes.HANDLE]
    handle = k32.CreateFileW(str(path), GENERIC_READ, FILE_SHARE_RWD, None,
                             OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, None)
    if not handle or handle == INVALID_HANDLE:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        chunks = []
        buf = ctypes.create_string_buffer(1 << 20)
        nread = wintypes.DWORD(0)
        while True:
            if not k32.ReadFile(handle, buf, len(buf), ctypes.byref(nread), None):
                raise ctypes.WinError(ctypes.get_last_error())
            if nread.value == 0:
                break
            chunks.append(buf.raw[:nread.value])
        return b"".join(chunks)
    finally:
        k32.CloseHandle(handle)


def _load_document(path: str):
    """Open a .docx, reading via shared access when the original is locked
    (open in Word) or held by OneDrive."""
    try:
        return Document(path)
    except (PackageNotFoundError, PermissionError, OSError):
        try:
            data = _read_file_shared(path)
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
