"""Shared helper: read a file even when another process holds it open.

Word / OneDrive open documents with FILE_SHARE_DELETE in their share mode, so a
reader must also request read+write+delete sharing. Python's open() does not,
which is why it fails with PermissionError while Windows CopyFile succeeds. We
open via CreateFileW with all three share flags. Non-Windows falls back to open().
"""
import os


def read_file_shared(path: str) -> bytes:
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
