"""LDFD ctypes bridge: drive the Lancius Live Data Feeding Framework.

LDFD is a C library, `manage_datasets.py` is Python, and this module is the
seam between them. It is deliberately *optional*: if libsnapshot.so or its
libcurl dependency is missing, `available()` returns False and the caller
falls back to the existing urllib path. The goal is to remove manual
downloading when the native library is present, never to make the dataset
pipeline depend on it.

    import ldfd
    if ldfd.available():
        with ldfd.session() as s:
            r = s.fetch_to_file(url, dest, decompress="gzip")
    else:
        ...urllib fallback...

The bridge deliberately exposes only `snap_fetch_to_buffer` plus the
gzip/tar helpers. Those take plain buffers, so nothing is ever called back
into Python from C: no ctypes trampolines on the fetch path, no risk of a
garbage-collected callback firing mid-transfer.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

__all__ = [
    "LDFDError",
    "SourceSpec",
    "Result",
    "available",
    "unavailable_reason",
    "lib_path",
    "session",
    "Session",
    "MAX_DOWNLOAD_BYTES",
]

# Reuse the dataset manager's existing cap so both paths agree on the limit.
MAX_DOWNLOAD_BYTES = 2 * 1024 * 1024 * 1024   # 2 GiB

# snap_error_t
SNAP_OK = 0
SNAP_ERR_NOMEM = 1
SNAP_ERR_CURL = 2
SNAP_ERR_PARSE = 3
SNAP_ERR_TRANSFORM = 4
SNAP_ERR_OUTPUT = 5
SNAP_ERR_CONFIG = 6

_ERROR_NAMES = {
    SNAP_ERR_NOMEM: "out of memory",
    SNAP_ERR_CURL: "network error (or HTTP status >= 400)",
    SNAP_ERR_PARSE: "corrupt or truncated payload",
    SNAP_ERR_TRANSFORM: "transform error",
    SNAP_ERR_OUTPUT: "output error (size cap exceeded or write failed)",
    SNAP_ERR_CONFIG: "configuration error",
}

_HERE = Path(__file__).resolve().parent
_CANDIDATE_LIBS = (
    _HERE / "3463-LDFD" / "libsnapshot.so",
    _HERE / "libsnapshot.so",
    _HERE.parent / "3463-LDFD" / "libsnapshot.so",
)

_lib: Optional[ctypes.CDLL] = None
_load_error: Optional[str] = None
_load_attempted = False


class LDFDError(RuntimeError):
    """A libsnapshot call failed, or the payload was rejected."""


def lib_path() -> Optional[Path]:
    for cand in _CANDIDATE_LIBS:
        if cand.is_file():
            return cand
    found = ctypes.util.find_library("snapshot")
    if found and os.path.isabs(found) and os.path.isfile(found):
        return Path(found)
    return None


def _load() -> Optional[ctypes.CDLL]:
    global _lib, _load_error, _load_attempted
    if _load_attempted:
        return _lib
    _load_attempted = True

    path = lib_path()
    if path is None:
        _load_error = ("libsnapshot.so not found "
                       "(build it with: make -C 3463-LDFD)")
        return None
    try:
        lib = ctypes.CDLL(str(path))
    except OSError as exc:
        _load_error = f"cannot load {path}: {exc}"
        return None

    # Explicit signatures everywhere: ctypes defaults every argument to
    # int, which silently truncates 64-bit pointers.
    sigs = {
        "snap_fetch_to_buffer": ([ctypes.c_char_p, ctypes.c_char_p,
                                  ctypes.c_void_p, ctypes.c_size_t], ctypes.c_int),
        "snap_gzip_is_gzip": ([ctypes.c_void_p, ctypes.c_size_t], ctypes.c_int),
        "snap_gzip_new": ([ctypes.c_void_p], ctypes.c_void_p),
        "snap_gzip_feed": ([ctypes.c_void_p, ctypes.c_void_p,
                            ctypes.c_size_t], ctypes.c_int),
        "snap_gzip_finish": ([ctypes.c_void_p], ctypes.c_int),
        "snap_gzip_free": ([ctypes.c_void_p], None),
        "snap_tar_extract": ([ctypes.c_void_p, ctypes.c_char_p,
                              ctypes.c_void_p], ctypes.c_int),
        "snap_buffer_init": ([ctypes.c_void_p, ctypes.c_size_t], None),
        "snap_buffer_free": ([ctypes.c_void_p], None),
        "snap_buffer_append": ([ctypes.c_void_p, ctypes.c_void_p,
                                ctypes.c_size_t], None),
    }
    for name, (argtypes, restype) in sigs.items():
        try:
            fn = getattr(lib, name)
        except AttributeError:
            _load_error = (f"{path} predates this bridge: missing {name}")
            return None
        fn.argtypes = argtypes
        fn.restype = restype

    _lib = lib
    return lib


def available() -> bool:
    """True when the native library is present and usable."""
    return _load() is not None


def unavailable_reason() -> Optional[str]:
    _load()
    return _load_error


class SnapBuffer(ctypes.Structure):
    """Mirrors snap_buffer_t in include/snapshot.h."""
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("len", ctypes.c_size_t),
        ("cap", ctypes.c_size_t),
    ]


class TarStats(ctypes.Structure):
    """Mirrors snap_tar_stats_t in include/decompress.h."""
    _fields_ = [
        ("files", ctypes.c_longlong),
        ("bytes", ctypes.c_longlong),
        ("skipped", ctypes.c_longlong),
        ("rejected", ctypes.c_longlong),
    ]


@dataclass
class SourceSpec:
    """One fetch target.

    decompress: None | "gzip" | "auto". "auto" inflates only when the
        payload actually carries the gzip magic, so plain .jsonl and .csv
        pass through untouched.
    extract_tar: unpack a tar payload into the destination's parent
        directory after fetching (CIFAR-10 ships .tar.gz).
    """
    url: str
    path: str
    name: Optional[str] = None
    interval_sec: int = 0
    decompress: Optional[str] = "auto"
    extract_tar: bool = False
    auth_header: Optional[str] = None


@dataclass
class Result:
    ok: bool
    path: Optional[str] = None
    bytes_written: int = 0
    inflated: bool = False
    tar_files: int = 0
    tar_skipped: int = 0
    tar_rejected: int = 0
    error: Optional[str] = None


def _describe(rc: int) -> str:
    return _ERROR_NAMES.get(rc, f"unknown error {rc}")


class Session:
    """Owns a libsnapshot handle for the duration of a `with` block."""

    def __init__(self, max_bytes: int = MAX_DOWNLOAD_BYTES):
        self._lib = _load()
        if self._lib is None:
            raise LDFDError(_load_error or "libsnapshot unavailable")
        self.max_bytes = max_bytes

    # -- buffer helpers ----------------------------------------------------
    def _buf(self, cap: int = 65536) -> SnapBuffer:
        buf = SnapBuffer()
        self._lib.snap_buffer_init(ctypes.byref(buf), max(cap, 1))
        return buf

    def _bytes(self, buf: SnapBuffer) -> bytes:
        if not buf.data or buf.len == 0:
            return b""
        return ctypes.string_at(buf.data, buf.len)

    def _free(self, buf: SnapBuffer) -> None:
        self._lib.snap_buffer_free(ctypes.byref(buf))

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        # Nothing persistent is held: each fetch owns its own task.
        return None

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- public API --------------------------------------------------------
    def fetch_bytes(self, url: str, auth_header: Optional[str] = None,
                    max_bytes: Optional[int] = None) -> bytes:
        cap = self.max_bytes if max_bytes is None else max_bytes
        buf = self._buf()
        try:
            rc = self._lib.snap_fetch_to_buffer(
                url.encode(),
                auth_header.encode() if auth_header else None,
                ctypes.byref(buf), cap)
            if rc != SNAP_OK:
                raise LDFDError(f"fetch {url}: {_describe(rc)}")
            return self._bytes(buf)
        finally:
            self._free(buf)

    def gunzip(self, data: bytes) -> bytes:
        """Inflate a gzip payload. Raises on a truncated or corrupt stream."""
        if not self._lib.snap_gzip_is_gzip(data, len(data)):
            raise LDFDError("payload is not gzip")
        out = self._buf(max(65536, len(data) * 3))
        try:
            gz = self._lib.snap_gzip_new(ctypes.byref(out))
            if not gz:
                raise LDFDError("snap_gzip_new returned NULL")
            try:
                rc = self._lib.snap_gzip_feed(gz, data, len(data))
                if rc == SNAP_OK:
                    rc = self._lib.snap_gzip_finish(gz)
            finally:
                self._lib.snap_gzip_free(gz)
            if rc != SNAP_OK:
                raise LDFDError(f"gzip: {_describe(rc)}")
            return self._bytes(out)
        finally:
            self._free(out)

    def untar(self, data: bytes, dest_dir: str):
        buf = self._buf(max(65536, len(data)))
        try:
            self._lib.snap_buffer_append(ctypes.byref(buf), data, len(data))
            stats = TarStats()
            rc = self._lib.snap_tar_extract(ctypes.byref(buf),
                                            str(dest_dir).encode(),
                                            ctypes.byref(stats))
            if rc != SNAP_OK:
                raise LDFDError(f"tar: {_describe(rc)}")
            return stats
        finally:
            self._free(buf)

    def fetch_to_file(self, url: str, dest: str,
                      decompress: Optional[str] = "auto",
                      extract_tar: bool = False,
                      auth_header: Optional[str] = None,
                      max_bytes: Optional[int] = None) -> Result:
        """Fetch `url` once and land it at `dest`, atomically.

        The bytes are written to a temporary file in the destination
        directory and then renamed, so an interrupted or failed fetch never
        leaves a truncated file where a dataset is expected. Nothing is
        removed if the fetch fails.
        """
        raw = self.fetch_bytes(url, auth_header, max_bytes)

        inflated = False
        payload = raw
        if decompress in ("gzip", "auto") and \
                self._lib.snap_gzip_is_gzip(raw, len(raw)):
            payload = self.gunzip(raw)
            inflated = True

        dest_path = Path(dest)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        if extract_tar:
            stats = self.untar(payload, str(dest_path.parent))
            return Result(ok=True, path=str(dest_path.parent),
                          bytes_written=stats.bytes, inflated=inflated,
                          tar_files=stats.files, tar_skipped=stats.skipped,
                          tar_rejected=stats.rejected)

        tmp_fd, tmp_name = tempfile.mkstemp(
            dir=str(dest_path.parent), prefix=dest_path.name + ".",
            suffix=".part")
        try:
            with os.fdopen(tmp_fd, "wb") as fh:
                fh.write(payload)
            os.replace(tmp_name, dest)
        except BaseException:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
            raise
        return Result(ok=True, path=str(dest), bytes_written=len(payload),
                      inflated=inflated)

    def fetch_spec(self, spec: SourceSpec) -> Result:
        return self.fetch_to_file(
            spec.url, spec.path,
            decompress=spec.decompress,
            extract_tar=spec.extract_tar,
            auth_header=spec.auth_header,
        )


def session(**kwargs) -> Session:
    """`with ldfd.session() as s: ...`"""
    return Session(**kwargs)