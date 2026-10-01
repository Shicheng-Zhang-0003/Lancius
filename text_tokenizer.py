"""Char byte-level problem encoder + manifest helpers (stdlib only).

R2-2 decision recorded (mute mathematician, v12R2):
  BPE REJECTED for v12R2 — speaking needs BPE, scoring does not.
  This module implements the deterministic problem encoder: UTF-8 bytes
  0..255 plus 4 special tokens. It always roundtrips ASCII + arbitrary
  UTF-8 (including astral plane, e.g. U+1F600). Reads problems, never speaks.

Vocab:
  0..255   : raw UTF-8 byte value
  256      : <pad>
  257      : <unk>
  258      : <bos>
  259      : <eos>
  VOCAB_SIZE = 260
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

VERSION = "char-v1"

SPECIAL_TOKENS: dict[str, int] = {
    "<pad>": 256,
    "<unk>": 257,
    "<bos>": 258,
    "<eos>": 259,
}
PAD_ID = 256
UNK_ID = 257
BOS_ID = 258
EOS_ID = 259
VOCAB_SIZE = 260
_SKIP_ON_DECODE = frozenset({PAD_ID, BOS_ID, EOS_ID})

__all__ = [
    "VERSION",
    "SPECIAL_TOKENS",
    "PAD_ID",
    "UNK_ID",
    "BOS_ID",
    "EOS_ID",
    "VOCAB_SIZE",
    "CharTokenizer",
    "sha256_file",
    "write_manifest",
    "verify_manifest",
]


class CharTokenizer:
    """Byte-level tokenizer: str <-> list[int] via UTF-8 bytes."""

    version: str = VERSION

    def __init__(self, special_tokens: dict[str, int] | None = None) -> None:
        specials = dict(SPECIAL_TOKENS) if special_tokens is None else dict(special_tokens)
        if specials != SPECIAL_TOKENS:
            raise ValueError(f"special_tokens mismatch: {specials!r} != {SPECIAL_TOKENS!r}")
        self.special_tokens: dict[str, int] = specials

    def encode(self, s: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        if not isinstance(s, str):
            raise TypeError(f"encode() expects str, got {type(s).__name__}")
        ids = list(s.encode("utf-8"))
        if add_bos:
            ids.insert(0, BOS_ID)
        if add_eos:
            ids.append(EOS_ID)
        return ids

    def decode(self, ids) -> str:
        buf = bytearray()
        for i in ids:
            if not isinstance(i, int):
                raise TypeError(f"decode() ids must be int, got {type(i).__name__}")
            if 0 <= i <= 255:
                buf.append(i)
            elif i in _SKIP_ON_DECODE:
                continue
            elif i == UNK_ID:
                raise ValueError("cannot decode <unk> (257) to bytes")
            else:
                raise ValueError(f"token id out of range 0..259: {i!r}")
        return bytes(buf).decode("utf-8", errors="strict")

    def save_sidecar(self, path: str | os.PathLike) -> Path:
        p = Path(path)
        payload = {
            "version": self.version,
            "kind": "char-byte-tokenizer",
            "encoding": "utf-8-bytes",
            "special_tokens": dict(self.special_tokens),
            "vocab_size": VOCAB_SIZE,
        }
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return p

    @classmethod
    def load_sidecar(cls, path: str | os.PathLike) -> "CharTokenizer":
        p = Path(path)
        payload = json.loads(p.read_text(encoding="utf-8"))
        ver = payload.get("version")
        if ver != VERSION:
            raise ValueError(f"sidecar version mismatch: got {ver!r}, expected {VERSION!r}")
        specials = payload.get("special_tokens")
        if specials != SPECIAL_TOKENS:
            raise ValueError(f"sidecar special_tokens mismatch: {specials!r}")
        if payload.get("vocab_size") != VOCAB_SIZE:
            raise ValueError(f"sidecar vocab_size mismatch: {payload.get('vocab_size')!r}")
        obj = cls(special_tokens=specials)
        obj.version = ver
        return obj

    def __len__(self) -> int:
        return VOCAB_SIZE


def sha256_file(path: str | os.PathLike, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _manifest_targets(dir_path: str | os.PathLike) -> list[Path]:
    d = Path(dir_path)
    targets: list[Path] = []
    for pattern in ("*.jsonl", "*.bin"):
        targets.extend(p for p in d.glob(pattern) if p.is_file())
    targets = [p for p in targets if p.name != "manifest.json"]
    targets.sort(key=lambda p: p.name)
    return targets


def write_manifest(dir_path: str | os.PathLike) -> Path:
    d = Path(dir_path)
    if not d.is_dir():
        raise NotADirectoryError(f"not a directory: {d}")
    files: dict[str, dict[str, int | str]] = {}
    for p in _manifest_targets(d):
        files[p.name] = {"sha256": sha256_file(p), "size": p.stat().st_size}
    payload = {
        "version": 1,
        "tokenizer_version": VERSION,
        "files": files,
    }
    out = d / "manifest.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def verify_manifest(dir_path: str | os.PathLike) -> bool:
    import sys

    d = Path(dir_path)
    manifest = d / "manifest.json"
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        entries = payload["files"]
        if not isinstance(entries, dict):
            raise ValueError("manifest 'files' must be an object")
    except Exception as e:
        print(f"verify_manifest: unreadable manifest {manifest}: {e}", file=sys.stderr)
        return False
    ok = True
    for name in sorted(entries):
        try:
            meta = entries[name]
            want_hash = meta["sha256"]
            want_size = meta["size"]
        except (TypeError, KeyError) as e:
            print(f"verify_manifest: bad entry {name!r}: {e}", file=sys.stderr)
            ok = False
            continue
        p = d / name
        if not p.is_file():
            print(f"verify_manifest: missing {name}", file=sys.stderr)
            ok = False
            continue
        got_size = p.stat().st_size
        if got_size != want_size:
            print(f"verify_manifest: size mismatch {name}", file=sys.stderr)
            ok = False
            continue
        got_hash = sha256_file(p)
        if got_hash != want_hash:
            print(f"verify_manifest: sha256 mismatch {name}", file=sys.stderr)
            ok = False
    return ok
