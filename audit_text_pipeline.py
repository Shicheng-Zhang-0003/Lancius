"""Audit for text_tokenizer.py (stdlib only). Exits 1 on failure."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from text_tokenizer import CharTokenizer, verify_manifest, write_manifest

ASCII_STR = "Solve 2+2=4 (easy)?"
ASTRAL_STR = "hello \U0001D11E\U0001F30D\U0001F600\U00010448"
OPENCOD_TMP = Path("/tmp/opencode")

failures: list[str] = []


def check(name: str, fn) -> None:
    try:
        fn()
    except Exception as e:
        print(f"FAIL {name}: {e!r}")
        failures.append(name)
    else:
        print(f"PASS {name}")


def test_roundtrip_ascii() -> None:
    tok = CharTokenizer()
    ids = tok.encode(ASCII_STR)
    assert all(0 <= i <= 255 for i in ids)
    assert tok.decode(ids) == ASCII_STR
    assert tok.decode(tok.encode("")) == ""


def test_roundtrip_astral() -> None:
    tok = CharTokenizer()
    assert any(ord(c) > 0xFFFF for c in ASTRAL_STR)
    ids = tok.encode(ASTRAL_STR)
    assert len(ids) > len(ASTRAL_STR)
    assert tok.decode(ids) == ASTRAL_STR
    combo = ASCII_STR + " " + ASTRAL_STR
    assert tok.decode(tok.encode(combo)) == combo


def test_sidecar() -> None:
    OPENCOD_TMP.mkdir(parents=True, exist_ok=True)
    tok = CharTokenizer()
    with tempfile.TemporaryDirectory(prefix="tok-sidecar-", dir=str(OPENCOD_TMP)) as td:
        sidecar = Path(td) / "tokenizer.sidecar.json"
        tok.save_sidecar(sidecar)
        assert sidecar.is_file()
        tok2 = CharTokenizer.load_sidecar(sidecar)
        assert tok2.version == "char-v1"
        for s in (ASCII_STR, ASTRAL_STR):
            assert tok2.decode(tok2.encode(s)) == s
            assert tok.encode(s) == tok2.encode(s)


def test_manifest() -> None:
    OPENCOD_TMP.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tok-manifest-", dir=str(OPENCOD_TMP)) as td:
        d = Path(td)
        (d / "train.jsonl").write_text('{"text": "Solve 2+2=4 (easy)?"}\n', encoding="utf-8")
        (d / "data.bin").write_bytes(bytes(range(256)))
        manifest = write_manifest(d)
        assert manifest.is_file()
        assert verify_manifest(d) is True
        with open(d / "data.bin", "ab") as f:
            f.write(b"\x00")
        assert verify_manifest(d) is False


def main() -> int:
    check("roundtrip_ascii", test_roundtrip_ascii)
    check("roundtrip_astral", test_roundtrip_astral)
    check("sidecar", test_sidecar)
    check("manifest", test_manifest)
    if failures:
        print(f"{len(failures)} FAILURES: {failures}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
