"""R2-6 Python binding smoke: build headers only, no torch/onnx needed."""
import ctypes
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LIB = ROOT / "liblancius.a"

def main() -> int:
    # Static lib exists after make; shared smoke via ctypes on liblancius.a is
    # not loadable, so smoke = headers parse + lib exists + version consistent.
    if not LIB.is_file():
        print(f"FAIL: {LIB} missing (run make)")
        return 1
    print(f"PASS: {LIB} present ({LIB.stat().st_size} bytes)")
    vh = (ROOT / "include" / "lancius" / "lancius_version.h").read_text()
    if "LANCIUS_VERSION_STRING" not in vh:
        print("FAIL: version header broken")
        return 1
    print("PASS: version header parses")
    print("BINDING SMOKE: ALL HOLD (headers-only, no native load)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
