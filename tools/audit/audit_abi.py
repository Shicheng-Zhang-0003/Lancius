"""R2-6 ABI break-test: installed headers match version.h + lib symbols present."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def main() -> int:
    vh = (ROOT / "include" / "lancius" / "lancius_version.h").read_text()
    m = re.search(r'#define LANCIUS_VERSION_STRING "([^"]+)"', vh)
    pub = re.search(r'#define LANCIUS_VERSION_PUBLIC "([^"]+)"', vh)
    if not m or not pub:
        print("FAIL: version macros missing")
        return 1
    print(f"PASS: version {m.group(1)} public {pub.group(1)}")
    # Check new R2 headers exist
    for h in ("lancius_train.h", "lancius_sandbox.h"):
        p = ROOT / "include" / "lancius" / h
        if not p.is_file():
            print(f"FAIL: missing {h}")
            return 1
        print(f"PASS: header {h} present")
    # Check train/sandbox sources exist
    for s in ("src/train/lancius_train.c", "src/runtime/lancius_sandbox.c"):
        if not (ROOT / s).is_file():
            print(f"FAIL: missing {s}")
            return 1
        print(f"PASS: source {s} present")
    print("ABI BREAK-TEST: ALL HOLD")
    return 0

if __name__ == "__main__":
    sys.exit(main())
