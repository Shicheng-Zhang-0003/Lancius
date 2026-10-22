"""R2-6 ABI break-test: installed headers match version.h + lib symbols present.

The original version only checked that ``build/lib/liblancius.a`` existed and
that the version header parsed, while printing "lib symbols present". That was
a false claim: a library with every public symbol removed still passed. This
version checks the archive with ``nm`` and requires every ``LANCIUS_EXPORT``
declaration in the stable-API header plus a curated core set to be defined.
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
LIB = ROOT / "build" / "lib" / "liblancius.a"

CORE_SYMBOLS = {
    "lancius_graph_create",
    "lancius_graph_destroy",
    "lancius_graph_save_v2",
    "lancius_graph_load_v2",
    "lancius_ir_schedule",
    "lancius_ir_autodiff",
    "lancius_arena_create",
    "lancius_arena_alloc",
    "lancius_arena_destroy",
    "lancius_compile_and_run",
    "lancius_sgd_step",
    "lancius_sgdm_step",
    "lancius_adamw_step",
    "lancius_clip_global_norm",
    "lancius_sandbox_check_graph",
    "lancius_sandbox_check_weights",
}


def exported_symbols(archive: Path):
    nm = shutil.which("nm")
    if not nm:
        return None, "nm not found"
    proc = subprocess.run(
        [nm, "-g", "--defined-only", str(archive)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None, proc.stderr.strip() or f"nm exited {proc.returncode}"
    symbols = set()
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[1] in {"T", "D", "B", "R", "W", "V"}:
            symbols.add(parts[2])
    return symbols, None


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

    if not LIB.is_file():
        print(f"FAIL: missing {LIB.relative_to(ROOT)}; build the library first")
        return 1
    symbols, err = exported_symbols(LIB)
    if symbols is None:
        print(f"FAIL: cannot inspect {LIB.name}: {err}")
        return 1

    stable_header = (ROOT / "include" / "lancius" / "lancius_stable_api.h").read_text()
    stable_exports = set(
        re.findall(
            r"LANCIUS_EXPORT\s+[^;]*?\b(lancius_[A-Za-z0-9_]+)\s*\(",
            stable_header,
            re.S,
        )
    )
    if not stable_exports:
        print("FAIL: stable-API header declares no LANCIUS_EXPORT functions")
        return 1
    required = stable_exports | CORE_SYMBOLS
    missing = sorted(required - symbols)
    if missing:
        print(f"FAIL: {len(missing)} required symbol(s) absent from {LIB.name}:")
        for name in missing:
            print(f"  missing: {name}")
        return 1

    print(f"PASS: {len(required)} required symbols present "
          f"({len(stable_exports)} stable API + {len(CORE_SYMBOLS)} core)")
    print("ABI BREAK-TEST: ALL HOLD")
    return 0


if __name__ == "__main__":
    sys.exit(main())
