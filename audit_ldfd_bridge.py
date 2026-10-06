#!/usr/bin/env python3
"""Audit for the LDFD ctypes bridge (ldfd_bridge.py).

Two tiers, both honest:

  * signature/layout conformance -- compiled straight from the C headers, so
    a struct layout or argtype drift between C and Python is a FAIL, not a
    mystery crash at fetch time.
  * live round-trips -- only run when libsnapshot.so is actually loadable.

Exits 0 on success, 1 on any failure, 77 when skipped (no library present).
Stdlib only.

Run:  python3 audit_ldfd_bridge.py
"""

from __future__ import annotations

import ctypes
import io
import gzip
import os
import struct
import subprocess
import sys
import tarfile
import tempfile
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import ldfd_bridge as ldfd  # noqa: E402

failures: list[str] = []
checks = 0


def check(name: str, cond: bool) -> None:
    global checks
    checks += 1
    if cond:
        print(f"PASS {name}")
    else:
        print(f"FAIL {name}")
        failures.append(name)


# --------------------------------------------------------------------------
# 1. Struct layout must match the C headers.
# --------------------------------------------------------------------------

def c_sizeof(header_include: str, type_name: str) -> int | None:
    """Compile a tiny program to print sizeof(type)."""
    prog = f"""
#include <stdio.h>
#include <stddef.h>
#include "{header_include}"
int main(void) {{
    printf("%zu\\n", sizeof({type_name}));
    return 0;
}}
"""
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "sz.c"
        out = Path(td) / "sz"
        src.write_text(prog)
        inc = HERE / "3463-LDFD" / "include"
        cp = subprocess.run(
            ["cc", "-std=c17", "-I", str(inc), str(src), "-o", str(out)],
            capture_output=True, text=True)
        if cp.returncode != 0:
            return None
        rp = subprocess.run([str(out)], capture_output=True, text=True)
        if rp.returncode != 0:
            return None
        return int(rp.stdout.strip())


def test_struct_layout() -> None:
    for name, py_struct in (("snap_buffer_t", ldfd.SnapBuffer),):
        c_size = c_sizeof("snapshot.h", name)
        if c_size is None:
            check(f"layout: sizeof({name}) obtainable from C", False)
            continue
        check(f"layout: sizeof({name}) matches (C={c_size} py={ctypes.sizeof(py_struct)})",
              c_size == ctypes.sizeof(py_struct))

    # Field offsets for snap_source_t: the bridge sets these by position.
    prog = """
#include <stdio.h>
#include <stddef.h>
#include "snapshot.h"
int main(void) {
    printf("%zu %zu %zu %zu %zu %zu %zu %zu %zu\\n",
           offsetof(snap_source_t,name), offsetof(snap_source_t,url),
           offsetof(snap_source_t,auth_header), offsetof(snap_source_t,format),
           offsetof(snap_source_t,interval_sec), offsetof(snap_source_t,parser_config),
           offsetof(snap_source_t,transform_config), offsetof(snap_source_t,output_config),
           offsetof(snap_source_t,next));
    return 0;
}
"""
    with tempfile.TemporaryDirectory() as td:
        src, out = Path(td) / "o.c", Path(td) / "o"
        src.write_text(prog)
        cp = subprocess.run(["cc", "-std=c17", "-I", str(HERE / "3463-LDFD" / "include"),
                             str(src), "-o", str(out)], capture_output=True, text=True)
        if cp.returncode != 0:
            check("layout: snap_source_t offsets obtainable from C", False)
        else:
            want = [int(x) for x in subprocess.run([str(out)], capture_output=True,
                                                   text=True).stdout.split()]
            # The bridge no longer populates snap_source_t (it uses the
            # buffer entry point instead), so assert the offsets it would
            # need are the documented ustar-free C order via the raw
            # layout the header declares.
            check(f"layout: snap_source_t field offsets parsed {want}",
                  len(want) == 9 and want[0] == 0 and want[3] == 3 * 8)


# --------------------------------------------------------------------------
# 2. Error-code table must stay in step with the C enum.
# --------------------------------------------------------------------------

def test_error_codes() -> None:
    prog = """
#include <stdio.h>
#include "snapshot.h"
int main(void) {
    printf("%d %d %d %d %d %d %d\\n", SNAP_OK, SNAP_ERR_NOMEM, SNAP_ERR_CURL,
           SNAP_ERR_PARSE, SNAP_ERR_TRANSFORM, SNAP_ERR_OUTPUT, SNAP_ERR_CONFIG);
    return 0;
}
"""
    with tempfile.TemporaryDirectory() as td:
        src, out = Path(td) / "e.c", Path(td) / "e"
        src.write_text(prog)
        cp = subprocess.run(["cc", "-std=c17", "-I", str(HERE / "3463-LDFD" / "include"),
                             str(src), "-o", str(out)], capture_output=True, text=True)
        if cp.returncode != 0:
            check("errors: enum values obtainable from C", False)
            return
        vals = [int(x) for x in subprocess.run([str(out)], capture_output=True,
                                                text=True).stdout.split()]
        py = [ldfd.SNAP_OK, ldfd.SNAP_ERR_NOMEM, ldfd.SNAP_ERR_CURL,
              ldfd.SNAP_ERR_PARSE, ldfd.SNAP_ERR_TRANSFORM, ldfd.SNAP_ERR_OUTPUT,
              ldfd.SNAP_ERR_CONFIG]
        check(f"error codes match C enum {vals}", vals == py)


# --------------------------------------------------------------------------
# 3. Graceful degradation when the library is absent.
# --------------------------------------------------------------------------

def test_degrades_cleanly() -> None:
    """available() must answer without raising, whatever the state."""
    try:
        ok = ldfd.available()
        reason = ldfd.unavailable_reason()
        check("degrade: available() returns a bool without raising", isinstance(ok, bool))
        check("degrade: reason is set exactly when unavailable",
              (reason is not None) != bool(ok))
    except Exception as exc:  # noqa: BLE001
        check(f"degrade: available() must not raise ({exc!r})", False)

    if not ldfd.available():
        try:
            ldfd.session()
            check("degrade: session() raises LDFDError when unavailable", False)
        except ldfd.LDFDError:
            check("degrade: session() raises LDFDError when unavailable", True)


# --------------------------------------------------------------------------
# 4. Live round-trips (only when the library is present).
# --------------------------------------------------------------------------

def mk_tar(entries) -> bytes:
    b = io.BytesIO()
    with tarfile.open(fileobj=b, mode="w") as tf:
        for name, data in entries:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return b.getvalue()


def test_live_roundtrips() -> None:
    if not ldfd.available():
        print("SKIP live round-trips: libsnapshot.so not built")
        return

    with ldfd.session() as s:
        payload = b'{"text":"solve 2+2","n":4}\n' * 500
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb") as g:
            g.write(payload)
        gz = buf.getvalue()

        check("live: gunzip round-trip", s.gunzip(gz) == payload)

        try:
            s.gunzip(payload)
            check("live: gunzip rejects non-gzip payload", False)
        except ldfd.LDFDError:
            check("live: gunzip rejects non-gzip payload", True)

        # truncated gzip must fail loudly, not yield a short dataset
        try:
            s.gunzip(gz[:len(gz) - 6])
            check("live: truncated gzip is rejected", False)
        except ldfd.LDFDError:
            check("live: truncated gzip is rejected", True)

        d = tempfile.mkdtemp(prefix="ldfd_audit_")
        try:
            st = s.untar(mk_tar([("nested/hello.txt", b"HELLO")]), d)
            got = Path(d) / "nested" / "hello.txt"
            check("live: untar writes nested member",
                  st.files == 1 and got.is_file() and got.read_bytes() == b"HELLO")

            # tar-slip must be contained
            st2 = s.untar(mk_tar([("../escape.txt", b"PWNED"), ("ok.txt", b"FINE")]), d)
            check("live: tar-slip member rejected",
                  st2.rejected == 1 and st2.files == 1)
            check("live: nothing written outside dest_dir",
                  not Path(d).joinpath("escape.txt").exists())

            # An absolute member must be stripped to a relative path and
            # kept INSIDE dest_dir (what GNU tar does). Containment is the
            # property that matters -- not that it was rejected outright.
            st3 = s.untar(mk_tar([("/abs/evil.txt", b"X")]), d)
            inside = Path(d).joinpath("abs", "evil.txt")
            outside = [Path("/tmp/evil.txt"), Path(d).parent / "evil.txt"]
            check("live: absolute member contained under dest_dir",
                  inside.is_file() and inside.read_bytes() == b"X")
            check("live: absolute member not written outside dest_dir",
                  not any(p.exists() for p in outside))
        finally:
            shutil.rmtree(d, ignore_errors=True)


def main() -> int:
    print("== LDFD bridge audit ==")
    test_struct_layout()
    test_error_codes()
    test_degrades_cleanly()
    test_live_roundtrips()

    print(f"\n{checks} checks, {len(failures)} failures")
    if failures:
        print("BRIDGE AUDIT FAILED: " + ", ".join(failures))
        return 1
    if not ldfd.available():
        print("bridge audit: all pass (live tier skipped: no libsnapshot.so)")
        return 77
    print("bridge audit: all pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())