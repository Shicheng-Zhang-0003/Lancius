#!/usr/bin/env python3
"""
safe_edit.py - surgical, idempotent, anchor-verified editing library for the
Lancius v12R1 repair campaign.

Invariants enforced for every repair script:
  1. Anchors must match EXACTLY ONCE, or the edit aborts (no guessing).
  2. Idempotency markers prevent double-application.
  3. Every modified file is backed up before writing.
"""
import os, shutil, time

HERE        = os.path.dirname(os.path.abspath(__file__))
REPAIR_ROOT = os.path.dirname(HERE)
REPO_ROOT   = os.path.dirname(REPAIR_ROOT)

def p(rel):
    return os.path.join(REPO_ROOT, rel)

def read(path):
    with open(path, "r") as f:
        return f.read()

def write(path, content):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(content)
    os.replace(tmp, path)

def backup(path):
    if os.path.exists(path):
        shutil.copy2(path, path + ".bak." + time.strftime("%Y%m%d-%H%M%S"))

def ensure_unique(content, anchor, label):
    n = content.count(anchor)
    if n != 1:
        raise RuntimeError(f"[ABORT] anchor for {label} matched {n} times (need 1): {anchor[:60]!r}")

def ensure_absent(content, marker, label):
    if marker in content:
        raise RuntimeError(f"[ABORT] marker already present for {label}: {marker[:60]!r}")

def insert_before(content, anchor, text):
    ensure_unique(content, anchor, "insert_before")
    return content.replace(anchor, text + anchor, 1)

def insert_after(content, anchor, text):
    ensure_unique(content, anchor, "insert_after")
    return content.replace(anchor, anchor + text, 1)

def replace_once(content, anchor, text):
    ensure_unique(content, anchor, "replace_once")
    return content.replace(anchor, text, 1)

MAKEFILE_SRC_ANCHOR = "src/compiler/lancius_quantize.c"

def add_makefile_source(src_rel):
    """Idempotently insert a source file into the makefile SRCS list."""
    mk = p("makefile")
    content = read(mk)
    if src_rel in content:
        print(f"  [skip] makefile already lists {src_rel}")
        return False
    new = replace_once(content, MAKEFILE_SRC_ANCHOR,
                       src_rel + " \\\n" + MAKEFILE_SRC_ANCHOR)
    backup(mk)
    write(mk, new)
    print(f"  [makefile] added {src_rel}")
    return True

def create_file(rel, content, marker):
    """Create a new file idempotently."""
    path = p(rel)
    if os.path.exists(path):
        if marker in read(path):
            print(f"  [skip] {rel} already present")
            return False
        raise RuntimeError(f"[ABORT] {rel} exists but lacks marker; refusing to overwrite")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write(path, content)
    print(f"  [create] {rel}")
    return True

def edit_file(rel, transform):
    """Apply transform(content)->content to an existing file, with backup."""
    path = p(rel)
    content = read(path)
    new = transform(content)
    if new == content:
        print(f"  [skip] {rel} unchanged")
        return False
    backup(path)
    write(path, new)
    print(f"  [edit] {rel}")
    return True
