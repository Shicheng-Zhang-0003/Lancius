#!/usr/bin/env python3
"""
203_debug.py — Home-made GDB for the MATMUL_BATCHED backward fix.
Prints every value, every equation, every byte that determines why
lancius_ir_autodiff still returns non-NULL for batched matmul.
"""
import os, sys, re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILE = os.path.join(REPO, "../src/math/lancius_autodiff.c")

def hr(title=""):
    print(f"\n{'='*72}")
    if title:
        print(f"  {title}")
        print(f"{'='*72}")

def main():
    hr("PHASE 203 DEBUG: MATMUL_BATCHED BACKWARD DIAGNOSIS")

    content = open(FILE, "r").read()
    lines = content.split("\n")

    # ─── 1. Is the marker present? ───
    hr("[1/8] MARKER CHECK")
    marker = "v12R1-203"
    if marker in content:
        print(f"  ✅ Marker '{marker}' IS present in file.")
    else:
        print(f"  ❌ Marker '{marker}' NOT present in file.")
        print(f"     → The 203 replacement was NEVER applied.")

    # ─── 2. Find all MATMUL_BATCHED references ───
    hr("[2/8] ALL LANCIUS_OP_MATMUL_BATCHED OCCURRENCES")
    for i, line in enumerate(lines, 1):
        if "MATMUL_BATCHED" in line:
            print(f"  Line {i:4d}: {line.rstrip()}")

    # ─── 3. Extract the backward-pass MATMUL_BATCHED block ───
    hr("[3/8] BACKWARD-PASS MATMUL_BATCHED BLOCK (raw bytes)")
    # Find the backward section (second occurrence of MATMUL_BATCHED in an else-if)
    backward_pattern = re.compile(
        r'else\s+if\s*\(\s*fwd_n->op\s*==\s*LANCIUS_OP_MATMUL_BATCHED\s*\)\s*\{',
        re.DOTALL
    )
    matches = list(backward_pattern.finditer(content))
    print(f"  Found {len(matches)} else-if MATMUL_BATCHED block(s).")
    for idx, m in enumerate(matches):
        start = m.start()
        # Find the matching closing brace
        brace_count = 0
        block_start = content.index('{', start)
        pos = block_start
        while pos < len(content):
            if content[pos] == '{':
                brace_count += 1
            elif content[pos] == '}':
                brace_count -= 1
                if brace_count == 0:
                    break
            pos += 1
        block = content[start:pos+1]
        print(f"\n  --- Block {idx+1} (char offset {start}..{pos}) ---")
        for j, bline in enumerate(block.split("\n")):
            print(f"    {bline}")
        print(f"\n  Block length: {len(block)} chars, {block.count(chr(10))} lines")

    # ─── 4. What the 203 script's anchor was looking for ───
    hr("[4/8] ORIGINAL 203 ANCHOR (what the script searched for)")
    old_anchor = (
        "} else if (fwd_n->op == LANCIUS_OP_MATMUL_BATCHED) {\n"
        "            accum_grad(tg->graph, grad_map, fwd_n->inputs[0]->id, grad_out, fwd_to_full);\n"
        "            accum_grad(tg->graph, grad_map, fwd_n->inputs[1]->id, grad_out, fwd_to_full);\n"
        "        } else if (fwd_n->op == LANCIUS_OP_PERMUTE) {"
    )
    print("  Anchor string (repr):")
    for i, aline in enumerate(old_anchor.split("\n")):
        print(f"    [{i}] {repr(aline)}")

    count = content.count(old_anchor)
    print(f"\n  Anchor match count: {count}")
    if count == 0:
        print("  ❌ ANCHOR DID NOT MATCH. Diagnosing why...")
    elif count == 1:
        print("  ✅ Anchor matches exactly once. Replacement should have worked.")
        print("     → Check if the script was actually run AFTER the file was modified.")
    else:
        print(f"  ⚠️  Anchor matches {count} times (expected 1).")

    # ─── 5. Character-level diff: anchor vs actual ───
    hr("[5/8] CHARACTER-LEVEL DIFF (anchor vs actual file)")
    if count == 0:
        # Find the closest match using fuzzy search
        # Search for the key line
        key_line = "fwd_n->op == LANCIUS_OP_MATMUL_BATCHED"
        positions = [m.start() for m in re.finditer(re.escape(key_line), content)]
        print(f"  '{key_line}' found at {len(positions)} position(s).")

        for pi, pos in enumerate(positions):
            # Extract surrounding context (200 chars before, 400 after)
            ctx_start = max(0, pos - 100)
            ctx_end = min(len(content), pos + 400)
            ctx = content[ctx_start:ctx_end]

            print(f"\n  --- Context around occurrence {pi+1} (offset {pos}) ---")
            ctx_lines = ctx.split("\n")
            for ci, cl in enumerate(ctx_lines):
                # Show repr to expose tabs/spaces
                print(f"    {repr(cl)}")

            # Now compare with anchor line by line
            print(f"\n  --- Line-by-line comparison with anchor ---")
            anchor_lines = old_anchor.split("\n")
            actual_block_start = content.rfind("}", 0, pos)
            actual_block = content[actual_block_start:actual_block_start + len(old_anchor) + 200]
            actual_lines = actual_block.split("\n")

            max_lines = max(len(anchor_lines), len(actual_lines))
            for li in range(min(max_lines, 10)):
                a_line = anchor_lines[li] if li < len(anchor_lines) else "<missing>"
                f_line = actual_lines[li] if li < len(actual_lines) else "<missing>"
                match = "✅" if a_line == f_line else "❌"
                print(f"    {match} anchor[{li}]: {repr(a_line)}")
                if a_line != f_line:
                    print(f"       file[{li}]:   {repr(f_line)}")
                    # Show exact char diff
                    for ci2 in range(min(len(a_line), len(f_line))):
                        if a_line[ci2] != f_line[ci2]:
                            print(f"       First diff at char {ci2}: anchor={repr(a_line[ci2])} file={repr(f_line[ci2])}")
                            break
                    if len(a_line) != len(f_line):
                        print(f"       Length diff: anchor={len(a_line)} file={len(f_line)}")

    # ─── 6. Check for tabs vs spaces ───
    hr("[6/8] WHITESPACE ANALYSIS")
    tab_count = content.count("\t")
    space_indent_lines = sum(1 for l in lines if l.startswith("    ") and not l.startswith("\t"))
    tab_indent_lines = sum(1 for l in lines if l.startswith("\t"))
    print(f"  Total tab characters in file: {tab_count}")
    print(f"  Lines starting with 4+ spaces: {space_indent_lines}")
    print(f"  Lines starting with tab: {tab_indent_lines}")

    # Check the specific MATMUL_BATCHED area
    for m in backward_pattern.finditer(content):
        region = content[m.start():m.start()+500]
        region_tabs = region.count("\t")
        region_spaces = len(re.findall(r'^ +', region, re.MULTILINE))
        print(f"\n  In MATMUL_BATCHED backward region:")
        print(f"    Tabs: {region_tabs}")
        print(f"    Space-indented lines: {region_spaces}")
        if region_tabs > 0:
            print(f"    ⚠️  FILE USES TABS. The 203 anchor used SPACES. This is the mismatch.")

    # ─── 7. Check if 203 was run but on wrong content ───
    hr("[7/8] SCRIPT EXECUTION HISTORY")
    log_path = os.path.join(REPO, "repair/repair.log")
    if os.path.exists(log_path):
        log = open(log_path).read()
        if "203" in log:
            # Find 203-related lines
            for ll in log.split("\n"):
                if "203" in ll:
                    print(f"  {ll}")
        else:
            print("  No 203 entries in repair.log")
    else:
        print("  repair.log not found")

    # Check for backup files
    bak_files = [f for f in os.listdir(os.path.dirname(FILE)) if "autodiff" in f and ".bak" in f]
    if bak_files:
        print(f"\n  Backup files found: {bak_files}")
    else:
        print(f"\n  No backup files for autodiff found.")

    # ─── 8. The actual current backward code for MATMUL_BATCHED ───
    hr("[8/8] CURRENT BACKWARD CODE (what actually runs)")
    # Find the exact block between MATMUL_BATCHED and the next else-if
    pattern = re.compile(
        r'(else\s+if\s*\(\s*fwd_n->op\s*==\s*LANCIUS_OP_MATMUL_BATCHED\s*\)\s*\{)'
        r'(.*?)'
        r'(\}\s*else\s+if)',
        re.DOTALL
    )
    m = pattern.search(content)
    if m:
        print(f"  Opener:  {repr(m.group(1).strip())}")
        print(f"  Body:")
        for bl in m.group(2).strip().split("\n"):
            print(f"    {repr(bl)}")
        print(f"  Closer:  {repr(m.group(3).strip())}")

        # Check if it still has the old accum_grad calls
        body = m.group(2)
        if "accum_grad" in body:
            print(f"\n  ❌ DIAGNOSIS: The body still contains accum_grad calls.")
            print(f"     The 203 replacement was NOT applied.")
            print(f"     The backward pass still silently produces wrong gradients.")
        elif "return NULL" in body:
            print(f"\n  ✅ The body contains 'return NULL'. Fix appears applied.")
            print(f"     But the test still fails — check if the .o file was rebuilt.")
        else:
            print(f"\n  ⚠️  Unexpected body content.")
    else:
        print("  Could not find MATMUL_BATCHED backward block with regex.")

    # ─── 9. Check if the .o file is stale ───
    hr("[9/8] BUILD ARTIFACT CHECK")
    o_file = os.path.join(REPO, "src/math/lancius_autodiff.o")
    c_file = FILE
    if os.path.exists(o_file):
        o_mtime = os.path.getmtime(o_file)
        c_mtime = os.path.getmtime(c_file)
        print(f"  {c_file}")
        print(f"    mtime: {c_mtime}")
        print(f"  {o_file}")
        print(f"    mtime: {o_mtime}")
        if o_mtime < c_mtime:
            print(f"  ❌ .o file is OLDER than .c file. STALE BUILD.")
            print(f"     Run: make clean && make")
        else:
            print(f"  ✅ .o file is newer than .c file. Build is current.")
    else:
        print(f"  {o_file} does not exist. Run make first.")

    hr("DIAGNOSIS COMPLETE")

if __name__ == "__main__":
    main()
