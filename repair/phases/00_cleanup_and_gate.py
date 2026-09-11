#!/usr/bin/env python3
"""Phase 0.0: Cleanup diagnostic scripts, fix the verify_build gate, and fix README drift."""
import os, sys, shutil
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "00_harness"))
import safe_edit as se

def main():
    print("[00] cleanup diagnostics, fix gate, fix docs")

    # 1. Move diagnostic scripts out of the phases/ directory so orchestrator ignores them
    phases_dir = se.p("repair/phases")
    diag_dir = se.p("repair/diagnostics")
    os.makedirs(diag_dir, exist_ok=True)

    for script in ["203_debug.py", "203_gdb.py"]:
        src = os.path.join(phases_dir, script)
        dst = os.path.join(diag_dir, script)
        if os.path.exists(src):
            shutil.move(src, dst)
            print(f"  [move] {script} -> diagnostics/")

    # 2. Fix verify_build.sh to actually run the fault injection audit
    gate_file = "repair/00_harness/verify_build.sh"
    gate_content = se.read(se.p(gate_file))
    if "audit_fault_injection" not in gate_content:
        gate_content = gate_content.replace(
            'make liblancius.a 2>&1',
            'make liblancius.a audit_fault_injection 2>&1\nrc=$?\nif [ $rc -ne 0 ]; then exit $rc; fi\necho "[verify_build] ./audit_fault_injection"\n./audit_fault_injection'
        )
        gate_content = gate_content.replace(
            'rc=$?\nif [ $rc -ne 0 ]; then\necho "[verify_build] BUILD FAILED (exit $rc)"\nelse\necho "[verify_build] BUILD OK"\nfi\nexit $rc',
            'rc=$?\nif [ $rc -ne 0 ]; then\necho "[verify_build] GATE FAILED (exit $rc)"\nelse\necho "[verify_build] GATE OK"\nfi\nexit $rc'
        )
        se.backup(se.p(gate_file))
        se.write(se.p(gate_file), gate_content)
        print(f"  [edit] {gate_file}")

    # 3. Fix README doc drift
    readme_file = "README.md"
    readme_content = se.read(se.p(readme_file))
    if "docs/v11S_SCOPE.md" in readme_content:
        readme_content = readme_content.replace("docs/v11S_SCOPE.md", "docs/v11A3_SCOPE.md")
        readme_content = readme_content.replace("docs/releases/v11S/RELEASE_NOTES_v11S.md", "docs/releases/v11S/GITHUB_RELEASE_v11S.md")
        se.backup(se.p(readme_file))
        se.write(se.p(readme_file), readme_content)
        print(f"  [edit] {readme_file}")

    print("[00] done")

if __name__ == "__main__":
    main()
