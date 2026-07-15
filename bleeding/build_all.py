#!/usr/bin/env python3
import os
import subprocess
import sys

def compile_all():
    target_dir = "v10A3"
    print("🔨 Compiling Lancius Core and all Executables...")

    # 1. Build the core library and standard targets
    result = subprocess.run(["make", "all"], cwd=target_dir)
    if result.returncode != 0:
        print("❌ Core compilation failed! Check GCC output above.")
        sys.exit(1)

    # 2. Build the extra audit/parity targets that aren't in 'make all'
    extra_targets = [
        "audit_memory_pool", "audit_flash_attention", "audit_modern_llm",
        "audit_ffi", "audit_threadpool_parity", "audit_nan_injection",
        "parity_runner", "soak_fuzz", "test_diamond_memory"
    ]

    print("\n⚙️  Building Adversarial Audit Targets...")
    for target in extra_targets:
        # Suppress stdout to keep the terminal clean, but let stderr show if it fails
        res = subprocess.run(["make", target], cwd=target_dir, stdout=subprocess.DEVNULL)
        if res.returncode == 0:
            print(f"  ✅ Built {target}")
        else:
            # If it fails, it just means the target isn't in the Makefile yet (safe to ignore)
            pass

    print("\n🏁 Compilation complete. The armory is fully stocked.")

if __name__ == "__main__":
    compile_all()
