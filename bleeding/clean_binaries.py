#!/usr/bin/env python3
import os

def clean_elf_binaries():
    target_dir = "v10A3"
    deleted_count = 0

    # ELF magic bytes (Linux executables and .o object files)
    ELF_MAGIC = b'\x7fELF'
    # AR archive magic (Linux .a static libraries)
    AR_MAGIC = b'!<arch>\n'

    print(f"🧹 Scanning '{target_dir}' for compiled binaries...")

    for root, dirs, files in os.walk(target_dir):
        for f in files:
            filepath = os.path.join(root, f)
            try:
                with open(filepath, 'rb') as file:
                    header = file.read(8)

                    # Check if it's an ELF binary or AR archive
                    if header.startswith(ELF_MAGIC) or header.startswith(AR_MAGIC):
                        os.remove(filepath)
                        print(f"  🗑️  Deleted: {filepath}")
                        deleted_count += 1
            except Exception:
                pass

    print(f"✅ Clean complete. Vaporized {deleted_count} compiled binaries/objects.")
    print("   (Source code, Python scripts, and data blobs remain perfectly intact).")

if __name__ == "__main__":
    clean_elf_binaries()
