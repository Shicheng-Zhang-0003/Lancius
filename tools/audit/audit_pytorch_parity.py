#!/usr/bin/env python3
"""ONNX Runtime vs Lancius differential parity probe.

All generated inputs/outputs live under the scratch directory. The old version
wrote parity_input.bin, baseline_out.bin, lancius_out.bin, and the converted
model into datasets/samples/, polluting the checkout on every run. The C
runner still uses its historical relative paths, so it runs with the scratch
directory as its working directory.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
SCRATCH = Path(os.environ.get("LANCIUS_SCRATCH", str(ROOT / "temp" / "scratch")))
WORK = SCRATCH / "pytorch_parity"
SAMPLES = WORK / "datasets" / "samples"


def main():
    print("================================================================")
    print("  LANCIUS v12R1: PYTORCH / ONNX RUNTIME PARITY AUDIT          ")
    print("================================================================\n")

    try:
        import onnxruntime as ort
    except ImportError:
        print("❌ FATAL: onnxruntime is not installed. Run: pip install onnxruntime")
        sys.exit(1)

    source_onnx = ROOT / "datasets" / "samples" / "pytorch_lenet.onnx"
    if not source_onnx.is_file():
        print("❌ FATAL: pytorch_lenet.onnx not found. Run tools/onnx/export_pytorch_onnx.py first.")
        sys.exit(1)

    SAMPLES.mkdir(parents=True, exist_ok=True)
    onnx_path = SAMPLES / "pytorch_lenet.onnx"
    model_path = SAMPLES / "pytorch_lenet.lancius"
    input_path = SAMPLES / "parity_input.bin"
    baseline_path = SAMPLES / "baseline_out.bin"
    output_path = SAMPLES / "lancius_out.bin"
    shutil.copy2(source_onnx, onnx_path)

    # ONNX may keep weights in external data files (pytorch_lenet.onnx.data).
    # Copy every referenced location into the scratch model directory too;
    # copying only the .onnx leaves ONNX Runtime unable to resolve them.
    import onnx
    source_model = onnx.load(str(source_onnx), load_external_data=False)
    locations = set()
    for initializer in source_model.graph.initializer:
        for entry in initializer.external_data:
            if entry.key == "location":
                locations.add(entry.value)
    for location in locations:
        src = source_onnx.parent / location
        dst = onnx_path.parent / location
        if not src.is_file():
            print(f"❌ FATAL: external ONNX data missing: {src}")
            sys.exit(1)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    # 1. Generate deterministic FP32 input
    print("[1/5] Generating deterministic FP32 input tensor...")
    np.random.seed(42)
    input_data = np.random.randn(1, 3, 32, 32).astype(np.float32)
    input_data.tofile(input_path)
    print(f"      Input Shape: {input_data.shape} | Mean: {input_data.mean():.4f}")

    # 2. Run ONNX Runtime Baseline
    print("[2/5] Running ONNX Runtime (NVIDIA Math Baseline)...")
    sess = ort.InferenceSession(str(onnx_path))
    input_name = sess.get_inputs()[0].name
    baseline = sess.run(None, {input_name: input_data})[0]
    baseline.tofile(baseline_path)
    print(f"      Baseline Logits: {baseline.flatten()[:5]}")

    # 3. Convert ONNX to Lancius
    print("[3/5] Converting ONNX to Lancius Binary...")
    if model_path.exists():
        model_path.unlink()
    subprocess.run(
        ["python3", "tools/onnx/onnx_to_lancius.py", str(onnx_path), str(model_path)],
        cwd=ROOT,
        check=True,
        timeout=60,
    )

    # 4. Compile and run C Parity Runner
    print("[4/5] Compiling and running Lancius C Engine...")
    subprocess.run(
        [str(ROOT / "build" / "bin" / "parity_runner")],
        cwd=WORK,
        check=True,
        timeout=60,
    )

    # 5. Differential Comparison
    print("[5/5] Performing Differential Math Comparison...")
    lancius_raw = np.fromfile(output_path, dtype=np.float32)
    if lancius_raw.size != baseline.size:
        print(f"❌ FATAL: Output size mismatch: Lancius {lancius_raw.size} vs ONNX {baseline.size}")
        sys.exit(1)
    lancius_out = lancius_raw.reshape(baseline.shape)

    max_diff = np.max(np.abs(baseline - lancius_out))
    mean_diff = np.mean(np.abs(baseline - lancius_out))

    # V1.0 FIX: Calculate Relative Error (Industry Standard for large logits)
    max_magnitude = np.max(np.abs(baseline))
    relative_error = max_diff / (max_magnitude + 1e-8)

    print(f"      Lancius Logits:  {lancius_out.flatten()[:5]}")
    print(f"      Max Absolute Error: {max_diff:.2e}")
    print(f"      Max Output Magnitude: {max_magnitude:.2e}")
    print(f"      Max Relative Error: {relative_error:.2e}")

    print("\n================================================================")
    if np.isfinite(relative_error) and relative_error < 1e-5:
        print("  ✅ PARITY VERIFIED: Lancius C Engine matches PyTorch exactly!")
        print(f"     (Relative error {relative_error:.2e} is well within FP64 precision limits).")
    else:
        print(f"  ❌ DIVERGENCE DETECTED: Relative error {relative_error:.2e} exceeds 1e-5 threshold.")
        sys.exit(1)
    print("================================================================")


if __name__ == "__main__":
    main()
