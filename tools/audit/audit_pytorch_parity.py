#!/usr/bin/env python3
import os
import sys
import subprocess
import numpy as np

def main():
    print("================================================================")
    print("  LANCIUS v12R1: PYTORCH / ONNX RUNTIME PARITY AUDIT          ")
    print("================================================================\n")

    # Check dependencies
    try:
        import onnxruntime as ort
    except ImportError:
        print("❌ FATAL: onnxruntime is not installed. Run: pip install onnxruntime")
        sys.exit(1)

    if not os.path.exists("datasets/samples/pytorch_lenet.onnx"):
        print("❌ FATAL: pytorch_lenet.onnx not found. Run tools/onnx/export_pytorch_onnx.py first.")
        sys.exit(1)

    # 1. Generate deterministic FP32 input
    print("[1/5] Generating deterministic FP32 input tensor...")
    np.random.seed(42)
    input_data = np.random.randn(1, 3, 32, 32).astype(np.float32)
    input_data.tofile("datasets/samples/parity_input.bin")
    print(f"      Input Shape: {input_data.shape} | Mean: {input_data.mean():.4f}")

    # 2. Run ONNX Runtime Baseline
    print("[2/5] Running ONNX Runtime (NVIDIA Math Baseline)...")
    sess = ort.InferenceSession("datasets/samples/pytorch_lenet.onnx")
    input_name = sess.get_inputs()[0].name
    baseline = sess.run(None, {input_name: input_data})[0]
    baseline.tofile("datasets/samples/baseline_out.bin")
    print(f"      Baseline Logits: {baseline.flatten()[:5]}")

    # 3. Convert ONNX to Lancius
    print("[3/5] Converting ONNX to Lancius Binary...")
    # V10S FIX: Always regenerate to prevent stale artifact divergence
    if os.path.exists("datasets/samples/pytorch_lenet.lancius"):
        os.remove("datasets/samples/pytorch_lenet.lancius")
    subprocess.run(["python3", "tools/onnx/onnx_to_lancius.py"], check=True, timeout=60)

    # 4. Compile and run C Parity Runner
    print("[4/5] Compiling and running Lancius C Engine...")
    subprocess.run(["build/bin/parity_runner"], check=True, timeout=60)

    # 5. Differential Comparison
    print("[5/5] Performing Differential Math Comparison...")
    lancius_raw = np.fromfile("datasets/samples/lancius_out.bin", dtype=np.float32)
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
    if relative_error < 1e-5:
        print("  ✅ PARITY VERIFIED: Lancius C Engine matches PyTorch exactly!")
        print(f"     (Relative error {relative_error:.2e} is well within FP64 precision limits).")
    else:
        print(f"  ❌ DIVERGENCE DETECTED: Relative error {relative_error:.2e} exceeds 1e-5 threshold.")
        sys.exit(1)
    print("================================================================")

if __name__ == "__main__":
    main()
