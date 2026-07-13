import os
import sys
import subprocess
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import onnxruntime as ort

def main():
    print("================================================================")
    print("  LANCIUS v10A2: TRAINED MODEL REALITY CHECK                   ")
    print("================================================================")

    if not os.path.exists("trained_lenet.onnx"):
        subprocess.run(["python3", "train_1_epoch.py"], check=True)

    print("\n[1/4] Preparing 100 CIFAR-10 Test Images...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)
    images, labels = next(iter(testloader))
    images.numpy().astype(np.float32).tofile("test_batch.bin")

    print("[2/4] Running ONNX Runtime Baseline...")
    sess = ort.InferenceSession("trained_lenet.onnx")
    onnx_out = sess.run(None, {'input': images.numpy().astype(np.float32)})[0]
    onnx_preds = np.argmax(onnx_out, axis=1).astype(np.int32)

    print("[3/4] Converting to Lancius Binary...")
    if os.path.exists("trained_lenet.lancius"): os.remove("trained_lenet.lancius")

    # Patch onnx_to_lancius.py to accept args dynamically
    with open("onnx_to_lancius.py", "r") as f: content = f.read()
    if "sys.argv" not in content:
        content = content.replace(
            'convert("pytorch_lenet.onnx", "pytorch_lenet.lancius")',
            'in_p = sys.argv[1] if len(sys.argv) > 1 else "pytorch_lenet.onnx"\n    out_p = sys.argv[2] if len(sys.argv) > 2 else "pytorch_lenet.lancius"\n    convert(in_p, out_p)'
        )
        with open("onnx_to_lancius.py", "w") as f: f.write(content)

    subprocess.run(["python3", "onnx_to_lancius.py", "trained_lenet.onnx", "trained_lenet.lancius"], check=True)

    print("[4/4] Compiling and running Lancius C Engine on 100 images...")
    compile_cmd = [
        "gcc", "-O3", "-march=native", "-fopenmp", "-std=c11",
        "-I./include", "-o", "run_trained_batch",
        "examples/run_trained_batch.c", "liblancius.a", "-lm", "-lpthread"
    ]
    subprocess.run(compile_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["./run_trained_batch"], check=True)

    lancius_preds = np.fromfile("lancius_preds.bin", dtype=np.int32)

    print("\n" + "="*60)
    matches = np.sum(onnx_preds == lancius_preds)
    print(f"  Argmax Class Match: {matches}/100 ({matches}%)")

    ground_truth = labels.numpy()
    onnx_acc = np.sum(onnx_preds == ground_truth)
    lancius_acc = np.sum(lancius_preds == ground_truth)
    print(f"  ONNX Accuracy vs Ground Truth: {onnx_acc}%")
    print(f"  Lancius Accuracy vs Ground Truth: {lancius_acc}%")

    if matches >= 95:
        print("\n  ✅ REALITY CHECK PASSED: Trained weights survive the pipeline perfectly!")
        print("  🏆 The deployment stack handles real-world distributions.")
    else:
        print(f"\n  ❌ DIVERGENCE DETECTED: Only {matches}% Argmax match.")
    print("="*60)

if __name__ == "__main__":
    main()
