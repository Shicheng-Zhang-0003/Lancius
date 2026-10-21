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
    print("  LANCIUS v12R1: TRAINED MODEL REALITY CHECK                  ")
    print("================================================================")

    if not os.path.exists("datasets/samples/trained_lenet.onnx"):
        subprocess.run(["python3", "tools/data/train_1_epoch.py"], check=True, timeout=600)

    print("\n[1/4] Preparing 100 CIFAR-10 Test Images...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = torchvision.datasets.CIFAR10(root='datasets/cifar10/torchvision', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)
    images, labels = next(iter(testloader))
    images_np = images.cpu().numpy().astype(np.float32)
    images_np.tofile("datasets/samples/test_batch.bin")

    print("[2/4] Running ONNX Runtime Baseline...")
    sess = ort.InferenceSession("datasets/samples/trained_lenet.onnx")
    onnx_out = sess.run(None, {'input': images_np})[0]
    onnx_preds = np.argmax(onnx_out, axis=1).astype(np.int32)

    print("[3/4] Converting to Lancius Binary...")
    if os.path.exists("datasets/samples/trained_lenet.lancius"): os.remove("datasets/samples/trained_lenet.lancius")

    subprocess.run(["python3", "tools/onnx/onnx_to_lancius.py", "datasets/samples/trained_lenet.onnx", "datasets/samples/trained_lenet.lancius"], check=True, timeout=60)

    print("[4/4] Compiling and running Lancius C Engine on 100 images...")
    subprocess.run(["build/bin/run_trained_batch"], check=True, timeout=60)

    lancius_preds = np.fromfile("datasets/samples/lancius_preds.bin", dtype=np.int32)
    if lancius_preds.size != onnx_preds.size:
        print(f"FATAL: Prediction count mismatch: Lancius {lancius_preds.size} vs ONNX {onnx_preds.size}")
        sys.exit(1)

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
    if matches < 95:
        sys.exit(1)

if __name__ == "__main__":
    main()
