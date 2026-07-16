#!/usr/bin/env python3
import os
import sys
import shutil
import urllib.request
import tarfile
import gzip

# Files and directories that are safe to nuke
CLEANUP_TARGETS = [
    "cifar-10-batches-bin",
    "cifar-10-batches-py",
    "data",  # Torchvision cache
    "cifar.tar.gz",
    "cifar-10-binary.tar.gz",
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte",
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
    "test_batch.bin",
    "parity_input.bin",
    "baseline_out.bin",
    "lancius_out.bin",
    "lancius_preds.bin",
    "fuzz.lancius",
    "garbage.lancius",
    "trunc.lancius",
    "malicious.lancius"
]

def clean_datasets():
    print("🧹 Cleaning up raw datasets and intermediate binaries...")
    freed = 0
    for target in CLEANUP_TARGETS:
        if os.path.exists(target):
            if os.path.isdir(target):
                size = sum(os.path.getsize(os.path.join(dp, f)) for dp, dn, fn in os.walk(target) for f in fn)
                shutil.rmtree(target)
                freed += size
                print(f"  🗑️  Deleted directory: {target}")
            else:
                size = os.path.getsize(target)
                os.remove(target)
                freed += size
                print(f"  🗑️  Deleted file: {target}")

    print(f"\n✅ Cleanup complete. Freed ~{freed / (1024*1024):.2f} MB.")
    print("💡 Note: Your trained .lancius, .onnx, and .gguf models are SAFE and untouched.")

def download_mnist():
    print("📥 Downloading MNIST...")
    base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
    ]
    for f in files:
        out_name = f.replace(".gz", "")
        if not os.path.exists(out_name):
            print(f"  Fetching {f}...")
            urllib.request.urlretrieve(base_url + f, f)
            with gzip.open(f, 'rb') as f_in:
                with open(out_name, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(f)
    print("✅ MNIST ready.")

def download_cifar10():
    print("📥 Downloading CIFAR-10 (Binary version for C)...")
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
    tar_name = "cifar-10-binary.tar.gz"
    if not os.path.exists("cifar-10-batches-bin/data_batch_1.bin"):
        print(f"  Fetching {tar_name}...")
        urllib.request.urlretrieve(url, tar_name)
        with tarfile.open(tar_name, "r:gz") as tar:
            tar.extractall()
        os.remove(tar_name)
    print("✅ CIFAR-10 ready.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Lancius Dataset Manager")
        print("-----------------------")
        print("Usage:")
        print("  python3 manage_datasets.py clean         # Deletes raw datasets to free space")
        print("  python3 manage_datasets.py download all  # Downloads MNIST and CIFAR-10")
        print("  python3 manage_datasets.py download mnist")
        print("  python3 manage_datasets.py download cifar10")
        sys.exit(1)

    action = sys.argv[1]
    if action == "clean":
        clean_datasets()
    elif action == "download":
        target = sys.argv[2] if len(sys.argv) > 2 else "all"
        if target in ["all", "mnist"]:
            download_mnist()
        if target in ["all", "cifar10"]:
            download_cifar10()
    else:
        print("Unknown action.")
