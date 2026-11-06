#!/bin/bash
# resume_i6_training_low.sh — resume I6 training with CPU throttling so the
# desktop stays responsive (Firefox usable).
#
# Limits OpenMP threads to N cores (default 4) and runs at low nice priority.
#
# Usage: ./resume_i6_training_low.sh [--cores N] [--epochs M] [--lr L]

set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO"

CORES=4
EPOCHS=20
LR=0.0008

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cores) CORES="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --lr)     LR="$2";     shift 2 ;;
        *) echo "unknown arg: $1"; exit 2 ;;
    esac
done

if [[ ! -x build/bin/train_i6 ]]; then
    echo "build/bin/train_i6 not found — run: make build/bin/train_i6"
    exit 1
fi

CKPT="generation_1/I6/models/last.bin"
CORPUS="generation_1/I6/corpus"
MUSD="generation_1/I6/gen1_mu_sd.txt"
INIT="generation_1/I3/alu_pool/alu_01.lancius"

for f in "$CKPT" "$CORPUS/corpus.hdr" "$MUSD" "$INIT"; do
    if [[ ! -e "$f" ]]; then
        echo "MISSING: $f — cannot resume"
        exit 1
    fi
done

echo "============================================================"
echo "  I6 TRAINING RESUME (LOW CPU IMPACT)"
echo "  cores      : $CORES (OpenMP threads)"
echo "  checkpoint : $CKPT"
echo "  corpus     : $CORPUS"
echo "  mu/sd      : $MUSD"
echo "  epochs     : $EPOCHS  lr : $LR"
echo "  log        : generation_1/I6/models/train_i6.log"
echo "  nice       : 19 (lowest priority)"
echo "============================================================"

# OMP_NUM_THREADS limits OpenMP parallelism; nice lowers CPU scheduling priority.
# train_i6 auto-resumes from save_dir/last.bin when --resume is passed.
export OMP_NUM_THREADS=$CORES

nohup nice -n 19 build/bin/train_i6 "$CORPUS" 4096 2560 "$EPOCHS" 256 \
    --seed 20261026 --lr "$LR" \
    --init-from "$INIT" \
    --mu-sd "$MUSD" \
    --export-lancius generation_1/I6/models/i6_alu_00.lancius \
    --verify-lancius --save-dir generation_1/I6/models \
    --resume "$CKPT" \
    > generation_1/I6/models/train_i6.log 2>&1 &

echo "Started PID: $! (using $CORES cores, nice 19)"
echo "To watch:    tail -f generation_1/I6/models/train_i6.log"
echo "To stop:     kill <PID>"
echo "============================================================"
