#!/bin/bash
# resume_i6_training.sh — resume I6 training from the last checkpoint.
# Safe to run after a reboot; checkpoints and corpus are on disk.
#
# Usage: ./resume_i6_training.sh [--epochs N] [--lr L]
#
# Resumes from generation_1/I6/models/last.bin (auto-resume is built into
# train_i6: if --resume is omitted but save_dir/last.bin exists, it loads it).

set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO"

EPOCHS=20
LR=0.0008

while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs) EPOCHS="$2"; shift 2 ;;
        --lr)     LR="$2";     shift 2 ;;
        *) echo "unknown arg: $1"; exit 2 ;;
    esac
done

# Sanity: make sure the trainer binary exists
if [[ ! -x build/bin/train_i6 ]]; then
    echo "build/bin/train_i6 not found — run: make build/bin/train_i6"
    exit 1
fi

# Sanity: make sure checkpoint + corpus exist
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
echo "  I6 TRAINING RESUME"
echo "  checkpoint : $CKPT"
echo "  corpus     : $CORPUS"
echo "  mu/sd      : $MUSD"
echo "  init-from  : $INIT (only used if no checkpoint)"
echo "  epochs     : $EPOCHS  lr : $LR"
echo "  log        : generation_1/I6/models/train_i6.log"
echo "============================================================"

# train_i6 auto-resumes from save_dir/last.bin if --resume is omitted.
# We pass --resume explicitly to be safe.
nohup build/bin/train_i6 "$CORPUS" 4096 2560 "$EPOCHS" 256 \
    --seed 20261026 --lr "$LR" \
    --init-from "$INIT" \
    --mu-sd "$MUSD" \
    --export-lancius generation_1/I6/models/i6_alu_00.lancius \
    --verify-lancius --save-dir generation_1/I6/models \
    --resume "$CKPT" \
    > generation_1/I6/models/train_i6.log 2>&1 &

echo "Started PID: $!"
echo "To watch:    tail -f generation_1/I6/models/train_i6.log"
echo "To stop:     kill <PID>"
echo "============================================================"
