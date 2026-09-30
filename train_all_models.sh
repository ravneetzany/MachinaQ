#!/usr/bin/env bash
# Train pointnet, through-hole, unified, and operation-classifier in sequence.
# Safe to re-run after an interruption: models whose final checkpoint already
# exists are skipped, and in-progress runs resume from their per-epoch
# checkpoint under outputs/checkpoints/ instead of restarting from scratch.
set -uo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate

run_model() {
    local name="$1"; shift
    local final_path="outputs/machinaq_${name//-/_}.pth"
    if [[ -f "$final_path" ]]; then
        echo "[train_all] $name already trained -> $final_path, skipping"
        return 0
    fi
    echo "[train_all] training $name ..."
    python run_train.py --model "$name" "$@"
    local status=$?
    if [[ $status -ne 0 ]]; then
        echo "[train_all] $name FAILED (exit $status)"
        return $status
    fi
    echo "[train_all] $name complete -> $final_path"
}

run_model pointnet || exit 1
run_model through-hole || exit 1
run_model unified --merge-weights || exit 1
run_model operation-classifier || exit 1

echo "[train_all] ALL TRAINING COMPLETE"
