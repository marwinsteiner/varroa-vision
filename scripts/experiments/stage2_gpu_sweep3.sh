#!/usr/bin/env sh
# Stage 2 sweep 3: the mosaic recipe on (a) VarroaDataset + EV2 pseudo-labels and
# (b) the date-grouped split. Run after sweep 2:
#   nohup scripts/queue_after.sh <sweep2 pid> sh scripts/experiments/stage2_gpu_sweep3.sh > logs/stage2_gpu_sweep3.log 2>&1 &
set -u
UV="${UV:-$HOME/.local/bin/uv}"
DEV="${DEV:-0}"
run() {
    echo "=== $(date) start $*"
    "$UV" run --no-sync python -m varroa_vision.train --device "$DEV" --workers 8 "$@"
    echo "=== $(date) done"
}
run --config configs/mite_yolo11n_combined.yaml
run --config configs/mite_yolo11n_date.yaml --set patience=25
echo "SWEEP_DONE"
