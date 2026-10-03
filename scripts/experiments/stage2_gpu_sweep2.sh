#!/usr/bin/env sh
# Stage 2 sweep 2: mosaic kept, stronger augmentation, 320 vs 416 px, plus yolo11s.
#   nohup sh scripts/experiments/stage2_gpu_sweep2.sh > logs/stage2_gpu_sweep2.log 2>&1 &
set -u
UV="${UV:-$HOME/.local/bin/uv}"
DEV="${DEV:-0}"
run() {
    echo "=== $(date) start $*"
    "$UV" run --no-sync python -m varroa_vision.train --device "$DEV" --workers 8 "$@"
    echo "=== $(date) done"
}
run --config configs/mite_yolo11n_strongaug.yaml
run --config configs/mite_yolo11n_416_strongaug.yaml
run --config configs/mite_yolo11n_strongaug.yaml --name yolo11s_320_official_strongaug --set model=yolo11s.pt
echo "SWEEP_DONE"
