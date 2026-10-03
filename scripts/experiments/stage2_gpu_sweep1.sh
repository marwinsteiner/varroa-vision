#!/usr/bin/env sh
# Stage 2 recipe sweep on the GPU box, run from the repo root:
#   nohup sh scripts/experiments/stage2_gpu_sweep1.sh > logs/stage2_gpu_sweep1.log 2>&1 &
# Each run is a few minutes on an RTX 4060. Results land under runs/mite/<name>.
set -u
UV="${UV:-$HOME/.local/bin/uv}"
DEV="${DEV:-0}"
run() {
    echo "=== $(date) start $*"
    "$UV" run --no-sync python -m varroa_vision.train --device "$DEV" --workers 8 "$@"
    echo "=== $(date) done"
}
run --config configs/mite_yolo11n_nomosaic.yaml --name yolo11n_320_official_nomosaic_gpu
run --config configs/mite_yolo11n.yaml --name yolo11n_320_official_mosaic_gpu --set patience=25
run --config configs/mite_yolo11n_warmfix.yaml
run --config configs/mite_yolo11n_warmfix_freeze.yaml
echo "SWEEP_DONE"
