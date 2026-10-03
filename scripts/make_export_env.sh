#!/usr/bin/env sh
# Build a separate CPU-only environment for TFLite (LiteRT) export.
# ultralytics 8.4 exports LiteRT through litert-torch, which pins torch < 2.14, while
# the training environment uses the current torch. Run from the repo root:
#   sh scripts/make_export_env.sh
# then export with:
#   .venv-export/bin/python -m varroa_vision.export --weights <best.pt> --formats tflite
set -eu
UV="${UV:-$HOME/.local/bin/uv}"
"$UV" venv .venv-export --python 3.12 --seed
"$UV" pip install --python .venv-export/bin/python \
    --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple \
    "torch==2.13.*" "torchvision"
"$UV" pip install --python .venv-export/bin/python \
    -e . "litert-torch>=0.9.0" "ai-edge-litert>=2.1.4" "ai-edge-quantizer>=0.6.0" "onnx>=1.16" "onnxslim" "onnxruntime"
.venv-export/bin/python -c "import torch, litert_torch, ultralytics; print(torch.__version__, litert_torch.__version__, ultralytics.__version__)"
