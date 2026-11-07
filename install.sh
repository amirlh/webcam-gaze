#!/bin/bash
# Installs the Python environment in ~/webcam_gaze_venv (or $GAZE_VENV): Linux x86_64, an NVIDIA GPU if any.
#   bash install.sh
# Undo: rm -rf ~/webcam_gaze_venv ~/.cache/huggingface/hub/models--UniGaze--UniGaze-models (the network's weights, about 2.5 GB).
set -euo pipefail
cd "$(dirname "$0")"

# uv (Python package manager), if absent
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
UV="$(command -v uv || echo "$HOME/.local/bin/uv")"

ENV_DIR="${GAZE_VENV:-$HOME/webcam_gaze_venv}"
"$UV" venv "$ENV_DIR" --python 3.11
PY="$ENV_DIR/bin/python"

# PyTorch: the default Linux wheels carry CUDA (NVIDIA). Without an NVIDIA GPU they run on the processor (slow).
"$UV" pip install --python "$PY" torch torchvision
# MediaPipe 0.10.21. Its OpenCV is swapped for the "headless" one, whose missing windows avoid a clash between OpenCV's Qt and PyQt5.
"$UV" pip install --python "$PY" "mediapipe==0.10.21" numpy scipy PyQt5 safetensors huggingface_hub "timm==1.0.30"
OCV="$("$PY" -c "import importlib.metadata as m; print(m.version('opencv-contrib-python'))")"
"$UV" pip uninstall --python "$PY" opencv-contrib-python
"$UV" pip install --python "$PY" "opencv-contrib-python-headless==$OCV"
# UniGaze: its package pins timm 0.3.2, too old for current PyTorch; timm 1.0.30 runs it.
"$UV" pip install --python "$PY" --no-deps "unigaze==0.1.3"

"$PY" - <<'EOF'
import torch, cv2, mediapipe, unigaze, PyQt5
print("torch", torch.__version__, "| CUDA", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no NVIDIA GPU seen")
print("opencv", cv2.__version__, "| mediapipe", mediapipe.__version__)
unigaze.load("unigaze_h14_joint", device="cpu")   # downloads ViT-H's weights once (about 2.5 GB)
print("weights unigaze_h14_joint ok")
EOF
echo "Installed: $ENV_DIR. Run the program from this folder with: $PY -m gaze"
