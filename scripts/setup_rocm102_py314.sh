#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_CACHE_DIR="$PWD/.cache/uv"
export YOLO_CONFIG_DIR="$PWD/.cache/ultralytics"
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME
PYTHON_FOR_ROCM10_PY314="${PYTHON_FOR_ROCM10_PY314:-$HOME/.local/share/uv/python/cpython-3.14.3-linux-x86_64-gnu/bin/python3.14}"
if [ ! -x .venv-rocm102-py314/bin/python ]; then
  uv venv --python "$PYTHON_FOR_ROCM10_PY314" .venv-rocm102-py314
fi
# Keep compiled compute/image packages wheel-only. The rocm metadata package
# is distributed as an sdist and needs a local wheel build, as in the 3.12 setup.
# Keep AMD as the first index; no cross-index unsafe-best-match.
uv pip install --only-binary torch,torchvision,amd-torch-device-gfx1151,amd-torch-device-gfx115x,numpy,scipy,opencv-python,matplotlib,contourpy,kiwisolver,pillow,polars-runtime-32 --python .venv-rocm102-py314/bin/python \
  --index https://stable.repo.amd.com/rocm/whl-next/ \
  'torch[device-gfx1151]==2.13.0+rocm10.2.0' \
  'torchvision[device-gfx1151]==0.28.0+rocm10.2.0'
uv pip install --only-binary torch,torchvision,amd-torch-device-gfx1151,amd-torch-device-gfx115x,numpy,scipy,opencv-python,matplotlib,contourpy,kiwisolver,pillow,polars-runtime-32 --python .venv-rocm102-py314/bin/python \
  --index https://stable.repo.amd.com/rocm/whl-next/ \
  --constraint configs/rocm102-constraints.txt --requirement configs/rocm10-py314-yolo.txt
uv pip check --python .venv-rocm102-py314/bin/python
.venv-rocm102-py314/bin/python -B - <<'PY'
import importlib.metadata as m
import json
from pathlib import Path
import sys
import torch, torchvision, ultralytics
assert sys.version_info[:2] == (3, 14), sys.version
assert torch.__version__ == '2.13.0+rocm10.2.0', torch.__version__
assert torchvision.__version__ == '0.28.0+rocm10.2.0', torchvision.__version__
assert torch.version.hip, 'Not a HIP build'
assert Path(torch.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), torch.__file__
Path('results').mkdir(exist_ok=True)
Path('results/packages-rocm102-py314.json').write_text(json.dumps(dict(
    python=sys.version, executable=sys.executable, torch_file=torch.__file__, hip=torch.version.hip,
    packages=sorted((d.metadata['Name'],d.version) for d in m.distributions())), indent=2))
print('ROCm 10.2 environment ready:', torch.__version__, torchvision.__version__, ultralytics.__version__)
PY
