#!/bin/bash
# Install the differentiable training/VRAM environment from a Lonestar6 login
# node. PufferLib is deliberately excluded: it is a PPO-only optional extra and
# its isolated CUDA extension build can select a PyTorch CUDA version that does
# not match TACC's loaded toolkit.

set -euo pipefail

echo "=== Starting Pip/Login Installation ==="

: "${SCRATCH:?SCRATCH must be set; run this script on TACC}"

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

echo ">> Loading TACC modules..."
module load gcc/13.2.0 cuda/12.8 ninja/1.13.1

export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export SSL_CERT_DIR=/etc/pki/tls/certs/
export PIP_CACHE_DIR="$SCRATCH/.pip-cache"
export TMPDIR="$SCRATCH/tmp"
export TORCH_CUDA_ARCH_LIST="8.0"

mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

if [[ ! -x .venv/bin/python ]]; then
    echo ">> Creating Python 3.10 virtual environment..."
    uv venv --python 3.10
else
    echo ">> Reusing existing virtual environment..."
fi
source .venv/bin/activate

if ! python -m pip --version >/dev/null 2>&1; then
    echo ">> Bootstrapping pip..."
    get_pip="$TMPDIR/diffcam-get-pip.py"
    curl -sS https://bootstrap.pypa.io/get-pip.py -o "$get_pip"
    python "$get_pip"
fi

python -m pip install --upgrade pip setuptools wheel

echo ">> Installing CUDA 12.8 PyTorch..."
python -m pip install "torch==2.10.0+cu128" \
    --index-url https://download.pytorch.org/whl/cu128

echo ">> Installing project without the optional PPO/PufferLib extra..."
python -m pip install -e .

echo ">> Verifying TACC-compatible imports..."
python -c 'import torch, taichi, tqdm; assert torch.version.cuda == "12.8", torch.version.cuda; print("torch:", torch.__version__); print("CUDA:", torch.version.cuda); print("taichi:", taichi.__version__); print("tqdm:", tqdm.__version__)'
python -m pip check

echo "=== Installation Complete ==="
echo "To activate: source .venv/bin/activate"
echo "PufferLib is omitted; install the optional PPO extra separately if needed."
