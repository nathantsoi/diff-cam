#!/bin/bash
# scripts/01_install_login.sh

set -e 

echo "=== Starting Pip/Login Installation ==="

# 1. Load Modules
echo ">> Loading TACC modules..."
module load gcc/13.2.0 cuda/12.8 ninja/1.13.1

# 2. Create Venv
echo ">> Creating Virtual Environment..."
uv venv --python 3.10

# 3. Bootstrap Pip
echo ">> Bootstrapping Pip..."
source .venv/bin/activate
curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
python /tmp/get-pip.py

# 4. Configure Environment
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export SSL_CERT_DIR=/etc/pki/tls/certs/
export PIP_CACHE_DIR=$SCRATCH/.pip-cache
export TMPDIR=$SCRATCH/tmp
export TORCH_CUDA_ARCH_LIST="8.0"

mkdir -p $PIP_CACHE_DIR $TMPDIR

# 5. Install
echo ">> Installing PyTorch..."
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128

echo ">> Installing Project (Editable mode)..."
python -m pip install -e .

echo "=== Installation Complete ==="