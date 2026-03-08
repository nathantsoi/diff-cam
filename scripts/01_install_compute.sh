#!/bin/bash
# scripts/01_install_compute.sh

set -e # Exit immediately if a command exits with a non-zero status

# Check if we are potentially on a login node (hostname usually contains 'login')
if [[ $(hostname) == *"login"* ]]; then
    echo "!! WARNING: You appear to be on a login node."
    echo "!! The UV/Sync method usually requires a compute node for heavy compilation."
    echo "!! Please run: idev -p gpu-a100-dev -N 1 -n 1 -t 01:00:00"
    echo "!! Then run this script again."
    read -p "Press enter to continue anyway, or Ctrl+C to cancel..."
fi

echo "=== Starting UV/Compute Installation ==="

# 1. Load Modules
echo ">> Loading TACC modules..."
module load gcc/13.2.0 cuda/12.8 ninja/1.13.1

# 2. Setup Environment Variables
export TORCH_CUDA_ARCH_LIST="8.0" # A100
export UV_CACHE_DIR=$SCRATCH/.uv-cache
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export SSL_CERT_DIR=/etc/pki/tls/certs/

# Ensure cache exists
mkdir -p $UV_CACHE_DIR

# 3. Create Virtual Env
echo ">> Creating Virtual Environment (Python 3.10)..."
uv venv --python 3.10

# 4. Sync
echo ">> Running uv sync..."
uv sync

echo "=== Installation Complete ==="
echo "To activate: source .venv/bin/activate"