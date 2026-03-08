#!/bin/bash
# scripts/00_init_machine.sh

set -e  # Exit on error

echo "=== TACC Machine Initialization ==="

# 1. Install uv if not present
if ! command -v uv &> /dev/null; then
    echo ">> Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Add to path for this session
    export PATH="$HOME/.local/bin:$PATH"
else
    echo ">> uv is already installed."
fi

# 2. Setup Cache Directories in Scratch
# TACC $HOME is small (10GB usually). $SCRATCH is huge.
if [ -z "$SCRATCH" ]; then
    echo "!! WARNING: \$SCRATCH environment variable not found."
    echo "!! Are you sure you are on a TACC node?"
else
    echo ">> Configuring caches to use $SCRATCH..."
    mkdir -p "$SCRATCH/.uv-cache"
    mkdir -p "$SCRATCH/.pip-cache"
    mkdir -p "$SCRATCH/tmp"
fi

echo "=== Initialization Complete ==="
echo "You can now run the installation scripts."