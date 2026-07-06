#!/usr/bin/env bash
# Create the `diffcam-occwl` conda environment for STEP -> SDF conversion
# (utils/step_to_sdf.py). Scripted version of the documented setup:
#
#   1. conda create -n diffcam-occwl python=3.10
#   2. conda install -c lambouj -c conda-forge occwl   (CAD engine)
#   3. pip install numpy trimesh scipy                 (mesh & math)
#   4. pip install open3d                              (fast SDF raycaster --
#      without it the pure-Python fallback takes over an hour at high res)
#
# Idempotent: safe to re-run; existing steps are skipped/no-ops.
#
# No conda? The converter also runs on the pip-installable OpenCascade wheel,
# straight into the project venv -- no separate environment needed:
#   uv pip install cadquery-ocp open3d
#
# Verify afterwards (should print "Sampling SDF using Open3D raycaster..."):
#   conda activate diffcam-occwl
#   python -m utils.step_to_sdf utils/STEPs/Extrusion.STEP -o utils/extrusion.npz
set -euo pipefail

ENV_NAME=diffcam-occwl
PY_VERSION=3.10

if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found. Install Miniconda/Miniforge, or skip conda entirely with:" >&2
    echo "  uv pip install cadquery-ocp open3d" >&2
    exit 1
fi

# Step 1: create the environment (python 3.10 has the best CAD compatibility)
if conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
    echo "Environment '$ENV_NAME' already exists; skipping creation."
else
    conda create -n "$ENV_NAME" python="$PY_VERSION" -y
fi

# Step 2: the CAD engine (occwl + OpenCASCADE); the lambouj channel is required
conda install -n "$ENV_NAME" -c lambouj -c conda-forge occwl -y

# Steps 3+4: mesh/math libraries and the Open3D raycaster accelerator,
# installed with uv into the env's interpreter.
ENV_PYTHON="$(conda run -n "$ENV_NAME" python -c 'import sys; print(sys.executable)')"
if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$ENV_PYTHON" numpy trimesh scipy open3d
else
    "$ENV_PYTHON" -m pip install numpy trimesh scipy open3d
fi

# Verify
conda run -n "$ENV_NAME" python - <<'EOF'
from OCC.Core.STEPControl import STEPControl_Reader  # the import the converter needs
import occwl, trimesh, open3d, numpy
print("pythonocc OK; occwl OK; trimesh", trimesh.__version__,
      "; open3d", open3d.__version__, "; numpy", numpy.__version__)
EOF

echo
echo "Environment '$ENV_NAME' is ready. Convert a STEP file with:"
echo "  conda activate $ENV_NAME"
echo "  python -m utils.step_to_sdf utils/STEPs/Extrusion.STEP -o utils/extrusion.npz"
