#!/usr/bin/env bash
# Build a disclosed, bounded CNC-Net modernized-stack smoke in a NEW venv + workdir.
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  printf 'Usage: %s NEW_VENV_DIR NEW_RUN_DIR [GPU_INDEX]\n' "$0" >&2
  exit 2
fi

VENV_DIR=$1
RUN_DIR=$2
GPU_INDEX=${3:-0}
RECIPE_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(CDPATH= cd -- "${RECIPE_DIR}/../.." && pwd)"
cd -- "${PROJECT_ROOT}"

CNC_REVISION=ba0652f9941686058c07c98617b9748e72f5f019
PYTORCH3D_REVISION=33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba
PYTORCH3D_URL=https://github.com/facebookresearch/pytorch3d.git
PYTORCH3D_DIR="${PYTORCH3D_DIR:-${PROJECT_ROOT}/external/pytorch3d-0.7.9}"
CPU_THREADS=${CPU_THREADS:-4}
export UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/diffcam-uv-diffcam-worker-3}

if [[ ! "${GPU_INDEX}" =~ ^[0-9]+$ ]]; then
  printf 'GPU_INDEX must be a nonnegative integer, got %q\n' "${GPU_INDEX}" >&2
  exit 2
fi
if [[ -e "${VENV_DIR}" || -e "${RUN_DIR}" || -e "${RUN_DIR}.stdout.log" || -e "${RUN_DIR}.freeze.txt" ]]; then
  printf 'Refusing to overwrite an existing environment/run path: %s / %s\n' "${VENV_DIR}" "${RUN_DIR}" >&2
  exit 2
fi
mkdir -p -- "$(dirname -- "${VENV_DIR}")" "$(dirname -- "${RUN_DIR}")"

# This is the ONLY external model/reference acquisition performed by the script,
# and it always uses the immutable revision from configs/references.json.
uv run --locked python -I "${PROJECT_ROOT}/scripts/pull_references.py" \
  --subset cnc-net --root "${PROJECT_ROOT}" --manifest "${PROJECT_ROOT}/configs/references.json"

# Fetch/build a second exact, explicitly declared pin; never a moving HEAD/tag.
if [[ ! -d "${PYTORCH3D_DIR}/.git" ]]; then
  if [[ -e "${PYTORCH3D_DIR}" ]]; then
    printf 'PyTorch3D source path exists but is not a git checkout: %s\n' "${PYTORCH3D_DIR}" >&2
    exit 2
  fi
  git init --quiet "${PYTORCH3D_DIR}"
  git -C "${PYTORCH3D_DIR}" remote add origin "${PYTORCH3D_URL}"
  GIT_TERMINAL_PROMPT=0 git -C "${PYTORCH3D_DIR}" fetch --quiet --depth 1 origin "${PYTORCH3D_REVISION}"
  git -C "${PYTORCH3D_DIR}" checkout --quiet --detach "${PYTORCH3D_REVISION}"
fi
if [[ $(git -C "${PYTORCH3D_DIR}" rev-parse HEAD) != "${PYTORCH3D_REVISION}" ]]; then
  printf 'Unexpected PyTorch3D revision in %s\n' "${PYTORCH3D_DIR}" >&2
  exit 2
fi
if [[ -n $(git -C "${PYTORCH3D_DIR}" status --porcelain) ]]; then
  printf 'Refusing a dirty PyTorch3D checkout: %s\n' "${PYTORCH3D_DIR}" >&2
  exit 2
fi

CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.9}
if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  printf 'BLOCKED: CUDA compiler not found at %s/bin/nvcc\n' "${CUDA_HOME}" >&2
  exit 1
fi
CUB_HOME=${CUB_HOME:-${CUDA_HOME}/include}

# Fresh interpreter never uses or updates the workspace lock. Direct URL/version
# pins (including Torch and every transitive dependency) are in requirements.txt.
env -u UV_ENV_FILE uv --no-config venv --python 3.11.13 "${VENV_DIR}"
env -u UV_ENV_FILE uv --no-config pip install --python "${VENV_DIR}/bin/python" \
  -r "${RECIPE_DIR}/requirements.txt"

export CUDA_HOME CUB_HOME FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=12.0
export MAX_JOBS=${CPU_THREADS} CMAKE_BUILD_PARALLEL_LEVEL=${CPU_THREADS}
export CC=/usr/bin/gcc CXX=/usr/bin/g++
export PATH="${CUDA_HOME}/bin:${PATH}"
env -u UV_ENV_FILE uv --no-config pip install --python "${VENV_DIR}/bin/python" \
  --no-build-isolation --no-deps "${PYTORCH3D_DIR}"
env -u UV_ENV_FILE uv --no-config pip freeze --python "${VENV_DIR}/bin/python" > "${RUN_DIR}.freeze.txt"

# Refuse to repoint this bounded case at a changed source revision or unknown input.
export PYTHONPATH="${RECIPE_DIR}/overrides"
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=${CPU_THREADS} MKL_NUM_THREADS=${CPU_THREADS} OPENBLAS_NUM_THREADS=1 CUDA_LAUNCH_BLOCKING=1

timeout -k 10s 480s \
  "${VENV_DIR}/bin/python" "${RECIPE_DIR}/run_modernized_smoke.py" \
  --source "${PROJECT_ROOT}/external/cnc-net" --source-revision "${CNC_REVISION}" \
  --work-dir "${RUN_DIR}" --gpu "${GPU_INDEX}" \
  > >(tee "${RUN_DIR}.stdout.log") 2>&1

"${VENV_DIR}/bin/python" -I "${RECIPE_DIR}/inspect_native_result.py" \
  --runtime "${RUN_DIR}" --experiment t50-tracked-smoke \
  --output "${RUN_DIR}/artifact-inventory.json"
