#!/usr/bin/env bash
#
# One-click environment setup for MMSegmentation + EfficientViT on Linux.
#
# Creates an isolated conda environment (or a venv when conda is unavailable),
# installs PyTorch 2.1.0 / MMCV 2.1.0 / MMEngine 0.10.7, installs this
# repository in editable mode and verifies the EfficientViT model.
#
# Usage:
#   bash scripts/setup_linux.sh                 # CUDA 11.8
#   bash scripts/setup_linux.sh --cpu           # CPU only
#   bash scripts/setup_linux.sh --force         # recreate the environment
#   bash scripts/setup_linux.sh --with-tests    # also install test deps
#   bash scripts/setup_linux.sh --dry-run       # print commands only
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

DEVICE="cu118"
ENV_DIR=".venv-mmseg"
FORCE=0
WITH_TESTS=0
SKIP_VERIFY=0
DRY_RUN=0

usage() {
    sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu) DEVICE="cpu"; shift ;;
        --cuda) DEVICE="cu118"; shift ;;
        --env-dir) ENV_DIR="${2:?--env-dir needs a value}"; shift 2 ;;
        --force) FORCE=1; shift ;;
        --with-tests) WITH_TESTS=1; shift ;;
        --skip-verify) SKIP_VERIFY=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
    esac
done

export PYTHONNOUSERSITE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

ENV_ABS="$(realpath -m "$ENV_DIR")"
if [[ ! -f "$REPO_ROOT/setup.py" ]]; then
    echo "setup.py not found under $REPO_ROOT; run this script from the " \
         "mmsegmentation repository." >&2
    exit 1
fi
if [[ "$ENV_ABS" == "$REPO_ROOT" ]]; then
    echo "The environment directory must not be the repository root." >&2
    exit 1
fi
IS_INSIDE_REPO=1
case "$ENV_ABS" in
    "$REPO_ROOT"/*) ;;
    *) IS_INSIDE_REPO=0 ;;
esac

run() {
    echo "==> $PY $*"
    if [[ $DRY_RUN -eq 1 ]]; then
        printf '    '
        printf '%q ' "$PY" "$@"
        printf '\n'
        return 0
    fi
    "$PY" "$@"
}

run_conda() {
    echo "==> conda $*"
    if [[ $DRY_RUN -eq 1 ]]; then
        printf '    conda '
        printf '%q ' "$@"
        printf '\n'
        return 0
    fi
    conda "$@"
}

find_python() {
    local candidate version
    for candidate in python3.10 python3.11 python3; do
        if command -v "$candidate" >/dev/null 2>&1; then
            version="$("$candidate" -c \
                'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
            if [[ "$version" == "3.10" || "$version" == "3.11" ]]; then
                echo "$candidate"
                return 0
            fi
        fi
    done
    return 1
}

trap 'echo "Setup failed at line $LINENO." >&2' ERR

echo "Repository : $REPO_ROOT"
echo "Environment: $ENV_ABS"
echo "Device     : $DEVICE"
echo

# ---------------------------------------------------------------- environment
CONDA_AVAILABLE=0
if command -v conda >/dev/null 2>&1; then
    CONDA_AVAILABLE=1
fi

if [[ $FORCE -eq 1 && -d "$ENV_ABS" ]]; then
    if [[ "$IS_INSIDE_REPO" -ne 1 ]]; then
        echo "Refusing to delete an environment outside the repository: $ENV_ABS" >&2
        exit 1
    fi
    echo "==> Remove existing environment"
    if [[ $CONDA_AVAILABLE -eq 1 ]]; then
        if [[ $DRY_RUN -eq 1 ]]; then
            echo "    conda env remove -y -p $ENV_ABS"
        else
            conda env remove -y -p "$ENV_ABS" >/dev/null 2>&1 || true
        fi
    fi
    if [[ $DRY_RUN -eq 0 && -d "$ENV_ABS" ]]; then
        rm -rf -- "$ENV_ABS"
    fi
fi

if [[ $DRY_RUN -eq 1 ]]; then
    if [[ $CONDA_AVAILABLE -eq 1 ]]; then
        echo "    conda create -y -p \"$ENV_ABS\" python=3.10"
    else
        echo "    python3.10 -m venv \"$ENV_ABS\""
    fi
    PY="$ENV_ABS/bin/python"
elif [[ $CONDA_AVAILABLE -eq 1 ]]; then
    if [[ ! -x "$ENV_ABS/bin/python" ]]; then
        echo "==> Create conda environment (python 3.10)"
        if ! run_conda create -y -p "$ENV_ABS" python=3.10; then
            echo "conda create failed. Accept the Anaconda Terms of Service " \
                 "(conda tos accept) or use --env-dir with a venv instead." >&2
            exit 1
        fi
    fi
    PY="$ENV_ABS/bin/python"
else
    PY="$ENV_ABS/bin/python"
    if [[ ! -x "$PY" ]]; then
        echo "==> Create venv (conda not found)"
        PYTHON_BIN="$(find_python)" || {
            echo "Python 3.10/3.11 not found. Install it (e.g. " \
                 "sudo apt install python3.10 python3.10-venv) or install " \
                 "Miniconda (recommended)." >&2
            exit 1
        }
        if ! "$PYTHON_BIN" -m venv "$ENV_ABS"; then
            echo "venv creation failed. On Debian/Ubuntu install the venv " \
                 "package (e.g. sudo apt install python3.10-venv) or use " \
                 "Miniconda." >&2
            exit 1
        fi
    fi
fi

if [[ $DRY_RUN -eq 0 ]]; then
    PY_VERSION="$("$PY" -c \
        'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    if [[ "$PY_VERSION" != "3.10" && "$PY_VERSION" != "3.11" ]]; then
        echo "Python $PY_VERSION found, but 3.10 or 3.11 is required for " \
             "the pinned MMCV wheels. Use conda or install Python 3.10." >&2
        exit 1
    fi
fi

# ------------------------------------------------------------------ packages
run -m pip install --upgrade pip setuptools wheel
run -m pip install "numpy==1.26.4" "opencv-python==4.8.1.78" \
    --only-binary=opencv-python

if [[ "$DEVICE" == "cpu" ]]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    MMCV_INDEX="https://download.openmmlab.com/mmcv/dist/cpu/torch2.1.0/index.html"
else
    TORCH_INDEX="https://download.pytorch.org/whl/cu118"
    MMCV_INDEX="https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html"
fi

run -m pip install "torch==2.1.0" "torchvision==0.16.0" \
    --index-url "$TORCH_INDEX"
run -m pip install "mmengine==0.10.7"
run -m pip install "mmcv==2.1.0" --only-binary=mmcv -f "$MMCV_INDEX"
run -m pip install "mmdet==3.3.0" "mmpretrain==1.2.0"
run -m pip install -r requirements/runtime.txt

if [[ $WITH_TESTS -eq 1 ]]; then
    run -m pip install -r requirements/tests.txt
fi

run -m pip install -e . --no-deps --no-build-isolation

# ------------------------------------------------------------------- verify
if [[ $SKIP_VERIFY -eq 0 ]]; then
    echo "==> Verify environment"
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "    $PY tools/verify_env.py"
    else
        "$PY" tools/verify_env.py
    fi
fi

echo
echo "Environment ready: $ENV_ABS"
if [[ $CONDA_AVAILABLE -eq 1 ]]; then
    echo "Activate with: conda activate \"$ENV_ABS\""
else
    echo "Activate with: source \"$ENV_ABS/bin/activate\""
fi
echo "Train:"
echo "  python tools/train.py configs/efficientvit/efficientvit-b0_4xb3-30k_crack-512x512.py"
echo
echo "If OpenCV complains about libGL.so.1, install the system libraries:"
echo "  sudo apt-get install -y libgl1 libglib2.0-0"
