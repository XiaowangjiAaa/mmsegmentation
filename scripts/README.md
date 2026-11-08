# Environment setup scripts

## Semantic structure transition (2026-11-08)

Use `scripts/efficientvit_transition_20261108/batch_train.py` to replace the
stride 8->16 MBConv and reuse its first LiteMLA inside semantic-conditioned
structure aggregation. This round defaults to scratch initialization.
See [mechanisms, checkpoint migration and commands](efficientvit_transition_20261108/README.md).

## Encoder structure experiments (2026-11-06)

Use `scripts/efficientvit_encoder_20261106/batch_train.py` for structure-aware
downsampling, structure-conditioned LiteMLA queries and context-guided local
feature extraction. The default four variants retain the original decode head.
See [mechanisms, protocols and commands](efficientvit_encoder_20261106/README.md).

## Subpixel mechanism upgrades (2026-11-05)

Use `scripts/efficientvit_subpixel_20261105/batch_train.py` for residual-memory
writing, centered reconstruction and filtered sampling experiments.
See [training and checkpoint diagnostics](efficientvit_subpixel_20261105/README.md).

## Structure-module experiments (2026-11-04)

The three new cross-scale modules use
`scripts/efficientvit_structure_20261104/batch_train.py`.
See [module definitions and training commands](efficientvit_structure_20261104/README.md).

## Lightweight EfficientViT experiments (2026-11-02)

Use `scripts/efficientvit_light_20261102/batch_train.py` for the new paired-seed
baseline/magnitude/polarity/KV/channel/detail/CRA experiments. See
[`efficientvit_light_20261102/README.md`](efficientvit_light_20261102/README.md)
for protocols, smoke verification and metric summaries.

One-click scripts that create an isolated environment and install a
known-good stack for this repository:

| Component | Version |
| --- | --- |
| Python | 3.10 (3.11 also works) |
| PyTorch / torchvision | 2.1.0 / 0.16.0 (CUDA 11.8 by default) |
| MMCV | 2.1.0 (prebuilt wheel) |
| MMEngine | 0.10.7 |
| MMDetection / MMPretrain | 3.3.0 / 1.2.0 (ConvNeXt, Mask2Former) |
| setuptools | `<81` (torch still imports `pkg_resources`) |
| MMSegmentation | this checkout, editable |
| NumPy | 1.26.4 |
| OpenCV | 4.8.1.78 |

Both scripts use `conda` when available (creating the environment at
`./.venv-mmseg`) and fall back to a Python `venv` otherwise. They finish by
running `tools/verify_env.py`, which checks the versions, runs an EfficientViT
forward pass, builds a config and (when `dataset/` exists) builds the crack
train/val/test datasets.

## Windows

```powershell
# CUDA 11.8 (default)
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1

# CPU only
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1 -Cpu

# recreate the environment, also install test dependencies
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1 -Force -WithTests

# inspect the commands without executing anything
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1 -Cpu -DryRun
```

`scripts\setup_windows.bat` is a convenience wrapper for cmd.exe / double-click.

After setup:

```powershell
conda activate .\.venv-mmseg
python tools\train.py configs\efficientvit\efficientvit-b0_4xb3-30k_crack-512x512.py

# or without activating:
.\.venv-mmseg\python.exe tools\train.py configs\efficientvit\efficientvit-b0_4xb3-30k_crack-512x512.py
```

## Linux

```bash
# CUDA 11.8 (default)
bash scripts/setup_linux.sh

# CPU only
bash scripts/setup_linux.sh --cpu

# recreate the environment, also install test dependencies
bash scripts/setup_linux.sh --force --with-tests

# inspect the commands without executing anything
bash scripts/setup_linux.sh --cpu --dry-run
```

After setup:

```bash
conda activate ./.venv-mmseg
python tools/train.py configs/efficientvit/efficientvit-b0_4xb3-30k_crack-512x512.py

# or without activating:
./.venv-mmseg/bin/python tools/train.py configs/efficientvit/efficientvit-b0_4xb3-30k_crack-512x512.py
```

If OpenCV cannot load `libGL.so.1` on a headless Linux machine:

```bash
sudo apt-get install -y libgl1 libglib2.0-0
```

## Common options

| Windows | Linux | Meaning |
| --- | --- | --- |
| `-Cpu` | `--cpu` | install CPU-only PyTorch/MMCV |
| `-EnvDir PATH` | `--env-dir PATH` | environment location (default `./.venv-mmseg`) |
| `-Force` | `--force` | delete and recreate the environment |
| `-WithTests` | `--with-tests` | install `requirements/tests.txt` (pytest, flake8, ...) |
| `-SkipVerify` | `--skip-verify` | skip `tools/verify_env.py` |
| `-DryRun` | `--dry-run` | print commands without running them |

## Notes

- The scripts pin MMCV 2.1.0 because it is the last release compatible with
  MMSegmentation 1.2.2 (`mmcv>=2.0.0rc4,<2.2.0`) and has prebuilt wheels for
  PyTorch 2.1.0 on Windows and Linux.
- `setuptools` is pinned to `<81`: newer releases removed `pkg_resources`,
  which `torch.utils.cpp_extension` still imports through
  `mmengine.collect_env`.
- `conda` may ask you to accept the Anaconda Terms of Service on first use
  (`conda tos accept`); alternatively install the environment with a venv by
  ensuring Python 3.10/3.11 is on `PATH` and conda is not.
- No NVIDIA driver is installed automatically. CUDA 11.8 wheels need a driver
  that supports CUDA 11.8 or newer; CPU mode works without a GPU.
- The repository is installed with
  `pip install -e . --no-deps --no-build-isolation`, so changes to `mmseg/`
  (including the EfficientViT code) take effect without reinstalling.

## Batch crack training

After the environment is ready, run the 9 lightweight algorithms on two GPUs:

```powershell
.venv-mmseg\python.exe scripts\batch_train.py --gpus 0,1
```

```bash
./.venv-mmseg/bin/python scripts/batch_train.py --gpus 0,1
```

One algorithm runs per GPU; when a run finishes, the next one in the queue
starts on that GPU.  Configs live in
[`configs/crack_batch`](../configs/crack_batch), logs and the best checkpoints
are written to `work_dirs/crack_batch`.  See
[`configs/crack_batch/README.md`](../configs/crack_batch/README.md) for the
algorithm list, options and checkpoint policy.
