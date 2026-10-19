#!/usr/bin/env python
"""Download and convert the official EfficientViT-B0 segmentation checkpoint.

The converted checkpoint is written to
``work_dirs/pretrained/efficientvit_b0_crack.pth`` and is used by
``configs/crack_batch/efficientvit_b0.py`` (``load_from``).

Example:
    .venv-mmseg\\python.exe scripts\\prepare_efficientvit_b0.py
"""
import argparse
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FILE_ID = '1Ix1Dh3xlpaf0Wzh01Xmo-hAYkoXt1EAD'
DEFAULT_URL = (
    f'https://drive.usercontent.google.com/download?id={FILE_ID}'
    '&export=download&confirm=t')
DEFAULT_SRC = REPO_ROOT / 'work_dirs/pretrained/efficientvit_b0_cityscapes.pt'
DEFAULT_DST = REPO_ROOT / 'work_dirs/pretrained/efficientvit_b0_crack.pth'
CONVERTER = REPO_ROOT / 'tools/model_converters/efficientvit_to_mmseg.py'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--src', default=str(DEFAULT_SRC))
    parser.add_argument('--dst', default=str(DEFAULT_DST))
    parser.add_argument('--url', default=DEFAULT_URL)
    parser.add_argument('--num-classes', type=int, default=2)
    parser.add_argument(
        '--force', action='store_true', help='re-download and reconvert')
    return parser.parse_args()


def download(url, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f'downloading: {url}')
    request = urllib.request.Request(
        url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(request, timeout=120) as response, open(
            dst, 'wb') as file:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            file.write(chunk)
    print(f'saved: {dst} ({dst.stat().st_size} bytes)')


def main():
    args = parse_args()
    src = Path(args.src)
    dst = Path(args.dst)
    if args.force or not src.exists():
        download(args.url, src)
    if args.force or not dst.exists():
        subprocess.run(
            [
                sys.executable, str(CONVERTER), '--src',
                str(src), '--dst',
                str(dst), '--num-classes',
                str(args.num_classes)
            ],
            cwd=REPO_ROOT,
            check=True)
    else:
        print(f'converted checkpoint already exists: {dst}')
    print(f'EfficientViT-B0 pretrained checkpoint: {dst}')


if __name__ == '__main__':
    main()
