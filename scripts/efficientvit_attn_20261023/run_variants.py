"""Run the EfficientViT-B0 attention ablation variants sequentially.

Examples:
    python scripts/efficientvit_attn_20261023/run_variants.py --dry-run
    python scripts/efficientvit_attn_20261023/run_variants.py --gpus 0
    python scripts/efficientvit_attn_20261023/run_variants.py --variants coord
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VARIANTS = {
    'baseline': 'configs/crack_batch/efficientvit_b0.py',
    'coord':
    'configs/efficientvit_attn_20261023/'
    'efficientvit-b0-coord_crack-512x512.py',
    'hybrid':
    'configs/efficientvit_attn_20261023/'
    'efficientvit-b0-hybrid_crack-512x512.py',
    'window':
    'configs/efficientvit_attn_20261023/'
    'efficientvit-b0-window_crack-512x512.py',
    'replace_window':
    'configs/efficientvit_attn_20261023/'
    'efficientvit-b0-replace-window_crack-512x512.py',
    'replace_global':
    'configs/efficientvit_attn_20261023/'
    'efficientvit-b0-replace-global_crack-512x512.py',
    'replace_hybrid':
    'configs/efficientvit_attn_20261023/'
    'efficientvit-b0-replace-hybrid_crack-512x512.py',
    'asgla':
    'configs/efficientvit_attn_20261023/'
    'efficientvit-b0-asgla_crack-512x512.py',
    'cdgl':
    'configs/efficientvit_attn_20261023/'
    'efficientvit-b0-cdgl_crack-512x512.py',
    'cdgl_asgla':
    'configs/efficientvit_attn_20261023/'
    'efficientvit-b0-cdgl-asgla_crack-512x512.py',
    'foveal':
    'configs/efficientvit_attn_20261023/'
    'efficientvit-b0-foveal_crack-512x512.py',
    'oala':
    'configs/efficientvit_attn_20261023/'
    'efficientvit-b0-oala_crack-512x512.py',
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--variants',
        default='oala',
        help='comma separated: baseline,coord,hybrid,window, '
        'replace_window,replace_global,replace_hybrid, '
        'asgla,cdgl,cdgl_asgla,foveal,oala')
    parser.add_argument('--gpus', default='0', help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument(
        '--max-iters',
        type=int,
        default=0,
        help='0 keeps the config (40k); use a small value for smoke tests')
    parser.add_argument(
        '--work-root',
        default='work_dirs/efficientvit_attn_20261023',
        help='root directory for per-variant work dirs')
    parser.add_argument(
        '--amp',
        action='store_true',
        help='enable AMP; default is fp32 for numerical stability')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--list', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.list:
        for name, config in VARIANTS.items():
            print(f'{name:10s} {config}')
        return

    names = [name.strip() for name in args.variants.split(',') if name.strip()]
    unknown = sorted(set(names) - set(VARIANTS))
    if unknown:
        raise SystemExit(
            f'Unknown variants: {unknown}; choose from {list(VARIANTS)}')

    work_root = REPO_ROOT / args.work_root
    work_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(args.gpus)
    env['PYTHONUTF8'] = '1'
    env['PYTHONUNBUFFERED'] = '1'
    env['PYTHONPATH'] = str(REPO_ROOT) + os.pathsep + env.get('PYTHONPATH', '')

    for name in names:
        config = VARIANTS[name]
        work_dir = work_root / name
        cmd = [
            sys.executable, 'tools/train.py', config, '--work-dir',
            str(work_dir)
        ]
        if args.amp:
            cmd.append('--amp')
        cfg_options = [f'train_dataloader.batch_size={args.batch_size}']
        if args.max_iters > 0:
            cfg_options.append(f'train_cfg.max_iters={args.max_iters}')
            cfg_options.append('train_cfg.val_interval=1000')
        cmd.extend(['--cfg-options', *cfg_options])

        print(f'\n===== {name} =====')
        print(' '.join(cmd))
        if args.dry_run:
            continue
        subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)
        print(f'===== {name} finished =====')


if __name__ == '__main__':
    main()
