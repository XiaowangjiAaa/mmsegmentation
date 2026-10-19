#!/usr/bin/env python
"""Run the crack-segmentation experiments sequentially on multiple GPUs.

Each GPU owns one training process.  When a process finishes (successfully or
not), the next algorithm in the queue is started on the freed GPU.

Example (Windows, two GPUs):

    .venv-mmseg\\python.exe scripts\\batch_train.py --gpus 0,1

The queue uses the lightweight variant of every algorithm:

    convnext_tiny, deeplabv3plus_r18, mask2former_r50, segformer_b0,
    segnext_t, swin_tiny, twins_pcpvt_s, unet_s5, efficientvit_b0

All runs use the crack dataset, batch size 12 per GPU, 40k iterations,
validation every 1000 iterations, and only keep the best checkpoint by mIoU.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

RUNS = [
    ('convnext_tiny', 'configs/crack_batch/convnext_tiny.py'),
    ('deeplabv3plus_r18', 'configs/crack_batch/deeplabv3plus_r18.py'),
    ('mask2former_r50', 'configs/crack_batch/mask2former_r50.py'),
    ('segformer_b0', 'configs/crack_batch/segformer_b0.py'),
    ('segnext_t', 'configs/crack_batch/segnext_t.py'),
    ('swin_tiny', 'configs/crack_batch/swin_tiny.py'),
    ('twins_pcpvt_s', 'configs/crack_batch/twins_pcpvt_s.py'),
    ('unet_s5', 'configs/crack_batch/unet_s5.py'),
    ('efficientvit_b0', 'configs/crack_batch/efficientvit_b0.py'),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run crack experiments on multiple GPUs, one per GPU.')
    parser.add_argument(
        '--gpus', default='0,1',
        help='comma separated GPU ids, e.g. "0,1" or "0"')
    parser.add_argument(
        '--batch-size', type=int, default=12,
        help='per-GPU batch size (default: 12; 0 keeps the config value)')
    parser.add_argument(
        '--amp', dest='amp', action='store_true', default=True,
        help='enable AMP (default)')
    parser.add_argument(
        '--no-amp', dest='amp', action='store_false',
        help='disable AMP')
    parser.add_argument(
        '--work-root', default='work_dirs/crack_batch',
        help='root directory for work dirs and logs')
    parser.add_argument(
        '--log-dir', default=None,
        help='log directory (default: <work-root>/logs)')
    parser.add_argument(
        '--python', default=sys.executable,
        help='Python interpreter used for tools/train.py')
    parser.add_argument(
        '--poll', type=float, default=10,
        help='seconds between process checks (default: 10)')
    parser.add_argument(
        '--log-interval', type=int, default=0,
        help='iterations between loss log lines; 0 keeps the config (50), '
        '1 logs every iteration')
    parser.add_argument(
        '--only', default=None,
        help='comma separated run names to execute')
    parser.add_argument(
        '--skip', default=None,
        help='comma separated run names to skip')
    parser.add_argument(
        '--stop-on-failure', action='store_true',
        help='stop the queue when a run fails')
    parser.add_argument(
        '--dry-run', action='store_true',
        help='print commands and exit without training')
    parser.add_argument(
        '--list', action='store_true',
        help='list run names and configs')
    return parser.parse_args()


def parse_names(value):
    if not value:
        return None
    return [name.strip() for name in value.split(',') if name.strip()]


def select_runs(args):
    runs = list(RUNS)
    only = parse_names(args.only)
    skip = set(parse_names(args.skip) or [])
    if only is not None:
        unknown = sorted(set(only) - {name for name, _ in runs})
        if unknown:
            raise SystemExit(f'Unknown --only names: {unknown}')
        runs = [run for run in runs if run[0] in only]
    if skip:
        unknown = sorted(skip - {name for name, _ in runs})
        if unknown:
            raise SystemExit(f'Unknown --skip names: {unknown}')
        runs = [run for run in runs if run[0] not in skip]
    if not runs:
        raise SystemExit('No run selected.')
    return runs


def build_command(args, config, work_dir):
    cmd = [args.python, 'tools/train.py', config, '--work-dir', str(work_dir)]
    if args.amp:
        cmd.append('--amp')
    cfg_options = []
    if args.batch_size > 0:
        cfg_options.append(
            f'train_dataloader.batch_size={args.batch_size}')
    if args.log_interval > 0:
        cfg_options.append(
            f'default_hooks.logger.interval={args.log_interval}')
    if cfg_options:
        cmd.extend(['--cfg-options', *cfg_options])
    return cmd


def stop_process(proc):
    if proc.poll() is not None:
        return
    if os.name == 'nt':
        subprocess.run(
            ['taskkill', '/T', '/F', '/PID', str(proc.pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False)
    else:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def check_runtime(args, gpus):
    """Validate visible GPUs and disable AMP when running on CPU."""
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment specific
        print(f'WARNING: could not import torch ({exc}); skipping GPU check.',
              file=sys.stderr)
        return
    if not torch.cuda.is_available():
        if args.amp:
            print('WARNING: CUDA is not available; disabling AMP.',
                  file=sys.stderr)
            args.amp = False
        return
    count = torch.cuda.device_count()
    if len(gpus) > count:
        raise SystemExit(
            f'Requested {len(gpus)} GPUs ({gpus}) but only {count} CUDA '
            'device(s) are visible. Check CUDA_VISIBLE_DEVICES or use '
            '--gpus with the correct ids.')


def ensure_efficientvit_pretrained(runs):
    """Download/convert the EfficientViT-B0 checkpoint when it is missing."""
    if not any(name == 'efficientvit_b0' for name, _ in runs):
        return
    checkpoint = (
        REPO_ROOT / 'work_dirs/pretrained/efficientvit_b0_crack.pth')
    if checkpoint.is_file():
        return
    print('==> Preparing EfficientViT-B0 pretrained checkpoint')
    try:
        subprocess.run(
            [sys.executable, 'scripts/prepare_efficientvit_b0.py'],
            cwd=REPO_ROOT,
            check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            'Failed to prepare EfficientViT-B0 pretrained checkpoint. '
            'Run scripts/prepare_efficientvit_b0.py manually and retry.'
        ) from exc


def main():
    args = parse_args()
    if args.list:
        for name, config in select_runs(args):
            print(f'{name:<20} {config}')
        return

    gpus = [gpu.strip() for gpu in args.gpus.split(',') if gpu.strip()]
    if not gpus:
        raise SystemExit('--gpus must not be empty.')
    if len(set(gpus)) != len(gpus):
        raise SystemExit('Duplicate GPU ids are not allowed.')

    runs = select_runs(args)
    work_root = (REPO_ROOT / args.work_root).resolve()
    log_dir = (REPO_ROOT / args.log_dir).resolve() \
        if args.log_dir else work_root / 'logs'
    work_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    missing = [
        config for _, config in runs if not (REPO_ROOT / config).is_file()
    ]
    if missing:
        raise SystemExit(f'Missing config files: {missing}')
    if not (REPO_ROOT / 'tools' / 'train.py').is_file():
        raise SystemExit(f'tools/train.py not found under {REPO_ROOT}')
    if not (REPO_ROOT / 'dataset' / 'Image').is_dir():
        print('WARNING: dataset/Image not found; training may fail.',
              file=sys.stderr)

    queue = deque(runs)
    pending_names = [name for name, _ in runs]
    active = {}
    records = []
    started_at = datetime.now()

    if args.dry_run:
        print(f'Repo      : {REPO_ROOT}')
        print(f'GPUs      : {gpus}')
        print(f'Runs ({len(runs)}): {pending_names}')
        if (any(name == 'efficientvit_b0' for name, _ in runs)
                and not (REPO_ROOT /
                         'work_dirs/pretrained/efficientvit_b0_crack.pth'
                         ).is_file()):
            print('NOTE: EfficientViT-B0 pretrained checkpoint is missing; '
                  'run scripts/prepare_efficientvit_b0.py first.')
        for index, (name, config) in enumerate(runs):
            gpu = gpus[index % len(gpus)]
            work_dir = work_root / name
            cmd = build_command(args, config, work_dir)
            env_prefix = f'CUDA_VISIBLE_DEVICES={gpu} '
            print(f'\n[{name}] {env_prefix}{" ".join(cmd)}')
        return

    print(f'Repo      : {REPO_ROOT}')
    print(f'GPUs      : {gpus}')
    print(f'Work root : {work_root}')
    print(f'Logs      : {log_dir}')
    print(f'Runs ({len(runs)}): {pending_names}\n')

    check_runtime(args, gpus)
    ensure_efficientvit_pretrained(runs)

    def start_run(run, gpu):
        name, config = run
        work_dir = work_root / name
        cmd = build_command(args, config, work_dir)
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = gpu
        env['PYTHONUNBUFFERED'] = '1'
        env['PYTHONUTF8'] = '1'
        env['PYTHONPATH'] = str(REPO_ROOT) + os.pathsep + env.get(
            'PYTHONPATH', '')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = log_dir / f'{name}_{timestamp}.log'
        log_file = open(log_path, 'w', encoding='utf-8')
        print(f'[{datetime.now():%H:%M:%S}] GPU{gpu} start {name}')
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=REPO_ROOT,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True)
        except Exception as exc:  # pragma: no cover - environment specific
            log_file.close()
            print(f'[{datetime.now():%H:%M:%S}] GPU{gpu} launch failed '
                  f'{name}: {exc}', file=sys.stderr)
            records.append({
                'name': name,
                'config': config,
                'gpu': gpu,
                'work_dir': str(work_dir),
                'log_path': str(log_path),
                'status': 'launch_failed',
                'error': str(exc),
                'started_at': datetime.now().isoformat(timespec='seconds'),
                'finished_at': datetime.now().isoformat(timespec='seconds'),
            })
            return None
        return {
            'name': name,
            'config': config,
            'gpu': gpu,
            'work_dir': str(work_dir),
            'log_path': str(log_path),
            'proc': proc,
            'log_file': log_file,
            'started_at': datetime.now().isoformat(timespec='seconds'),
        }

    try:
        while queue or active:
            for gpu in gpus:
                if gpu not in active and queue:
                    run = queue.popleft()
                    state = start_run(run, gpu)
                    if state is not None:
                        active[gpu] = state

            for gpu, state in list(active.items()):
                return_code = state['proc'].poll()
                if return_code is None:
                    continue
                state['log_file'].close()
                state.pop('proc')
                state.pop('log_file')
                state['return_code'] = return_code
                state['status'] = 'ok' if return_code == 0 else 'failed'
                state['finished_at'] = datetime.now().isoformat(
                    timespec='seconds')
                records.append(state)
                print(f'[{datetime.now():%H:%M:%S}] GPU{gpu} done '
                      f'{state["name"]} (rc={return_code}) '
                      f'log={state["log_path"]}')
                del active[gpu]
                if return_code != 0 and args.stop_on_failure:
                    for other in active.values():
                        stop_process(other.pop('proc'))
                        other.pop('log_file').close()
                        other['return_code'] = -1
                        other['status'] = 'interrupted'
                        other['finished_at'] = datetime.now().isoformat(
                            timespec='seconds')
                        records.append(other)
                    active.clear()
                    summary_path = work_root / 'batch_summary.json'
                    summary_path.write_text(
                        json.dumps(records, indent=2, ensure_ascii=False),
                        encoding='utf-8')
                    raise SystemExit(1)

            if active:
                time.sleep(args.poll)
    except KeyboardInterrupt:
        print('\nInterrupted; stopping running jobs...')
        for state in active.values():
            stop_process(state.pop('proc'))
            state.pop('log_file').close()
            state['return_code'] = -1
            state['status'] = 'interrupted'
            state['finished_at'] = datetime.now().isoformat(timespec='seconds')
            records.append(state)
        active.clear()
        summary_path = work_root / 'batch_summary.json'
        summary_path.write_text(
            json.dumps(records, indent=2, ensure_ascii=False),
            encoding='utf-8')
        print(f'Summary: {summary_path}')
        raise SystemExit(130)

    summary_path = work_root / 'batch_summary.json'
    summary_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding='utf-8')

    print('\n================ Summary ================')
    for record in records:
        print(f'{record["name"]:<20} GPU{record["gpu"]:<3} '
              f'{record["status"]:<12} rc={record.get("return_code")}')
    print(f'Summary: {summary_path}')
    print(f'Total time: {datetime.now() - started_at}')

    if any(record['status'] != 'ok' for record in records):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
