"""Batch runner for the 2026-10-23 EfficientViT attention variants.

Each selected variant runs on one GPU; when a GPU finishes, the next queued
variant starts on it. Example (two GPUs):

    python scripts/efficientvit_attn_20261023/batch_train_attn.py \
        --gpus 0,1 --variants coord,hybrid,window

The script keeps the same conventions as scripts/batch_train.py:
per-run logs under ``work_dirs/efficientvit_attn_20261023/batch/logs``,
``batch_summary.json`` at the work root, and ``best_mIoU_iter_*.pth`` in each
variant directory.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
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
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--gpus', default='0,1', help='comma separated GPU ids, e.g. 0,1')
    parser.add_argument(
        '--variants',
        default='replace_hybrid,replace_window,replace_global',
        help='comma separated: baseline,coord,hybrid,window, '
        'replace_window,replace_global,replace_hybrid')
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument(
        '--max-iters',
        type=int,
        default=0,
        help='0 keeps the config (40k); use a small value for a smoke test')
    parser.add_argument(
        '--work-root',
        default='work_dirs/efficientvit_attn_20261023/batch',
        help='root directory for logs and per-variant work dirs')
    parser.add_argument(
        '--amp',
        action='store_true',
        help='enable AMP for all runs; default is fp32')
    parser.add_argument(
        '--stream-logs',
        dest='stream_logs',
        action='store_true',
        default=True,
        help='stream child logs to the terminal (default)')
    parser.add_argument(
        '--no-stream-logs',
        dest='stream_logs',
        action='store_false',
        help='write child logs only to files')
    parser.add_argument(
        '--stop-on-failure',
        action='store_true',
        help='stop the queue when a run fails')
    parser.add_argument(
        '--poll', type=float, default=10,
        help='seconds between process checks')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--list', action='store_true')
    return parser.parse_args()


def parse_gpus(value):
    gpus = [item.strip() for item in value.split(',') if item.strip()]
    if not gpus:
        raise SystemExit('At least one GPU id is required.')
    return gpus


def check_runtime(gpus):
    import torch
    count = torch.cuda.device_count()
    print(f'Visible CUDA devices: {count}')
    if count == 0:
        raise SystemExit('CUDA is not available.')
    if len(gpus) > count:
        raise SystemExit(
            f'Requested {len(gpus)} GPUs ({gpus}), but only {count} visible.')
    for gpu in gpus:
        if gpu.isdigit() and int(gpu) >= count:
            raise SystemExit(
                f'GPU id {gpu} is out of range; '
                f'visible ids are 0..{count - 1}.')


def build_command(args, config, work_dir):
    cmd = [
        sys.executable, 'tools/train.py', config, '--work-dir',
        str(work_dir)
    ]
    if args.amp:
        cmd.append('--amp')
    cfg_options = [f'train_dataloader.batch_size={args.batch_size}']
    if args.max_iters > 0:
        cfg_options.extend([
            f'train_cfg.max_iters={args.max_iters}',
            'train_cfg.val_interval=1000',
        ])
    cmd.extend(['--cfg-options', *cfg_options])
    return cmd


def pump_output(stream, log_file, prefix):
    try:
        for line in iter(stream.readline, ''):
            log_file.write(line)
            log_file.flush()
            print(f'{prefix} {line}', end='', flush=True)
    finally:
        try:
            stream.close()
        except OSError:
            pass


def read_tail(path, count=15):
    try:
        with open(path, encoding='utf-8', errors='replace') as file:
            lines = file.readlines()
    except OSError:
        return []
    return [line.rstrip('\n') for line in lines[-count:]]


def ensure_pretrained():
    checkpoint = REPO_ROOT / 'work_dirs/pretrained/efficientvit_b0_crack.pth'
    if checkpoint.is_file():
        return
    print('==> EfficientViT pretrained checkpoint missing; preparing it')
    subprocess.run(
        [sys.executable, 'scripts/prepare_efficientvit_b0.py'],
        cwd=REPO_ROOT,
        check=True)


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
    if not names:
        raise SystemExit('No variant selected.')

    gpus = parse_gpus(args.gpus)
    work_root = REPO_ROOT / args.work_root
    log_dir = work_root / 'logs'
    work_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    missing = [
        name for name in names
        if not (REPO_ROOT / VARIANTS[name]).is_file()
    ]
    if missing:
        raise SystemExit(f'Missing configs: {missing}')
    if not (REPO_ROOT / 'tools' / 'train.py').is_file():
        raise SystemExit('tools/train.py not found.')

    if args.dry_run:
        print(f'Repo      : {REPO_ROOT}')
        print(f'GPUs      : {gpus}')
        print(f'Variants ({len(names)}): {names}')
        for index, name in enumerate(names):
            gpu = gpus[index % len(gpus)]
            work_dir = work_root / name
            cmd = build_command(args, VARIANTS[name], work_dir)
            print(f'\n[{name}] CUDA_VISIBLE_DEVICES={gpu} '
                  f'{" ".join(cmd)}')
        return

    check_runtime(gpus)
    ensure_pretrained()

    print(f'Repo      : {REPO_ROOT}')
    print(f'GPUs      : {gpus}')
    print(f'Work root : {work_root}')
    print(f'Logs      : {log_dir}')
    print(f'Variants ({len(names)}): {names}')
    print(f'AMP       : {args.amp}\n')

    queue = deque(names)
    active = {}
    records = []
    started_at = datetime.now()

    def start_run(name, gpu):
        work_dir = work_root / name
        cmd = build_command(args, VARIANTS[name], work_dir)
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = gpu
        env['PYTHONUTF8'] = '1'
        env['PYTHONUNBUFFERED'] = '1'
        env['PYTHONPATH'] = str(REPO_ROOT) + os.pathsep + env.get(
            'PYTHONPATH', '')

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = log_dir / f'{name}_{timestamp}.log'
        log_file = open(log_path, 'w', encoding='utf-8')
        print(f'[{datetime.now():%H:%M:%S}] GPU{gpu} start {name}')
        try:
            if args.stream_logs:
                proc = subprocess.Popen(
                    cmd,
                    cwd=REPO_ROOT,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1)
                pump = threading.Thread(
                    target=pump_output,
                    args=(proc.stdout, log_file, f'[{name}]'),
                    daemon=True)
                pump.start()
            else:
                proc = subprocess.Popen(
                    cmd,
                    cwd=REPO_ROOT,
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True)
                pump = None
        except Exception as exc:
            log_file.close()
            print(f'[{datetime.now():%H:%M:%S}] GPU{gpu} launch failed '
                  f'{name}: {exc}', file=sys.stderr)
            records.append({
                'name': name,
                'config': VARIANTS[name],
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
            'config': VARIANTS[name],
            'gpu': gpu,
            'work_dir': str(work_dir),
            'log_path': str(log_path),
            'proc': proc,
            'log_file': log_file,
            'pump_thread': pump,
            'started_at': datetime.now().isoformat(timespec='seconds'),
        }

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

    def close_state(state):
        pump = state.pop('pump_thread', None)
        if pump is not None:
            pump.join(timeout=10)
        state.pop('log_file').close()
        state.pop('proc')

    try:
        while queue or active:
            for gpu in gpus:
                if gpu not in active and queue:
                    name = queue.popleft()
                    state = start_run(name, gpu)
                    if state is not None:
                        active[gpu] = state

            for gpu, state in list(active.items()):
                return_code = state['proc'].poll()
                if return_code is None:
                    continue
                close_state(state)
                state['return_code'] = return_code
                state['status'] = 'ok' if return_code == 0 else 'failed'
                state['finished_at'] = datetime.now().isoformat(
                    timespec='seconds')
                if return_code != 0:
                    tail = read_tail(state['log_path'])
                    state['error_tail'] = tail
                    print(f'--- {state["name"]} failed (rc={return_code}), '
                          f'last {len(tail)} lines ---')
                    for line in tail:
                        print(line)
                    print('--- end of failure tail ---')
                records.append(state)
                print(f'[{datetime.now():%H:%M:%S}] GPU{gpu} done '
                      f'{state["name"]} (rc={return_code}) '
                      f'log={state["log_path"]}')
                del active[gpu]
                if return_code != 0 and args.stop_on_failure:
                    for other in active.values():
                        stop_process(other['proc'])
                        close_state(other)
                        other['return_code'] = -1
                        other['status'] = 'interrupted'
                        other['finished_at'] = datetime.now().isoformat(
                            timespec='seconds')
                        records.append(other)
                    active.clear()
                    queue.clear()
                    break

            if active:
                time.sleep(args.poll)
    except KeyboardInterrupt:
        print('\nInterrupted; stopping running jobs...')
        for state in active.values():
            stop_process(state['proc'])
            close_state(state)
            state['return_code'] = -1
            state['status'] = 'interrupted'
            state['finished_at'] = datetime.now().isoformat(timespec='seconds')
            records.append(state)
        active.clear()

    summary_path = work_root / 'batch_summary.json'
    summary_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding='utf-8')

    print('\n================ Summary ================')
    for record in records:
        print(f'{record["name"]:12s} GPU{record["gpu"]}  '
              f'{record["status"]:12s} rc={record.get("return_code", "-")}')
    print(f'Summary: {summary_path}')
    if records:
        elapsed = datetime.now() - started_at
        print(f'Total time: {elapsed}')


if __name__ == '__main__':
    main()
