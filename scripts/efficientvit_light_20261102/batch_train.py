"""GPU queue for paired-seed lightweight EfficientViT experiments."""
import argparse
from collections import deque
from datetime import datetime
import json
import math
import os
import signal
from pathlib import Path
import statistics
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
VARIANTS = {name: f'configs/efficientvit_light_20261102/{name}.py' for name in (
    'baseline', 'magnitude', 'polarity', 'kv_reweight', 'late_channel',
    'detail_resize', 'detail', 'magnitude_detail', 'cra_fixed')}
METRICS = ('mIoU', 'IoU/crack', 'F1/crack', 'Precision/crack', 'Recall/crack')


def parse_args(argv=None, default_variants='baseline,magnitude,detail,magnitude_detail',
               default_work_root='work_dirs/efficientvit_light_20261102'):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variants', default=default_variants,
                        help='comma-separated variant names, or all')
    parser.add_argument('--gpus', default='0,1', help='logical visible CUDA ids')
    parser.add_argument('--seeds', default='42,43,44')
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--max-iters', type=int, default=40000)
    parser.add_argument('--scratch', action='store_true')
    parser.add_argument('--amp', action='store_true', help='default fp32')
    parser.add_argument('--work-root', default=default_work_root)
    parser.add_argument('--stop-on-failure', action='store_true')
    parser.add_argument('--poll', type=float, default=1)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.max_iters < 1 or args.poll <= 0:
        parser.error('batch-size, max-iters and poll must be positive')
    return args


def make_jobs(args, variants=None):
    variants = VARIANTS if variants is None else variants
    names = list(variants) if args.variants == 'all' else args.variants.split(',')
    seeds = [int(value) for value in args.seeds.split(',')]
    if (len(set(names)) != len(names) or not set(names) <= variants.keys()
            or len(set(seeds)) != len(seeds) or min(seeds) < 0):
        raise ValueError('Use unique known variants and unique non-negative seeds')
    if 'cra_fixed' in names and not args.scratch:
        raise ValueError('cra_fixed requires --scratch for a fair baseline control')
    if 'cra_fixed' in names and args.amp:
        raise ValueError('Use fp32 for CRA; remove --amp')
    root = Path(args.work_root)
    if not root.is_absolute():
        root = REPO_ROOT / root
    # Isolate precision and initialization as well as variant and seed.
    root = root / ('scratch' if args.scratch else 'pretrained')
    root = root / ('amp' if args.amp else 'fp32')
    return [dict(name=name, seed=seed, config=variants[name],
                 work_dir=str(root / f'seed{seed}' / name))
            for seed in seeds for name in names]


def build_command(args, job):
    options = [f'train_dataloader.batch_size={args.batch_size}',
               f'randomness.seed={job["seed"]}',
               f'train_cfg.max_iters={args.max_iters}',
               f'train_cfg.val_interval={min(1000, args.max_iters)}',
               f'param_scheduler.1.end={args.max_iters}',
               f'default_hooks.checkpoint.interval={args.max_iters + 1}']
    # Keep a valid warmup even for 1-2 iteration smoke runs.
    warmup = min(1500, max(1, args.max_iters // 10))
    if args.max_iters == 1:
        options += ['param_scheduler.0.end=1',
                    'param_scheduler.1.begin=0']
    else:
        options += [f'param_scheduler.0.end={warmup}',
                    f'param_scheduler.1.begin={warmup}']
    if args.scratch:
        options.append('load_from=None')
    cmd = [sys.executable, 'tools/train.py', job['config'],
           '--work-dir', job['work_dir']]
    if args.amp:
        cmd.append('--amp')
    return cmd + ['--cfg-options', *options]


def read_metrics(work_dir):
    files = sorted(Path(work_dir).glob('*/vis_data/scalars.json'))
    if not files:
        return {}
    rows = []
    for line in files[-1].read_text(encoding='utf-8').splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if 'mIoU' in row and math.isfinite(row['mIoU']):
            rows.append(row)
    if not rows:
        return {}
    best = max(rows, key=lambda row: row['mIoU'])
    return dict(best_step=best.get('step'), last_val_step=rows[-1].get('step'),
                **{key: best[key] for key in METRICS if key in best})


def aggregate(records):
    result = {}
    references = {
        name: {r['seed']: r.get('metrics', {}).get('IoU/crack')
               for r in records if r['name'] == name and r['status'] == 'ok'}
        for name in ('baseline', 'detail_resize', 'detail', 'subpixel')}
    for name in dict.fromkeys(r['name'] for r in records):
        runs = [r for r in records if r['name'] == name and r['status'] == 'ok']
        summary = {'completed_runs': len(runs)}
        for key in METRICS:
            values = [r['metrics'][key] for r in runs if key in r.get('metrics', {})]
            if values:
                summary[key] = dict(n=len(values), mean=statistics.mean(values),
                                    std=statistics.stdev(values) if len(values) > 1 else None)
        paired = {}
        for ref_name, ref in references.items():
            pairs = [r['metrics']['IoU/crack'] - ref[r['seed']] for r in runs
                     if r.get('metrics', {}).get('IoU/crack') is not None
                     and ref.get(r['seed']) is not None]
            if pairs:
                paired[ref_name] = dict(
                    n=len(pairs), mean=statistics.mean(pairs),
                    std=statistics.stdev(pairs) if len(pairs) > 1 else None)
        if paired:
            summary['paired_deltas_IoU/crack'] = paired
        if 'baseline' in paired:
            # Preserve the legacy baseline field for existing consumers.
            summary['paired_delta_IoU/crack'] = paired['baseline']
        result[name] = summary
    return result


def save_summary(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(runs=records, aggregate=aggregate(records)),
                                    indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def main(argv=None, variants=None,
         default_variants='baseline,magnitude,detail,magnitude_detail',
         default_work_root='work_dirs/efficientvit_light_20261102'):
    variants = VARIANTS if variants is None else variants
    args = parse_args(argv, default_variants, default_work_root)
    if args.list:
        for name, config in variants.items():
            print(f'{name:20s} {config}')
        return 0
    jobs = make_jobs(args, variants)
    gpus = args.gpus.split(',')
    if len(set(gpus)) != len(gpus) or not all(g.isdigit() for g in gpus):
        raise ValueError('gpus must be unique non-negative logical indices')
    for job in jobs:
        if not (REPO_ROOT / job['config']).is_file():
            raise FileNotFoundError(job['config'])
    if args.dry_run:
        for index, job in enumerate(jobs):
            print(f'GPU slot {gpus[index % len(gpus)]}: {build_command(args, job)}')
        return 0
    import torch
    if max(map(int, gpus)) >= torch.cuda.device_count():
        raise ValueError('Requested GPU is unavailable; check CUDA_VISIBLE_DEVICES')
    inherited = os.environ.get('CUDA_VISIBLE_DEVICES')
    slots = inherited.split(',') if inherited else None
    devices = {g: slots[int(g)] if slots else g for g in gpus}
    if not args.scratch:
        checkpoint = REPO_ROOT / 'work_dirs/pretrained/efficientvit_b0_crack.pth'
        if not checkpoint.is_file():
            subprocess.run([sys.executable, 'scripts/prepare_efficientvit_b0.py'],
                           cwd=REPO_ROOT, check=True)
    for job in jobs:
        directory = Path(job['work_dir'])
        if directory.exists():
            raise FileExistsError(f'Use a new --work-root; existing run: {directory}')
    root = Path(jobs[0]['work_dir']).parents[1]
    summary_path = root / 'batch_summary.json'
    queue, active, records = deque(jobs), {}, []
    interrupted = False

    def finish(gpu, status):
        state = active.pop(gpu)
        proc, log = state.pop('proc'), state.pop('log')
        log.close()
        state.update(return_code=proc.returncode, status=status,
                     finished_at=datetime.now().isoformat(),
                     metrics=read_metrics(state['work_dir']))
        if status == 'ok' and not state['metrics']:
            state['status'] = status = 'no_validation_metrics'
        records.append(state)
        save_summary(summary_path, records)
        print(f'GPU {gpu}: {state["name"]} seed{state["seed"]} {status}', flush=True)

    def stop_active():
        for gpu, state in list(active.items()):
            if state['proc'].poll() is None:
                if os.name == 'nt':
                    subprocess.run(['taskkill', '/T', '/F', '/PID',
                                    str(state['proc'].pid)],
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, check=False)
                else:
                    os.killpg(state['proc'].pid, signal.SIGTERM)
                try:
                    state['proc'].wait(timeout=10)
                except subprocess.TimeoutExpired:
                    if os.name == 'nt':
                        state['proc'].kill()
                    else:
                        os.killpg(state['proc'].pid, signal.SIGKILL)
                    state['proc'].wait()
            finish(gpu, 'interrupted')

    try:
        while queue or active:
            for gpu in gpus:
                if gpu in active or not queue:
                    continue
                job = queue.popleft()
                directory = Path(job['work_dir'])
                directory.mkdir(parents=True, exist_ok=False)
                cmd = build_command(args, job)
                env = os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=devices[gpu], PYTHONUTF8='1',
                           PYTHONUNBUFFERED='1')
                env['PYTHONPATH'] = str(REPO_ROOT) + os.pathsep + env.get('PYTHONPATH', '')
                log_path = directory / 'train.log'
                log = log_path.open('w', encoding='utf-8')
                try:
                    proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env,
                                            stdout=log, stderr=subprocess.STDOUT,
                                            start_new_session=os.name != 'nt')
                except Exception as exc:
                    log.close()
                    records.append(dict(job, status='launch_failed', error=str(exc)))
                    raise
                active[gpu] = dict(job, gpu=gpu, device=devices[gpu], command=cmd,
                                   log_path=str(log_path), log=log, proc=proc,
                                   started_at=datetime.now().isoformat())
                print(f'GPU {gpu}: start {job["name"]} seed{job["seed"]}; {log_path}', flush=True)
            failed = False
            for gpu, state in list(active.items()):
                code = state['proc'].poll()
                if code is not None:
                    finish(gpu, 'ok' if code == 0 else 'failed')
                    failed |= records[-1]['status'] != 'ok'
            if failed and args.stop_on_failure:
                interrupted = True
                break
            if active:
                time.sleep(args.poll)
    except KeyboardInterrupt:
        interrupted = True
    finally:
        stop_active()
        for job in queue:
            records.append(dict(job, status='not_started'))
        save_summary(summary_path, records)
    return 1 if interrupted or any(r['status'] != 'ok' for r in records) else 0


if __name__ == '__main__':
    sys.exit(main())
