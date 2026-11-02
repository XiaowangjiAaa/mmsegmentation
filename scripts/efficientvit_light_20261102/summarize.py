"""Rebuild best-mIoU and paired-seed summaries from saved scalar logs."""
import argparse
from pathlib import Path

from batch_train import read_metrics, save_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path,
                        help='one protocol root, e.g. .../scratch/fp32')
    args = parser.parse_args()
    records = []
    for seed_dir in sorted(args.root.glob('seed*')):
        if not seed_dir.is_dir() or not seed_dir.name[4:].isdigit():
            continue
        for run in sorted(seed_dir.iterdir()):
            if not run.is_dir():
                continue
            metrics = read_metrics(run)
            if metrics:
                # Scalar files alone do not prove process completion.
                records.append(dict(name=run.name, seed=int(seed_dir.name[4:]),
                                    work_dir=str(run), status='ok', metrics=metrics))
    if not records:
        raise SystemExit('No validation scalars found at root/seed*/variant')
    # Never overwrite the authoritative process-status batch_summary.json.
    target = args.root / 'metrics_summary.json'
    save_summary(target, records)
    print(f'{len(records)} runs with validation metrics: {target}')
    print('This metric-only summary may include partial runs; check batch_summary.json.')


if __name__ == '__main__':
    main()
