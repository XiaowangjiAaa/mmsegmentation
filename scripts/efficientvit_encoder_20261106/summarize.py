"""Recompute same-seed comparisons against baseline and original subpixel."""
from batch_train import load_shared

if __name__ == '__main__':
    runner = load_shared('batch_train')
    load_shared('summarize').main(
        read_metrics_fn=runner.read_metrics,
        save_summary_fn=runner.save_summary)
