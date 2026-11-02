# Paired from-scratch control for the corrected CRA implementation.
_base_ = ['../crack_batch/efficientvit_b0.py']

load_from = None
randomness = dict(seed=42)
work_dir = 'work_dirs/efficientvit_recheck_20261102/seed42/baseline'
