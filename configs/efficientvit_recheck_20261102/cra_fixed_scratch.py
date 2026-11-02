# Corrected seed-grid coverage and a single context identity shortcut.
# Historical CRA scores do not describe this implementation.
_base_ = ['../efficientvit_cra_20261031/efficientvit-b0-cra_crack-512x512.py']

load_from = None
randomness = dict(seed=42)
work_dir = 'work_dirs/efficientvit_recheck_20261102/seed42/cra_fixed'
