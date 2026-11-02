_base_ = ['../crack_batch/efficientvit_b0.py']

# Dataset split seed and training seed are separate controls.
randomness = dict(seed=42)
