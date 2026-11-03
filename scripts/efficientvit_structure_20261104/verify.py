"""Verify all structure configs with true segmentation losses and backward."""
from batch_train import VARIANTS, load_shared

if __name__ == '__main__':
    load_shared('verify').main(variants=VARIANTS)
