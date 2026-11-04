"""Check real segmentation loss and backward for all subpixel configurations."""
from batch_train import VARIANTS, load_shared

if __name__ == '__main__':
    load_shared('verify').main(variants=VARIANTS)
