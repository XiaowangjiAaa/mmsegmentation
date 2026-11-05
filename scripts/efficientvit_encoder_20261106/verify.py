"""Build every encoder/head combination and check segmentation loss/backward."""
from batch_train import VARIANTS, load_shared

if __name__ == '__main__':
    load_shared('verify').main(variants=VARIANTS)
