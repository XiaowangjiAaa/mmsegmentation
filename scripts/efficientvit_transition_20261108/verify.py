"""Exercise every transition control with genuine segmentation loss/backward."""
from batch_train import VARIANTS, load_shared

if __name__ == '__main__':
    load_shared('verify').main(variants=VARIANTS)
