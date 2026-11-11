"""Exercise genuine segmentation loss/backward for all core attention modes."""
from batch_train import VARIANTS, load_shared

if __name__ == '__main__':
    load_shared('verify').main(variants=VARIANTS)
