"""Train independent structure mechanisms using the shared paired-seed queue."""
import importlib.util
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
VARIANTS = {
    name: f'configs/efficientvit_structure_20261104/{name}.py'
    for name in ('baseline', 'detail_resize', 'detail', 'partition',
                 'connectivity', 'subpixel')
}


def load_shared(name):
    path = REPO_ROOT / 'scripts/efficientvit_light_20261102' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(
        f'efficientvit_shared_{name}', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    return load_shared('batch_train').main(
        argv,
        variants=VARIANTS,
        default_variants=','.join(VARIANTS),
        default_work_root='work_dirs/efficientvit_structure_20261104')


if __name__ == '__main__':
    sys.exit(main())
