"""Semantic structure transition: scratch-first, matched-seed experiments."""
import importlib.util
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
VARIANTS = {
    name: f'configs/efficientvit_transition_20261108/{name}.py'
    for name in ('baseline', 'semantic', 'fixed', 'local', 'pooled',
                 'subpixel', 'semantic_subpixel')
}
DEFAULTS = dict(
    default_variants='baseline,semantic,fixed,local,pooled',
    default_work_root='work_dirs/efficientvit_transition_20261108',
    default_scratch=True)


def load_shared(name):
    path = REPO_ROOT / 'scripts/efficientvit_light_20261102' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(
        f'efficientvit_shared_{name}', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    runner = load_shared('batch_train')
    args = runner.parse_args(argv, **DEFAULTS)
    if not args.scratch and not (args.dry_run or args.list):
        subprocess.run([
            sys.executable,
            str(Path(__file__).with_name('prepare_pretrained.py'))
        ],
                       cwd=REPO_ROOT,
                       check=True)
    return runner.main(argv, variants=VARIANTS, **DEFAULTS)


if __name__ == '__main__':
    sys.exit(main())
