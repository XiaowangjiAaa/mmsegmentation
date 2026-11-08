"""Move the consumed first LiteMLA; discard the replaced strided MBConv.

This initializes compatible weights only, not an exact function conversion.
The old checkpoint must not be used to resume training the new architecture.
"""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def remap_state_dict(state):
    old_down = 'backbone.stages.2.op_list.0.'
    old_att = 'backbone.stages.2.op_list.1.context_module.'
    new_att = 'backbone.stages.2.op_list.0.context_module.'
    remapped, moved, dropped = {}, [], []
    for key, value in state.items():
        if key.startswith(old_down):
            dropped.append(key)
        elif key.startswith(old_att):
            target = new_att + key[len(old_att):]
            remapped[target] = value
            moved.append((key, target))
        else:
            remapped[key] = value
    if not moved or not dropped:
        raise ValueError(
            'Expected an original B0 segmentation checkpoint with its strided MBConv and first LiteMLA'
        )
    return remapped, dict(moved=moved, dropped=dropped)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--source',
        type=Path,
        default=ROOT / 'work_dirs/pretrained/efficientvit_b0_crack.pth')
    parser.add_argument(
        '--output',
        type=Path,
        default=ROOT /
        'work_dirs/pretrained/efficientvit_b0_semantic_transition.pth')
    parser.add_argument('--no-yapf-cache', action='store_true')
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error('output must differ from source')
    if not args.source.is_file():
        if args.source == ROOT / 'work_dirs/pretrained/efficientvit_b0_crack.pth':
            subprocess.run([
                sys.executable,
                str(ROOT / 'scripts/prepare_efficientvit_b0.py')
            ],
                           cwd=ROOT,
                           check=True)
        else:
            raise FileNotFoundError(args.source)
    if args.no_yapf_cache:
        from yapf_third_party._ylib2to3.pgen2 import grammar
        grammar.Grammar.dump = lambda *a, **kw: None
    sys.path.insert(0, str(ROOT))
    import torch
    from mmengine.config import Config
    from mmseg.registry import MODELS
    from mmseg.utils import register_all_modules
    register_all_modules(init_default_scope=True)
    source = torch.load(args.source, map_location='cpu')
    state, audit = remap_state_dict(source['state_dict'])
    model = MODELS.build(
        Config.fromfile(
            ROOT /
            'configs/efficientvit_transition_20261108/semantic.py').model)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise ValueError(f'Unexpected checkpoint keys: {unexpected}')
    allowed = ('backbone.stages.2.op_list.0.mean_projection.',
               'backbone.stages.2.op_list.0.structure_projection.',
               'backbone.stages.2.op_list.0.context_projection.',
               'backbone.stages.2.op_list.0.router.',
               'backbone.stages.2.op_list.0.fusion.', 'decode_head.conv_seg.')
    if not all(key.startswith(allowed) for key in missing):
        raise ValueError(f'Unexpected missing keys: {missing}')
    metadata = dict(
        source=str(args.source),
        source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        conversion='semantic_transition_v1',
        missing_keys=missing,
        **audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp')
    torch.save(dict(state_dict=state, meta=metadata), temporary)
    temporary.replace(args.output)
    print(
        f'Prepared {args.output}: moved {len(audit["moved"])} keys, dropped {len(audit["dropped"])} keys; new layers require training.'
    )


if __name__ == '__main__':
    main()
