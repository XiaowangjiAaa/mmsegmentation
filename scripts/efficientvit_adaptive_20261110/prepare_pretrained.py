"""Drop only the replaced spatial kernel; preserve all original attention keys."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
REPLACED = 'backbone.stages.2.op_list.0.main.depth_conv.conv.'


def remap_state_dict(state):
    dropped = [key for key in state if key.startswith(REPLACED)]
    if set(dropped) != {REPLACED + 'weight', REPLACED + 'bias'}:
        raise ValueError(
            'Expected original B0 strided depthwise weight and bias')
    return {
        key: value
        for key, value in state.items() if key not in dropped
    }, dict(dropped=dropped)


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
        'work_dirs/pretrained/efficientvit_b0_adaptive_downsample.pth')
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
            'configs/efficientvit_adaptive_20261110/structure.py').model)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or not all(
            k.startswith((REPLACED, 'decode_head.conv_seg.'))
            for k in missing):
        raise ValueError(
            f'Checkpoint incompatibility: missing={missing}, unexpected={unexpected}'
        )
    metadata = dict(
        source=str(args.source),
        source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        conversion='adaptive_downsample_v1',
        missing_keys=missing,
        **audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp')
    torch.save(dict(state_dict=state, meta=metadata), temporary)
    temporary.replace(args.output)
    print(
        f'Prepared {args.output}: dropped {len(audit["dropped"])} spatial-kernel keys; attention keys unchanged.'
    )


if __name__ == '__main__':
    main()
