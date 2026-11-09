"""Inspect learned local routing; activation statistics are not accuracy gains."""
import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('--checkpoint')
    parser.add_argument('--num-samples', type=int, default=8)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--no-yapf-cache', action='store_true')
    args = parser.parse_args()
    if args.num_samples < 1 or (not args.synthetic and not args.checkpoint):
        parser.error(
            'real diagnostics require checkpoint and positive num-samples')
    if args.no_yapf_cache:
        from yapf_third_party._ylib2to3.pgen2 import grammar
        grammar.Grammar.dump = lambda *a, **kw: None
    import torch
    from torch.nn import functional as F
    from mmengine.config import Config
    from mmseg.registry import DATASETS, MODELS
    from mmseg.utils import register_all_modules
    from mmseg.models.backbones.efficientvit.adaptive_downsample import LocalStructureAggregation
    register_all_modules(init_default_scope=True)
    torch.manual_seed(42)
    torch.set_num_threads(2)
    cfg = Config.fromfile(args.config)
    model = MODELS.build(cfg.model).to(args.device).eval()
    if args.checkpoint:
        model.load_state_dict(
            torch.load(args.checkpoint, map_location='cpu')['state_dict'],
            strict=True)
    backbone = model.backbone
    block = backbone.stages[2].op_list[0].main
    operator = block.depth_conv.conv
    if not isinstance(operator, LocalStructureAggregation):
        parser.error('select structure/content/static/uniform configuration')
    dataset = None if args.synthetic else DATASETS.build(
        cfg.val_dataloader.dataset)
    records = []

    def rms(x):
        return float(x.float().square().mean().sqrt())

    with torch.no_grad():
        for index in range(
                args.num_samples if args.
                synthetic else min(args.num_samples, len(dataset))):
            image = None
            if args.synthetic:
                inputs = torch.randn(1, 3, 64, 64, device=args.device)
            else:
                item = dataset[index]
                image = item['data_samples'].metainfo.get('img_path')
                inputs = model.data_preprocessor(
                    dict(
                        inputs=[item['inputs']],
                        data_samples=[item['data_samples']]),
                    training=False)['inputs']
            x = backbone.stages[1](
                backbone.stages[0](backbone.input_stem(inputs)))
            expanded = block.inverted_conv(x)
            output, parts = operator.forward_components(expanded)
            weights, valid = parts['weights'], parts['valid']
            uniform = F.avg_pool2d(expanded, 3, 2, 1, count_include_pad=False)
            entropy = -(weights * weights.clamp_min(1e-12).log()).sum(2)
            interior = valid.all(2)[0, 0]
            record = dict(
                index=index,
                image=image,
                aggregation_rms=rms(output),
                center_weight_mean=float(weights[:, :, 4].mean()),
                entropy_nats_mean=float(entropy.mean()),
                group_difference_rms=rms(weights -
                                         weights.mean(1, keepdim=True)),
                normalization_error_max=float(
                    (weights.sum(2) - 1).abs().max()),
                update_vs_uniform_rms_ratio=rms(output - uniform) /
                max(rms(output), 1e-8))
            if interior.any():
                record['interior_weight_std_mean'] = float(
                    weights[:, :, :, interior].std(-1, unbiased=False).mean())
            records.append(record)
    keys = {
        k
        for r in records
        for k, v in r.items() if isinstance(v, (int, float)) and k != 'index'
    }
    means = {
        k: statistics.mean(r[k] for r in records if k in r)
        for k in sorted(keys)
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            dict(
                mode=operator.mode,
                checkpoint=args.checkpoint,
                synthetic=args.synthetic,
                samples=records,
                mean=means,
                note=
                'Internal routing diagnostics only; not segmentation accuracy or causal contribution.'
            ),
            indent=2),
        encoding='utf-8')
    print(
        json.dumps(
            dict(mode=operator.mode, synthetic=args.synthetic, mean=means)))


if __name__ == '__main__':
    main()
