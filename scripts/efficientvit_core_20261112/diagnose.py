"""Inspect learned memory routes and local/global mass competition on images."""
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
    from mmseg.models.backbones.efficientvit.attn_plugins.core_attention import CoreLiteMLA
    register_all_modules(init_default_scope=True)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    cfg = Config.fromfile(args.config)
    model = MODELS.build(cfg.model).to(args.device).eval()
    if args.checkpoint:
        model.load_state_dict(
            torch.load(args.checkpoint, map_location='cpu')['state_dict'],
            strict=True)
    records, handles = [], []
    current_index = 0

    def make_hook(name):

        def record(module, inputs, output):
            _, details = module.forward_components(inputs[0])
            row = dict(
                index=current_index,
                module=name,
                mode=module.mode,
                output_rms=float(output.float().square().mean().sqrt()))
            if 'routes' in details:
                routes = details['routes']
                row['route_entropy_mean'] = float(
                    -(routes * routes.clamp_min(1e-12).log()).sum(-1).mean())
                row['route_spatial_std_mean'] = float(
                    routes.std(1, unbiased=False).mean())
                for bank in range(module.num_banks):
                    row[f'bank{bank}_occupancy_mean'] = float(
                        routes[..., bank].mean())
                if module.num_banks == 2:
                    memory = details['memory']
                    row['bank_memory_cosine_mean'] = float(
                        F.cosine_similarity(
                            memory[:, :, 0].flatten(2),
                            memory[:, :, 1].flatten(2),
                            dim=-1).mean())
            else:
                weight, fraction = details['global_weight'], details[
                    'effective_global_fraction']
                row['global_weight_mean'] = float(weight.mean())
                row['effective_global_fraction_mean'] = float(fraction.mean())
                row['effective_global_fraction_std'] = float(
                    fraction.std(unbiased=False))
                row['empty_global_fraction'] = float(
                    (details['global_count'] == 0).float().mean())
            records.append(row)

        return record

    for name, module in model.backbone.named_modules():
        if isinstance(module, CoreLiteMLA):
            handles.append(module.register_forward_hook(make_hook(name)))
    if not handles:
        parser.error('select a core attention configuration')
    dataset = None if args.synthetic else DATASETS.build(
        cfg.val_dataloader.dataset)
    with torch.no_grad():
        for index in range(
                args.num_samples if args.
                synthetic else min(args.num_samples, len(dataset))):
            current_index = index
            if args.synthetic:
                inputs = torch.randn(1, 3, 64, 64, device=args.device)
            else:
                item = dataset[index]
                inputs = model.data_preprocessor(
                    dict(
                        inputs=[item['inputs']],
                        data_samples=[item['data_samples']]),
                    training=False)['inputs']
            model.backbone(inputs)
    for handle in handles:
        handle.remove()
    means = {}
    for name in dict.fromkeys(r['module'] for r in records):
        group = [r for r in records if r['module'] == name]
        means[name] = {
            k: statistics.mean(r[k] for r in group)
            for k, v in group[0].items()
            if isinstance(v, (int, float)) and k != 'index'
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            dict(
                checkpoint=args.checkpoint,
                synthetic=args.synthetic,
                samples=records,
                per_module=means,
                note=
                'Internal activation diagnostics, not accuracy or causal mechanism evidence.'
            ),
            indent=2),
        encoding='utf-8')
    print(json.dumps(dict(synthetic=args.synthetic, per_module=means)))


if __name__ == '__main__':
    main()
