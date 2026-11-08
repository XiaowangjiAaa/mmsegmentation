"""Probe transition routing on real images; GT never enters the feature path."""
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
    parser.add_argument(
        '--synthetic',
        action='store_true',
        help='untrained random-input plumbing test')
    parser.add_argument('--no-yapf-cache', action='store_true')
    args = parser.parse_args()
    if args.num_samples < 1 or (not args.synthetic and not args.checkpoint):
        parser.error(
            'use positive num-samples; real diagnostics require a trained checkpoint'
        )
    if args.no_yapf_cache:
        from yapf_third_party._ylib2to3.pgen2 import grammar
        grammar.Grammar.dump = lambda *a, **kw: None
    import torch
    from mmengine.config import Config
    from mmseg.registry import DATASETS, MODELS
    from mmseg.utils import register_all_modules
    from mmseg.models.backbones.efficientvit.semantic_transition import (
        SemanticStructureTransition, phase_components)
    register_all_modules(init_default_scope=True)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    cfg = Config.fromfile(args.config)
    model = MODELS.build(cfg.model).to(args.device).eval()
    if args.checkpoint:
        model.load_state_dict(
            torch.load(args.checkpoint, map_location='cpu')['state_dict'],
            strict=True)
    backbone = model.backbone
    transition = backbone.stages[2].op_list[0]
    if not isinstance(transition, SemanticStructureTransition):
        parser.error(
            'select one of the semantic/local/fixed/pooled transition configurations'
        )
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
            output, parts = transition.forward_components(x)
            record = dict(
                index=index,
                image=image,
                output_rms=rms(output),
                context_rms=rms(parts['context']))
            if 'weights' in parts:
                weights, structure = parts['weights'], parts['structure']
                record['weights_mean'] = float(weights.mean())
                record['weights_std'] = float(
                    weights.float().std(unbiased=False))
                record['structure_rms'] = rms(structure)
                record['weighted_structure_rms'] = rms(weights * structure)
                _, *differences = phase_components(x)
                _, control = transition.aggregate(
                    parts['mean'], differences,
                    torch.zeros_like(parts['context']))
                record['context_condition_change_rms'] = rms(
                    weights - control['weights'])
                zero = torch.zeros_like(structure).flatten(1, 2)
                no_structure = transition.fusion(
                    torch.cat([parts['context'], zero], 1))
                record['structure_effect_to_output_rms_ratio'] = rms(
                    output - no_structure) / max(rms(output), 1e-8)
                for i, name in enumerate(
                    ['horizontal', 'vertical', 'diagonal']):
                    record[name + '_weight_mean'] = float(weights[:, i].mean())
            records.append(record)
    means = {
        key: statistics.mean(r[key] for r in records)
        for key, value in records[0].items()
        if isinstance(value, (int, float)) and key != 'index'
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            dict(
                gate_mode=transition.gate_mode,
                checkpoint=args.checkpoint,
                synthetic=args.synthetic,
                samples=records,
                mean=means,
                note='Activation diagnostics, not accuracy or causal benefit.'
            ),
            indent=2),
        encoding='utf-8')
    print(
        json.dumps(
            dict(
                gate_mode=transition.gate_mode,
                synthetic=args.synthetic,
                mean=means)))


if __name__ == '__main__':
    main()
