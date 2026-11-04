"""Inspect learned subpixel reads on validation images, without using GT.

This reports internal activation diagnostics, not segmentation accuracy or a
causal estimate of mechanism benefit. --synthetic is an untrained plumbing test.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('--checkpoint')
    parser.add_argument('--num-samples', type=int, default=4)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--no-yapf-cache', action='store_true')
    args = parser.parse_args()
    if args.num_samples < 1:
        parser.error('num-samples must be positive')
    if not args.synthetic and not args.checkpoint:
        parser.error('real validation diagnostics require --checkpoint')
    if args.no_yapf_cache:
        from yapf_third_party._ylib2to3.pgen2 import grammar
        grammar.Grammar.dump = lambda *a, **kw: None
    import torch
    from mmengine.config import Config
    from mmseg.registry import DATASETS, MODELS
    from mmseg.utils import register_all_modules
    from mmseg.models.decode_heads.efficientvit_subpixel_v2 import SubpixelMemoryV2
    from mmseg.models.decode_heads.efficientvit_structure_modules import resize_like

    register_all_modules(init_default_scope=True)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    cfg = Config.fromfile(args.config)
    model = MODELS.build(cfg.model).to(args.device).eval()
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    head = model.decode_head
    if not hasattr(head, 'structure_module'):
        parser.error('select a subpixel structure-head configuration')
    module = head.structure_module
    if not hasattr(module, 'forward_components'):
        if head.structure_mode != 'subpixel':
            parser.error('this script supports subpixel modes only')
        # Same parameters and function as the original subpixel, with explicit
        # component outputs. Numerical equivalence is covered by regression tests.
        adapter = SubpixelMemoryV2(module.query.in_channels).to(
            args.device).eval()
        adapter.load_state_dict(module.state_dict(), strict=True)
        module = adapter
    dataset = None if args.synthetic else DATASETS.build(
        cfg.val_dataloader.dataset)

    def rms(tensor):
        return float(tensor.float().square().mean().sqrt())

    records = []
    with torch.no_grad():
        count = args.num_samples if args.synthetic else min(
            args.num_samples, len(dataset))
        for index in range(count):
            if args.synthetic:
                inputs = torch.randn(1, 3, 64, 64, device=args.device)
                image_path = None
            else:
                item = dataset[index]
                image_path = item['data_samples'].metainfo.get('img_path')
                processed = model.data_preprocessor(
                    dict(
                        inputs=[item['inputs']],
                        data_samples=[item['data_samples']]),
                    training=False)
                inputs = processed['inputs']
            features = model.backbone(inputs)
            coarse_feature = head.forward_features(features)
            coarse = head.coarse_projection(coarse_feature)
            detail = head.structure_projection(
                features[head.structure_in_index])
            output, components = module.forward_components(coarse, detail)
            record = dict(
                index=index,
                image=image_path,
                residual_rms=rms(components['residual']))
            for name in ('semantic_read', 'geometry_read', 'contrast',
                         'context'):
                if name in components:
                    record[name + '_rms'] = rms(components[name])
            if 'semantic_read' in components:
                sem = components['semantic_read']
                record['semantic_phase_difference_ratio'] = (
                    rms(sem - sem.mean(1, keepdim=True)) / max(rms(sem), 1e-8))
                record['contrast_mean_abs_max'] = float(
                    components['contrast'].mean(1).abs().max())
                if getattr(module, 'center_reads', False):
                    record['shared_mean_error_abs_max'] = float(
                        (components['phases'].mean(1) -
                         components['shared_read']).abs().max())
            correction = head.structure_scale.tanh(
            ) * head.structure_classifier(output.to(detail.dtype))
            main_logits = resize_like(head.cls_seg(coarse_feature), detail)
            record['correction_to_main_rms_ratio'] = rms(correction) / max(
                rms(main_logits), 1e-8)
            records.append(record)
    numeric = sorted({
        k
        for r in records
        for k, v in r.items() if isinstance(v, (float, int)) and k != 'index'
    })
    means = {
        k: statistics.mean(r[k] for r in records if k in r)
        for k in numeric
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            dict(
                mode=head.structure_mode,
                checkpoint=args.checkpoint,
                synthetic=args.synthetic,
                config=str(args.config),
                samples=records,
                mean=means),
            indent=2),
        encoding='utf-8')
    print(
        json.dumps(
            dict(
                mode=head.structure_mode, synthetic=args.synthetic,
                mean=means)))


if __name__ == '__main__':
    main()
