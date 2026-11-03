"""Build all configs and exercise real MMSeg losses and gradients."""
import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def main(argv=None, variants=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--no-yapf-cache', action='store_true',
                        help='process-only workaround for a blocked Windows YAPF cache')
    args = parser.parse_args(argv)
    if args.size < 32:
        parser.error('size must be at least 32')
    if args.no_yapf_cache:
        from yapf_third_party._ylib2to3.pgen2 import grammar
        grammar.Grammar.dump = lambda *a, **kw: None
    import torch
    from mmengine.config import Config
    from mmengine.structures import PixelData
    from mmseg.registry import MODELS
    from mmseg.structures import SegDataSample
    from mmseg.utils import register_all_modules
    if variants is None:
        from batch_train import VARIANTS
        variants = VARIANTS

    register_all_modules(init_default_scope=True)
    torch.set_num_threads(2)
    records = []
    for name, path in variants.items():
        torch.manual_seed(42)
        cfg = Config.fromfile(REPO_ROOT / path)
        model = MODELS.build(cfg.model).train()
        inputs = torch.randn(2, 3, args.size, args.size)
        samples = []
        for _ in range(2):
            sample = SegDataSample()
            sample.gt_sem_seg = PixelData(data=torch.randint(0, 2, (1, args.size, args.size)))
            samples.append(sample)
        losses = model.loss(inputs, samples)
        loss = sum(value for key, value in losses.items() if 'loss' in key)
        assert torch.isfinite(loss), name
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads), name
        with torch.no_grad():
            logits = model.decode_head(model.backbone(inputs))
        head_cfg = cfg.model.decode_head
        stride = 4 if (head_cfg.get('detail_channels', 0) or
                       head_cfg.get('detail_upsample_only', False) or
                       head_cfg.type == 'EfficientViTStructureHead') else 8
        assert logits.shape == (2, 2, args.size // stride, args.size // stride)
        record = dict(name=name, parameters=sum(p.numel() for p in model.parameters()),
                      logits=list(logits.shape), loss=float(loss.detach()),
                      finite_gradients=True)
        records.append(record)
        print(json.dumps(record), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(records, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
