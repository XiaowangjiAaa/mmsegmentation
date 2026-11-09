"""Fixed attention/head, scratch protocol and conservative weight migration."""
import importlib.util
from pathlib import Path
import unittest

import torch
from mmengine.config import Config
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    path = ROOT / 'scripts/efficientvit_adaptive_20261110' / f'{name}.py'
    spec = importlib.util.spec_from_file_location('adaptive_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load('batch_train')


class TestAdaptiveRunner(unittest.TestCase):

    def test_default_protocol_and_explicit_pretrained(self):
        shared = runner.load_shared('batch_train')
        args = shared.parse_args([], **runner.DEFAULTS)
        jobs = shared.make_jobs(args, runner.VARIANTS)
        self.assertTrue(args.scratch)
        self.assertEqual(len(jobs), 15)
        self.assertIn('load_from=None', shared.build_command(args, jobs[0]))
        self.assertFalse(
            shared.parse_args(['--pretrained'], **runner.DEFAULTS).scratch)

    def test_head_and_protocol_are_identical(self):
        configs = {
            n: Config.fromfile(ROOT / p)
            for n, p in runner.VARIANTS.items()
        }
        base = configs['baseline']
        for name, cfg in configs.items():
            self.assertEqual(cfg.model.decode_head, base.model.decode_head)
            self.assertEqual(cfg.model.backbone.get('block_type', 'v1'), 'v1')
            self.assertEqual(
                cfg.model.backbone.get('transition_mode', 'v1'), 'v1')
            for key in [
                    'train_dataloader', 'val_dataloader', 'randomness',
                    'train_cfg', 'optim_wrapper', 'param_scheduler'
            ]:
                self.assertEqual(cfg[key], base[key], (name, key))

    def test_migration_drops_only_old_spatial_kernel(self):
        register_all_modules(init_default_scope=True)
        original = MODELS.build(
            Config.fromfile(ROOT / runner.VARIANTS['baseline']).model)
        source = original.state_dict()
        converter = load('prepare_pretrained')
        state, audit = converter.remap_state_dict(source)
        self.assertEqual(
            set(audit['dropped']),
            {converter.REPLACED + 'weight', converter.REPLACED + 'bias'})
        self.assertEqual(len(source) - len(state), 2)
        for key, value in state.items():
            torch.testing.assert_close(value, source[key])
        for name in ['structure', 'content', 'static', 'uniform']:
            model = MODELS.build(
                Config.fromfile(ROOT / runner.VARIANTS[name]).model)
            missing, unexpected = model.load_state_dict(state, strict=False)
            self.assertFalse(unexpected)
            self.assertTrue(
                all(k.startswith(converter.REPLACED) for k in missing))

    def test_summary_contains_mechanism_references(self):
        shared = runner.load_shared('batch_train')
        records = [
            dict(name=n, seed=42, status='ok', metrics={'IoU/crack': v})
            for n, v in [('baseline', 60), ('content',
                                            61), ('static',
                                                  59), ('uniform',
                                                        58), ('structure', 62)]
        ]
        pairs = shared.aggregate(
            records)['structure']['paired_deltas_IoU/crack']
        self.assertEqual({
            k: v['mean']
            for k, v in pairs.items()
        }, dict(baseline=2, content=1, static=3, uniform=4))


if __name__ == '__main__':
    unittest.main()
