"""Initialization protocol, paired references and attention relocation."""
import importlib.util
from pathlib import Path
import unittest

import torch
from mmengine.config import Config
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    path = ROOT / 'scripts/efficientvit_transition_20261108' / f'{name}.py'
    spec = importlib.util.spec_from_file_location('transition_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load('batch_train')


class TestTransitionRunner(unittest.TestCase):

    def test_default_scratch_and_explicit_pretrained_are_separate(self):
        shared = runner.load_shared('batch_train')
        args = shared.parse_args([], **runner.DEFAULTS)
        self.assertTrue(args.scratch)
        jobs = shared.make_jobs(args, runner.VARIANTS)
        self.assertEqual(len(jobs), 15)
        self.assertIn('load_from=None', shared.build_command(args, jobs[0]))
        self.assertTrue(all('scratch' in j['work_dir'] for j in jobs))
        args = shared.parse_args(['--pretrained', '--variants', 'all'],
                                 **runner.DEFAULTS)
        self.assertFalse(args.scratch)
        self.assertEqual(len(shared.make_jobs(args, runner.VARIANTS)), 21)
        self.assertFalse(shared.parse_args(
            []).scratch)  # Older runners unchanged.

    def test_fixed_head_and_common_training_protocol(self):
        configs = {
            n: Config.fromfile(ROOT / p)
            for n, p in runner.VARIANTS.items()
        }
        base = configs['baseline']
        for name, cfg in configs.items():
            for key in [
                    'train_dataloader', 'val_dataloader', 'optim_wrapper',
                    'param_scheduler', 'train_cfg', 'randomness'
            ]:
                self.assertEqual(cfg[key], base[key], (name, key))
            reference = configs['subpixel'] if name.endswith(
                'subpixel') else base
            self.assertEqual(cfg.model.decode_head,
                             reference.model.decode_head)

    def test_attention_is_moved_exactly_and_remaining_weights_load(self):
        register_all_modules(init_default_scope=True)
        original = MODELS.build(
            Config.fromfile(ROOT / runner.VARIANTS['baseline']).model)
        state, audit = load('prepare_pretrained').remap_state_dict(
            original.state_dict())
        for old, new in audit['moved']:
            torch.testing.assert_close(original.state_dict()[old], state[new])
        self.assertTrue(audit['dropped'])
        for mode in ['semantic', 'local', 'fixed', 'pooled']:
            model = MODELS.build(
                Config.fromfile(ROOT / runner.VARIANTS[mode]).model)
            missing, unexpected = model.load_state_dict(state, strict=False)
            self.assertFalse(unexpected)
            self.assertTrue(
                all(
                    k.startswith('backbone.stages.2.op_list.0.')
                    for k in missing))
        with self.assertRaises(ValueError):
            load('prepare_pretrained').remap_state_dict(state)

    def test_summary_pairs_semantic_against_controls(self):
        shared = runner.load_shared('batch_train')
        records = [
            dict(name=n, seed=42, status='ok', metrics={'IoU/crack': v})
            for n, v in [('baseline', 60), ('fixed',
                                            61), ('local',
                                                  62), ('pooled',
                                                        59), ('semantic', 63)]
        ]
        paired = shared.aggregate(
            records)['semantic']['paired_deltas_IoU/crack']
        self.assertEqual({
            k: v['mean']
            for k, v in paired.items()
        }, dict(baseline=3, fixed=2, local=1, pooled=4))


if __name__ == '__main__':
    unittest.main()
