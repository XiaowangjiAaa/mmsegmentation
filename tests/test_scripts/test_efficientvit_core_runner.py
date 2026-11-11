"""Independent core-mechanism protocols and unchanged pretrained parameter keys."""
import importlib.util
from pathlib import Path
import unittest

import torch
from mmengine.config import Config
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    'core_runner', ROOT / 'scripts/efficientvit_core_20261112/batch_train.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class TestCoreRunner(unittest.TestCase):

    def test_same_seed_preserves_original_full_model_initialization(self):
        register_all_modules(init_default_scope=True)
        torch.manual_seed(101)
        original = MODELS.build(
            Config.fromfile(ROOT / runner.VARIANTS['baseline']).model)
        rng_after = torch.get_rng_state().clone()
        for name in ['bank_structure', 'bank_content', 'joint', 'postfusion']:
            torch.manual_seed(101)
            model = MODELS.build(
                Config.fromfile(ROOT / runner.VARIANTS[name]).model)
            self.assertTrue(torch.equal(torch.get_rng_state(), rng_after))
            for key, value in original.state_dict().items():
                torch.testing.assert_close(
                    model.state_dict()[key], value, rtol=0, atol=0)

    def test_scratch_default_and_explicit_pretrained(self):
        shared = runner.load_shared('batch_train')
        args = shared.parse_args([], **runner.DEFAULTS)
        jobs = shared.make_jobs(args, runner.VARIANTS)
        self.assertTrue(args.scratch)
        self.assertEqual(len(jobs), 21)
        self.assertIn('load_from=None', shared.build_command(args, jobs[0]))
        self.assertFalse(
            shared.parse_args(['--pretrained'], **runner.DEFAULTS).scratch)

    def test_original_downsampling_head_and_training_protocol_are_fixed(self):
        configs = {
            n: Config.fromfile(ROOT / p)
            for n, p in runner.VARIANTS.items()
        }
        base = configs['baseline']
        for name, cfg in configs.items():
            self.assertEqual(cfg.model.decode_head, base.model.decode_head)
            self.assertEqual(
                cfg.model.backbone.get('downsample_mode', 'v1'), 'v1')
            self.assertEqual(
                cfg.model.backbone.get('transition_mode', 'v1'), 'v1')
            for key in [
                    'load_from', 'train_dataloader', 'val_dataloader',
                    'randomness', 'train_cfg', 'optim_wrapper',
                    'param_scheduler'
            ]:
                self.assertEqual(cfg[key], base[key], (name, key))

    def test_paired_references_for_both_core_mechanisms(self):
        shared = runner.load_shared('batch_train')
        records = [
            dict(name=n, seed=42, status='ok', metrics={'IoU/crack': v})
            for n, v in [('baseline',
                          60), ('bank_content',
                                61), ('bank_uniform',
                                      60), ('joint_fixed',
                                            59), ('postfusion',
                                                  61), ('bank_structure',
                                                        62), ('joint', 63)]
        ]
        summary = shared.aggregate(records)
        self.assertEqual(
            summary['bank_structure']['paired_deltas_IoU/crack']
            ['bank_content']['mean'], 1)
        self.assertEqual(
            summary['joint']['paired_deltas_IoU/crack']['postfusion']['mean'],
            2)


if __name__ == '__main__':
    unittest.main()
