"""Keep encoder-only controls and optional decoder combinations distinct."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    'encoder_runner',
    ROOT / 'scripts/efficientvit_encoder_20261106/batch_train.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class TestEncoderRunner(unittest.TestCase):

    def test_default_runs_only_independent_encoder_versions(self):
        shared = runner.load_shared('batch_train')
        args = shared.parse_args([], 'baseline,downsample,query,context_local')
        jobs = shared.make_jobs(args, runner.VARIANTS)
        self.assertEqual(len(jobs), 12)
        self.assertEqual({j['name']
                          for j in jobs},
                         {'baseline', 'downsample', 'query', 'context_local'})

    def test_all_variants_and_scratch_command(self):
        shared = runner.load_shared('batch_train')
        args = shared.parse_args(['--variants', 'all', '--scratch'])
        jobs = shared.make_jobs(args, runner.VARIANTS)
        self.assertEqual(len(jobs), 24)
        self.assertTrue(all('scratch' in j['work_dir'] for j in jobs))
        self.assertIn('load_from=None', shared.build_command(args, jobs[0]))

    def test_configs_preserve_head_and_training_protocol(self):
        from mmengine.config import Config
        configs = {
            n: Config.fromfile(ROOT / p)
            for n, p in runner.VARIANTS.items()
        }
        base = configs['baseline']
        for name, cfg in configs.items():
            for key in [
                    'load_from', 'train_dataloader', 'val_dataloader',
                    'optim_wrapper', 'param_scheduler', 'train_cfg'
            ]:
                self.assertEqual(cfg[key], base[key], (name, key))
            if name in [
                    'downsample', 'query', 'context_local', 'downsample_query'
            ]:
                self.assertEqual(cfg.model.decode_head, base.model.decode_head)
            elif name.endswith('subpixel'):
                self.assertEqual(cfg.model.decode_head,
                                 configs['subpixel'].model.decode_head)


if __name__ == '__main__':
    unittest.main()
