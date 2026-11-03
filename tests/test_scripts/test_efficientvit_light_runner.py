"""Queue and metric-selection checks; no GPU or dataset required."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    'light_runner', ROOT / 'scripts/efficientvit_light_20261102/batch_train.py')
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class TestLightRunner(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        (ROOT / 'work_dirs').mkdir(exist_ok=True)

    def test_paired_jobs_and_protocol_isolation(self):
        args = RUNNER.parse_args(['--scratch', '--variants', 'baseline,cra_fixed',
                                  '--seeds', '42,43'])
        jobs = RUNNER.make_jobs(args)
        self.assertEqual(len(jobs), 4)
        self.assertEqual(len({j['work_dir'] for j in jobs}), 4)
        self.assertIn('scratch', jobs[0]['work_dir'])
        cmd = RUNNER.build_command(args, jobs[0])
        self.assertIn('load_from=None', cmd)
        self.assertIn('randomness.seed=42', cmd)
        self.assertIn('param_scheduler.1.end=40000', cmd)

    def test_custom_registry_keeps_existing_runner_unchanged(self):
        registry = dict(baseline='custom/baseline.py', partition='custom/partition.py')
        args = RUNNER.parse_args(['--variants', 'all', '--seeds', '42'])
        jobs = RUNNER.make_jobs(args, registry)
        self.assertEqual([j['name'] for j in jobs], list(registry))
        self.assertEqual(jobs[1]['config'], 'custom/partition.py')
        self.assertIn('magnitude', RUNNER.VARIANTS)
        self.assertNotIn('partition', RUNNER.VARIANTS)

    def test_reject_unfair_cra_and_duplicates(self):
        for extra in (['--variants', 'cra_fixed'],
                      ['--seeds', '42,42'],
                      ['--variants', 'baseline,baseline']):
            with self.assertRaises(ValueError):
                RUNNER.make_jobs(RUNNER.parse_args(extra))

    def test_best_miou_selects_one_consistent_validation_point(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'work_dirs') as temp:
            file = Path(temp) / '20261102' / 'vis_data' / 'scalars.json'
            file.parent.mkdir(parents=True)
            rows = [dict(step=1000, mIoU=80, **{'IoU/crack': 62, 'F1/crack': 76}),
                    dict(step=2000, mIoU=79, **{'IoU/crack': 63, 'F1/crack': 75})]
            file.write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf-8')
            metrics = RUNNER.read_metrics(temp)
            self.assertEqual(metrics['best_step'], 1000)
            self.assertEqual(metrics['IoU/crack'], 62)
            self.assertEqual(metrics['last_val_step'], 2000)

    def test_paired_deltas_ignore_failed_and_unpaired_runs(self):
        records = [dict(name=name, seed=seed, status=status,
                        metrics={'IoU/crack': value}) for name, seed, status, value in (
                            ('baseline', 42, 'ok', 60),
                            ('baseline', 43, 'ok', 62),
                            ('magnitude', 42, 'ok', 61),
                            ('magnitude', 43, 'ok', 64),
                            ('magnitude', 44, 'failed', 90))]
        summary = RUNNER.aggregate(records)['magnitude']
        self.assertEqual(summary['paired_delta_IoU/crack']['mean'], 1.5)
        self.assertEqual(summary['paired_delta_IoU/crack']['n'], 2)

    def test_failed_child_sets_exit_code_and_preserves_queue(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'work_dirs') as temp:
            fake_torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 1))
            fake_proc = SimpleNamespace(poll=lambda: 2, returncode=2)
            argv = ['--scratch', '--gpus', '0', '--seeds', '42',
                    '--variants', 'baseline,magnitude', '--work-root', temp,
                    '--stop-on-failure']
            with patch.dict('sys.modules', {'torch': fake_torch}), \
                    patch.object(RUNNER.subprocess, 'Popen', return_value=fake_proc):
                self.assertEqual(RUNNER.main(argv), 1)
            summary = json.loads((Path(temp) / 'scratch/fp32/batch_summary.json').read_text())
            self.assertEqual([r['status'] for r in summary['runs']],
                             ['failed', 'not_started'])


if __name__ == '__main__':
    unittest.main()
