"""Source-equivalence and independent launcher checks; no external data/GPU."""
import contextlib
import copy
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import train_reproduce as entry
from src.PDETime.models.pdetime_model import PDETime, PDETimeLoss
from src.PDETime.preprocess.dataloader import PDETimeDataset


class TSLibEntryTests(unittest.TestCase):
    def args(self, *extra):
        return entry.parse_args(['--device', 'cpu', '--itr', '1', '--seq_len', '8', '--pred_len', '4',
                                 '--patch_length', '4', '--hidden_features', '8', '--latent_features', '8',
                                 '--temporal_latent_features', '4', '--mlp_hidden_features', '8',
                                 '--inr_layers', '2', '--no_plot', *extra])

    def test_original_train_configuration_forward_loss_and_adam_step_match(self):
        a = entry.parse_args(['--device', 'cpu', '--protocol', 'trainpy', '--itr', '1'])
        a.enc_in = 7
        torch.manual_seed(2024)
        original = PDETime(dimension=7, temporal_features=7, temporal_latent_features=32,
            lookback=480, horizon=96, s_cff=1, hidden_features=64, INR_layers=5,
            aggregation_layers=1, latent_features=64, patch_length=32,
            MLP_hidden_layers=3, MLP_hidden_features=64)
        torch.manual_seed(2024)
        adapted = entry.build_model(a, 7)
        self.assertIs(type(adapted), PDETime)
        x, t, tau = torch.randn(2, 480, 7), torch.randn(2, 576, 7), torch.rand(2, 576)
        labels = torch.cat([x, torch.randn(2, 96, 7)], 1)
        prediction1, prediction2 = original(x, t, tau), adapted(x, t, tau)
        torch.testing.assert_close(prediction1, prediction2, rtol=0, atol=0)
        c1, c2 = PDETimeLoss(nn.SmoothL1Loss(), 480, 96), PDETimeLoss(nn.SmoothL1Loss(beta=a.smooth_l1_beta), 480, 96)
        l1, l2 = c1(prediction1, labels), c2(prediction2, labels)
        torch.testing.assert_close(l1, l2, rtol=0, atol=0)
        o1 = torch.optim.Adam(original.parameters(), lr=1e-4)
        o2 = torch.optim.Adam(adapted.parameters(), lr=a.learning_rate)
        before = copy.deepcopy(adapted.encoder.agg_layers[0].state_dict())
        o1.zero_grad(); o2.zero_grad()
        l1.backward(); l2.backward()
        o1.step(); o2.step()
        for left, right in zip(original.parameters(), adapted.parameters()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        for name, value in adapted.encoder.agg_layers[0].state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        self.assertFalse(any('agg_layers' in key for key in adapted.state_dict()))

    def test_source_sharing_and_optimizer_coverage_are_preserved(self):
        a = self.args('--aggregation_layers', '3')
        a.enc_in = 7
        model = entry.build_model(a, 5)
        self.assertIs(model.encoder.agg_layers[0], model.encoder.agg_layers[2])
        self.assertIs(model.solver.mlp_dudt.net[2], model.solver.mlp_dudt.net[4])
        optimizer_ids = {id(p) for p in model.parameters()}
        self.assertTrue(all(id(p) not in optimizer_ids for p in model.encoder.agg_layers[0].parameters()))

    def test_checkpoint_restores_unregistered_weights_and_exact_prediction(self):
        a = self.args('--aggregation_layers', '2')
        a.enc_in = 7
        model = entry.build_model(a, 5)
        x, t, tau = torch.rand(2, 8, 7), torch.rand(2, 12, 5), torch.rand(2, 12)
        expected = model(x, t, tau)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'state.pt'
            torch.save(entry.complete_state(model), path)
            restored = entry.build_model(a, 5)
            self.assertFalse(torch.equal(restored(x, t, tau), expected))
            entry.restore_state(restored, torch.load(path, weights_only=True))
            torch.testing.assert_close(restored(x, t, tau), expected, rtol=0, atol=0)
            self.assertIs(restored.encoder.agg_layers[0], restored.encoder.agg_layers[1])

    def test_forecast_retains_reconstructed_initial_value(self):
        outputs = torch.tensor([[[2.], [3.], [4.], [5.]]])
        torch.testing.assert_close(entry.source_forecast(outputs, 2), torch.tensor([[[7.], [8.]]]))

    def test_quiet_source_dataset_matches_upstream_trainpy_split(self):
        with tempfile.TemporaryDirectory() as folder:
            frame = pd.DataFrame({'date': pd.date_range('2020-01-01', periods=200, freq='15min')})
            for i, c in enumerate(entry.ETT_COLUMNS):
                frame[c] = np.arange(200) * (i+1.)
            frame.to_csv(Path(folder) / 'ETTm1.csv', index=False)
            a = self.args('--protocol', 'trainpy', '--root_path', folder)
            datasets, manifest = entry.load_data(a)
            with contextlib.redirect_stdout(io.StringIO()):
                original = PDETimeDataset.train_test_val_split(frame.copy(), horizon=4, lookback=8,
                                                               val_ratio=.1, test_ratio=.2)
                for name, old in zip(['train', 'test', 'val'], original):
                    for left, right in zip(datasets[name][0], old[0]):
                        np.testing.assert_array_equal(left, right)
            self.assertEqual(manifest['context'], 'isolated, same as source train.py')
            self.assertEqual(datasets['train'].features.shape[1], 7)

    def test_paper_protocol_reports_its_actual_split_and_five_features(self):
        with tempfile.TemporaryDirectory() as folder:
            values = {'date': pd.date_range('2020-01-01', periods=200, freq='15min')}
            values.update({c: np.arange(200) for c in entry.ETT_COLUMNS})
            pd.DataFrame(values).to_csv(Path(folder) / 'ETTm1.csv', index=False)
            a = self.args('--root_path', folder)
            datasets, manifest = entry.load_data(a)
            self.assertEqual(manifest['splits']['train']['rows'], [0, 120])
            self.assertEqual(manifest['splits']['val']['rows'], [120, 160])
            self.assertEqual(len(manifest['temporal_columns']), 5)
            self.assertEqual(datasets['val'][0][3][8, 0], 128.)
            self.assertEqual(manifest['target_scaling'], 'none')

    def test_each_shell_runs_only_its_dataset_with_four_explicit_valid_commands(self):
        scripts = sorted((entry.ROOT / 'scripts/long_term_forecast').rglob('PDETime*.sh'))
        self.assertEqual(len(scripts), 7)
        with tempfile.TemporaryDirectory() as folder:
            stub = Path(folder) / 'python'
            log = Path(folder) / 'commands.jsonl'
            stub.write_text('#!/usr/bin/env python3\nimport os,sys,json\n'
                            'with open(os.environ["CAPTURE_COMMANDS"],"a") as out:\n'
                            ' out.write(json.dumps(sys.argv[1:])+"\\n")\n')
            stub.chmod(0o755)
            env = dict(os.environ, PATH=folder + os.pathsep + os.environ['PATH'], CAPTURE_COMMANDS=str(log))
            all_pairs = set()
            for script in scripts:
                subprocess.run(['bash', '-n', str(script)], check=True)
                log.write_text('')
                subprocess.run(['bash', str(script)], env=env, check=True)
                commands = [json.loads(row) for row in log.read_text().splitlines()]
                self.assertEqual(len(commands), 4)
                datasets = set()
                for tokens in commands:
                    self.assertEqual(tokens[:2], ['-u', 'train_reproduce.py'])
                    a = entry.parse_args(tokens[2:])
                    datasets.add(a.data)
                    all_pairs.add((a.data, a.pred_len))
                    n = entry.DATASETS[a.data][1]
                    train = int(n*(1-a.val_ratio-a.test_ratio))
                    val = int(n*a.val_ratio)
                    self.assertGreaterEqual(min(train, val, n-train-val), a.seq_len+a.pred_len)
                self.assertEqual(len(datasets), 1)
            self.assertEqual(len(all_pairs), 28)
            self.assertEqual({h for _,h in all_pairs}, {96,192,336,720})


if __name__ == '__main__':
    unittest.main()
