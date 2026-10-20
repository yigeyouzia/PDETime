"""Mechanism and split-boundary tests, independent of external datasets/GPU."""
import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch
from torch import nn

from scripts import reproduce_paper as runner


class PaperReproductionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        frame = pd.DataFrame({'date': pd.date_range('2020-01-01', periods=200, freq='15min')})
        for index, column in enumerate(runner.ETT_COLUMNS):
            frame[column] = np.arange(200) * (index + 1.) + 10.
        self.csv = self.folder / 'ETTm1.csv'
        frame.to_csv(self.csv, index=False)
        self.config = runner.parse_args([
            '--data', str(self.csv), '--run-dir', str(self.folder / 'run'), '--device', 'cpu',
            '--horizon', '4', '--lookback', '8', '--patch-length', '4',
            '--hidden-features', '8', '--latent-features', '8',
            '--temporal-latent-features', '4', '--inr-layers', '2',
            '--mlp-hidden-features', '8', '--aggregation-layers', '2',
            '--epochs', '1', '--batch-size', '2', '--no-plot'])

    def prepared(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return runner.PreparedData(self.config)

    def test_scaler_only_fits_training_and_targets_are_real_columns(self):
        self.config['value_scaler'] = 'standard'
        data = self.prepared()
        np.testing.assert_allclose(data.mean, np.arange(1., 8.) * 59.5 + 10.)
        self.assertEqual(data.values.shape, (200, 7))
        self.assertEqual(len(data.temporal_names), 5)
        self.assertEqual(data.bounds['train'], (0, 120))
        np.testing.assert_allclose(data.values[:120].mean(0), 0., atol=1e-6)
        self.assertTrue((data.values[160:].mean(0) > 1.).all())

    def test_carry_history_never_moves_future_labels_across_split(self):
        data = self.prepared()
        val = data.dataset(self.config, 'val')
        x, temporal, tau, labels = val[0]
        np.testing.assert_array_equal(x, data.values[112:120])
        np.testing.assert_array_equal(labels[8:], data.values[120:124])
        self.assertEqual(len(val), 37)
        self.assertEqual(temporal.shape, (12, 5))
        self.assertEqual(tau.shape, (12,))
        test = data.dataset(self.config, 'test')
        self.assertEqual(test.manifest['first_forecast_row'], 160)
        self.assertEqual(test.manifest['last_forecast_end_exclusive'], 200)

    def test_constant_perfect_prediction_has_zero_prediction_and_difference_loss(self):
        labels = torch.full((2, 12, 7), 10.)
        residual = torch.zeros_like(labels, requires_grad=True)
        result = {'residual': residual, 'continuity': residual.sum() * 0.}
        losses = runner.objective(result, labels, 8)
        self.assertEqual(losses['loss'].item(), 0.)
        losses['loss'].backward()
        self.assertEqual(residual.grad.abs().max().item(), 0.)

    def test_reconstruction_is_diagnostic_not_extra_outer_term(self):
        residual = torch.zeros(1, 12, 1)
        residual[:, :7] = 3.
        result = {'residual': residual, 'continuity': torch.tensor(0.)}
        losses = runner.objective(result, torch.zeros_like(residual), 8)
        self.assertGreater(losses['reconstruction_diagnostic'].item(), 0.)
        self.assertEqual(losses['loss'].item(), 0.)

    def test_tail_integral_and_patch_continuity_have_consistent_indices(self):
        v = torch.ones(1, 8, 1, requires_grad=True)
        anchor = torch.arange(8.).reshape(1, 8, 1).requires_grad_()
        z, lc = runner.tail_integral(v, anchor, 4)
        torch.testing.assert_close(z, torch.arange(8.).reshape(1, 8, 1))
        self.assertEqual(lc.item(), 0.)
        bad_anchor = anchor.detach().clone()
        bad_anchor[:, -1] += 2.
        bad_anchor.requires_grad_()
        _, bad_lc = runner.tail_integral(v, bad_anchor, 4)
        self.assertGreater(bad_lc.item(), 0.)
        bad_lc.backward()
        self.assertGreater(v.grad.abs().sum().item(), 0.)
        self.assertGreater(bad_anchor.grad.abs().sum().item(), 0.)

    def test_registered_independent_aggregation_and_explicit_solver_fallback(self):
        model = runner.PaperPDETime(self.config, 7, 5)
        self.assertIsInstance(model.encoder.agg_layers, nn.ModuleList)
        self.assertIsNot(model.encoder.agg_layers[0], model.encoder.agg_layers[1])
        self.assertTrue(any('encoder.agg_layers.1' in key for key in model.state_dict()))
        self.assertIs(model.mlp_dudt.net[2], model.mlp_dudt.net[4])
        self.config['solver_weight_sharing'] = 'independent'
        other = runner.PaperPDETime(self.config, 7, 5)
        self.assertIsNot(other.mlp_dudt[2], other.mlp_dudt[4])

    def test_forward_backward_and_safe_checkpoint_roundtrip(self):
        model = runner.PaperPDETime(self.config, 7, 5)
        x, temporal, tau = torch.randn(2, 8, 7), torch.randn(2, 12, 5), torch.randn(2, 12)
        result = model(x, temporal, tau)
        self.assertEqual(result['prediction'].shape, (2, 12, 7))
        labels = torch.cat([x, torch.randn(2, 4, 7)], dim=1)
        losses = runner.objective(result, labels, 8)
        losses['loss'].backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        path = self.folder / 'state.pt'
        torch.save({'model_state_dict': model.state_dict()}, path)
        restored = runner.PaperPDETime(self.config, 7, 5)
        restored.load_state_dict(torch.load(path, weights_only=True)['model_state_dict'])
        torch.testing.assert_close(restored(x, temporal, tau)['prediction'], result['prediction'])

    def test_observed_constant_history_is_exact_forecast_without_future_input(self):
        model = runner.PaperPDETime(self.config, 7, 5)
        x = torch.full((2, 8, 7), 10.)
        result = model(x, torch.rand(2, 12, 5), torch.rand(2, 12))
        torch.testing.assert_close(result['prediction'], torch.full((2, 12, 7), 10.))

    def test_all_dataset_adapters_keep_expected_channels_and_temporal_features(self):
        for name, (channels, _) in runner.DATASETS.items():
            with self.subTest(dataset=name):
                columns = runner.ETT_COLUMNS if name.startswith('ETT') else ['c' + str(i) for i in range(channels)]
                table = {'date': pd.date_range('2020-01-01', periods=200, freq='15min')}
                table.update({c: np.arange(200, dtype=float) + i for i, c in enumerate(columns)})
                path = self.folder / (name + '.csv')
                pd.DataFrame(table).to_csv(path, index=False)
                config = dict(self.config, dataset=name, data=str(path), aggregation_layers=1)
                with contextlib.redirect_stdout(io.StringIO()):
                    prepared = runner.PreparedData(config)
                expected_temporal = 5 if name in ('ETTm1', 'ETTm2', 'Weather') else 4
                self.assertEqual(prepared.values.shape[1], channels)
                self.assertEqual(len(prepared.temporal_names), expected_temporal)
                batch = next(iter(runner.loader(prepared.dataset(config, 'val'), config, False)))
                x, temporal, tau, _ = runner.move_batch(batch, 'cpu')
                result = runner.PaperPDETime(config, channels, expected_temporal)(x, temporal, tau)
                self.assertEqual(result['prediction'].shape, (2, 12, channels))

    def test_four_paper_horizons_forward_with_trainpy_patch_length(self):
        for horizon in runner.PAPER_HORIZONS:
            with self.subTest(horizon=horizon):
                config = dict(self.config, horizon=horizon, lookback=horizon, patch_length=32, aggregation_layers=1)
                model = runner.PaperPDETime(config, 7, 5)
                result = model(torch.rand(1, horizon, 7), torch.rand(1, 2*horizon, 5), torch.rand(1, 2*horizon))
                self.assertEqual(result['prediction'].shape, (1, 2*horizon, 7))
                self.assertTrue(torch.isfinite(result['prediction']).all())

    def test_search_selects_on_validation_before_any_final_test(self):
        config = dict(self.config, horizons=[4], mu_grid=[1, 3], lr_grid=[.001],
                      aggregation_grid=[1], repeats=2, lookback=4)
        calls = []
        def fake_train(trial, prepared, directory, test_after_selection):
            calls.append((trial['lookback_multiplier'], test_after_selection))
            metric = {'mse': 1., 'mae': 1., 'raw_mse': 1., 'raw_mae': 1.}
            return {'config': trial, 'best_validation': 1. / trial['lookback_multiplier'],
                    'partial': False, 'test': metric if test_after_selection else None}
        with mock.patch.object(runner, 'train_one', side_effect=fake_train):
            runner.run_search(config, None, self.folder)
        self.assertEqual(calls, [(1, False), (3, False), (3, True), (3, True)])
        selected = json.loads((self.folder / 'selected_H4.json').read_text())
        self.assertEqual(selected['lookback_multiplier'], 3)

    def test_search_cli_derives_default_lookback_without_treating_it_as_explicit(self):
        config = runner.parse_args(['--mode', 'search', '--run-dir', 'unused', '--dry-run'])
        self.assertEqual(config['lookback'], 480)
        self.assertEqual(config['mu_grid'], [1, 3, 5, 7, 9])
        source = runner.provenance(dict(self.config, data_sha256='test-hash'))
        self.assertIn('scripts/reproduce_paper.py', source['source_sha256'])

    def test_shell_enumerates_seven_datasets_four_horizons(self):
        import shlex
        process = subprocess.run(['bash', str(runner.ROOT / 'scripts/reproduce_paper.sh'), 'search', '--dry-run'],
                                 capture_output=True, text=True, check=True)
        commands = [shlex.split(line) for line in process.stdout.splitlines()]
        self.assertEqual(len(commands), 28)
        pairs = {(c[c.index('--dataset')+1], int(c[c.index('--horizon')+1])) for c in commands}
        self.assertEqual(pairs, set(__import__('itertools').product(runner.DATASETS, runner.PAPER_HORIZONS)))
        for c in commands:
            start = c.index('--mu-grid') + 1
            self.assertEqual(c[start:start+5], ['1', '3', '5', '7', '9'])


if __name__ == '__main__':
    unittest.main()
