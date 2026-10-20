#!/usr/bin/env python3
"""PDETime paper-derived runner; disclosed paper values + train.py fallbacks.

See docs/experiments/02_paper_reproduction.md for assumptions. This is a
reconstruction, not a claim of verified author-official or bitwise replication.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from src.PDETime.models.encoder import AggLayer, Encoder
from src.PDETime.models.decoder import Decoder
from src.PDETime.models.solver import MLPLayer

DATASETS = {
    'ETTh1': (7, 17420), 'ETTh2': (7, 17420),
    'ETTm1': (7, 69680), 'ETTm2': (7, 69680),
    'Electricity': (321, 26304), 'Traffic': (862, 52696),
    'Weather': (21, 17544),
}
ETT_COLUMNS = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
PAPER_HORIZONS = [96, 192, 336, 720]
PAPER_MU = [1, 3, 5, 7, 9]
PAPER_LR = [1e-4, 1e-3, 5e-3]
PAPER_N = [1, 2, 3, 5]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + '\n', encoding='utf-8')


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def git_output(*args):
    result = subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True,
                            text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def provenance(config):
    files = sorted((ROOT / 'src').rglob('*.py')) + [Path(__file__).resolve(), ROOT / 'train.py']
    return {
        'git_sha': git_output('rev-parse', 'HEAD'),
        'tracked_worktree_status': git_output('status', '--short', '--untracked-files=no'),
        'source_sha256': {str(p.relative_to(ROOT)): file_hash(p) for p in files},
        'command': sys.argv, 'python': sys.version, 'platform': platform.platform(),
        'torch': str(torch.__version__), 'numpy': str(np.__version__),
        'pandas': str(pd.__version__), 'cuda_runtime': torch.version.cuda,
        'device': config['device'],
        'gpu': torch.cuda.get_device_name(0) if config['device'] == 'cuda' else None,
        'data_sha256': config.get('data_sha256') or file_hash(config['data']),
        'status': 'paper-derived; undisclosed details inherit train.py or named assumptions',
    }


def parameter_sources(config):
    return {
        'paper_disclosed': {
            'horizon_candidates': PAPER_HORIZONS, 'mu_candidates': PAPER_MU,
            'lr_candidates': PAPER_LR, 'aggregation_candidates': PAPER_N,
            'inr_layers_default': 5, 'batch_size': 32, 'seed': 2024,
            'split': 'ETT 6:2:2; others 7:1:2',
            'temporal_features': 'Appendix A.2: ETTm/Weather 5; others 4',
            'objective': 'Eq.12-16: observed last value, residual ridge, Lp+Lc+Lf',
        },
        'trainpy_fallbacks': {
            'epochs': 10, 'hidden_features': 64, 'latent_features': 64,
            'temporal_latent_features': 32, 'patch_length': 32,
            'mlp_hidden_layers': 3, 'mlp_hidden_features': 64, 'cff_scales': 1,
            'ridge_lambda': 0.1, 'value_scaler': 'none',
            'time_index_mode': 'split', 'calendar_scaling': 'split',
            'solver_weight_sharing': 'trainpy',
        },
        'reconstruction_choices': {
            'aggregation_weight_sharing_default': 'independent registered layers',
            'context_policy_default': 'carry known history; forecast labels stay in their split',
            'selection_metric_default': 'prediction_loss (Smooth L1 on future residuals)',
            'continuity': 'tail-anchored discrete Eq.12, before final Solver MLP',
            'reduction': 'mean over batch/time/channel or latent elements',
            'repeat_seed_mode_default': 'fixed; paper wording, not independent seeds',
        },
        'unresolved': ['author-official source identity', 'paper value/metric scaling',
                       'exact split/window and time-coordinate convention',
                       'per-task best S/dimensions/ridge and stopping budget',
                       'exact independent-repeat and validation-loss definition'],
        'actual': config,
    }


def seed_everything(seed, deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(deterministic)


def calendar_values(dates, with_minutes):
    names = ['day_of_year', 'month', 'day_of_week', 'hour']
    raw = [dates.dayofyear.to_numpy(), dates.month.to_numpy(),
           dates.dayofweek.to_numpy(), dates.hour.to_numpy()]
    minimum = [1., 1., 0., 0.]
    span = [365., 11., 6., 23.]
    if with_minutes:
        names.append('minute')
        raw.append(dates.minute.to_numpy())
        minimum.append(0.)
        span.append(59.)
    return np.stack(raw, axis=-1).astype('float64'), names, minimum, span


class WindowDataset(Dataset):
    """All labels belong to this split; only optional history crosses its left edge."""
    def __init__(self, prepared, config, split):
        self.lookback, self.horizon = config['lookback'], config['horizon']
        self.split = split
        begin, end = prepared.bounds[split]
        start = max(0, begin - self.lookback) if config['context_policy'] == 'carry' else begin
        if split == 'train':
            start = begin
        self.start, self.end = start, end
        self.values = prepared.values[start:end]
        self.columns = prepared.columns
        raw = prepared.calendar[start:end]
        if config['calendar_scaling'] == 'canonical':
            low, span = np.asarray(prepared.calendar_low), np.asarray(prepared.calendar_span)
        else:
            low = raw.min(axis=0)
            span = np.maximum(raw.max(axis=0) - low, 1.)
        self.temporal = ((raw - low) / span).astype('float32')
        timestamps = prepared.dates.asi8.astype('float64')
        if config['time_index_mode'] == 'split':
            origin, scale = timestamps[start], max(timestamps[end - 1] - timestamps[start], 1.)
        else:
            train_end = prepared.bounds['train'][1]
            origin, scale = timestamps[0], max(timestamps[train_end - 1] - timestamps[0], 1.)
        self.tau = ((timestamps[start:end] - origin) / scale).astype('float32')
        self.window_tau = np.linspace(0., 1., self.lookback + self.horizon, dtype='float32')
        self.local_time = config['time_index_mode'] == 'window'
        self.count = len(self.values) - self.lookback - self.horizon + 1
        if self.count <= 0:
            raise ValueError(f'{split} has no windows for L={self.lookback}, H={self.horizon}')
        self.manifest = {
            'label_split_rows': [begin, end], 'context_rows': [start, end],
            'first_forecast_row': start + self.lookback,
            'last_forecast_end_exclusive': end, 'windows': self.count,
            'date_begin': str(prepared.dates[start]), 'date_end': str(prepared.dates[end-1]),
            'calendar_minimum': low.tolist(), 'calendar_span': span.tolist(),
            'tau_origin_ns': origin, 'tau_scale_ns': scale,
        }
        assert start + self.lookback >= begin

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if index < 0 or index >= self.count:
            raise IndexError(index)
        middle = index + self.lookback
        end = middle + self.horizon
        tau = self.window_tau if self.local_time else self.tau[index:end]
        return (self.values[index:middle], self.temporal[index:end], tau,
                self.values[index:end])


class PreparedData:
    def __init__(self, config):
        self.data_hash = file_hash(config['data'])
        frame = pd.read_csv(config['data'])
        if 'date' not in frame:
            raise ValueError('CSV requires a date column and numeric target columns')
        self.dates = pd.DatetimeIndex(pd.to_datetime(frame['date'], errors='raise'))
        if self.dates.hasnans or not self.dates.is_monotonic_increasing or not self.dates.is_unique:
            raise ValueError('date must be valid, strictly chronological and unique')
        if config['dataset'].startswith('ETT'):
            if set(frame.columns) != set(['date'] + ETT_COLUMNS):
                raise ValueError('ETT requires exactly date + 7 real target columns')
            self.columns = ETT_COLUMNS[:]
        else:
            self.columns = [c for c in frame.columns if c != 'date']
        expected_channels, expected_rows = DATASETS[config['dataset']]
        if len(self.columns) != expected_channels:
            raise ValueError(f"{config['dataset']} expects {expected_channels} target columns, got {len(self.columns)}")
        raw = frame[self.columns].to_numpy(dtype='float64')
        if not np.isfinite(raw).all():
            raise ValueError('Target values contain NaN/Inf; no automatic filling is applied')
        n = len(raw)
        ratios = config['split_ratios'] or ([.6, .2, .2] if config['dataset'].startswith('ETT') else [.7, .1, .2])
        train_end, val_end = int(n * ratios[0]), int(n * ratios[0]) + int(n * ratios[1])
        if not (0 < train_end < val_end < n):
            raise ValueError('Empty split; check dataset length/ratios')
        self.bounds = {'train': (0, train_end), 'val': (train_end, val_end), 'test': (val_end, n)}
        self.mean = raw[:train_end].mean(axis=0) if config['value_scaler'] == 'standard' else np.zeros(raw.shape[1])
        self.scale = raw[:train_end].std(axis=0) if config['value_scaler'] == 'standard' else np.ones(raw.shape[1])
        self.scale[self.scale < 1e-12] = 1.
        self.values = ((raw - self.mean) / self.scale).astype('float32')
        if not np.isfinite(self.values).all():
            raise ValueError('Target values are not representable as finite float32')
        self.calendar, self.temporal_names, self.calendar_low, self.calendar_span = calendar_values(
            self.dates, config['dataset'] in ('ETTm1', 'ETTm2', 'Weather'))
        self.manifest = {
            'rows': n, 'columns': self.columns, 'temporal_columns': self.temporal_names,
            'split_ratios': ratios, 'split_rows': self.bounds,
            'value_scaler': config['value_scaler'], 'scaler_fit_rows': [0, train_end],
            'mean': self.mean.tolist(), 'scale': self.scale.tolist(),
            'paper_table5_rows': expected_rows, 'matches_table5_rows': n == expected_rows,
        }
        if n != expected_rows:
            message = f'CSV rows={n}; paper Table 5 reports {expected_rows}; data is not silently truncated'
            if config['strict_paper_shape']:
                raise ValueError(message)
            print('DATA VERSION NOTE:', message, flush=True)

    def dataset(self, config, split):
        return WindowDataset(self, config, split)


def independent_mlp(in_features, hidden, depth, out_features):
    layers = [nn.Linear(in_features, hidden), nn.GELU()]
    for _ in range(depth - 1):
        layers.extend([nn.Linear(hidden, hidden), nn.GELU()])
    layers.extend([nn.Linear(hidden, out_features), nn.GELU()])
    return nn.Sequential(*layers)


def tail_integral(derivative, anchor, patch_length):
    """Appendix Algorithm 2: integrate backwards from each patch's last value.

    derivative/anchor: [B,T,D]. Across patch endpoints a and b=a+S,
    Eq.12 continuity compares anchor[b] to anchor[a]+sum(v[a+1:b+1]).
    """
    b, t, d = derivative.shape
    if t % patch_length:
        raise ValueError('L+H must be divisible by patch length')
    v = derivative.reshape(b, -1, patch_length, d)
    endpoints = anchor.reshape(b, -1, patch_length, d)[:, :, -1:, :]
    negative_reversed = -v.flip(2)[:, :, :-1, :]
    z = torch.cat([endpoints, negative_reversed], dim=2).cumsum(2).flip(2)
    if endpoints.shape[1] > 1:
        continuity = F.smooth_l1_loss(endpoints[:, 1:, 0],
                                    endpoints[:, :-1, 0] + v[:, 1:].sum(2))
    else:
        continuity = derivative.sum() * 0.
    return z.reshape(b, t, d), continuity


class PaperPDETime(nn.Module):
    def __init__(self, config, channels, temporal_channels):
        super().__init__()
        self.lookback, self.horizon = config['lookback'], config['horizon']
        self.patch_length = config['patch_length']
        d, hidden, depth = config['latent_features'], config['mlp_hidden_features'], config['mlp_hidden_layers']
        self.encoder = Encoder(
            spatial_dim=channels, temporal_features=temporal_channels,
            temporal_latent_features=config['temporal_latent_features'],
            lookback=self.lookback, horizon=self.horizon, s_cff=config['cff_scales'],
            hidden_features=config['hidden_features'], INR_layers=config['inr_layers'],
            aggregation_layers=1, latent_features=d)
        first = self.encoder.agg_layers[0]
        layers = [first]
        for _ in range(config['aggregation_layers'] - 1):
            layers.append(first if config['aggregation_weight_sharing'] == 'trainpy' else AggLayer(
                latent_features=d, temporal_latent_features=config['temporal_latent_features'],
                spatial_dim=channels, lookback=self.lookback, horizon=self.horizon, attn_dim=64))
        self.encoder.agg_layers = nn.ModuleList(layers)
        factory = MLPLayer if config['solver_weight_sharing'] == 'trainpy' else independent_mlp
        self.mlp_dudt = factory(d, hidden, depth, d)
        self.mlp_u = factory(d, hidden, depth, d)
        self.mlp_integrate = factory(d, hidden, depth, d)
        self.decoder = Decoder(self.lookback, self.horizon, lambda_=config['ridge_lambda'])

    def forward(self, x, temporal, tau):
        encoded = self.encoder(tau, x, temporal)
        latent, continuity = tail_integral(self.mlp_dudt(encoded), self.mlp_u(encoded), self.patch_length)
        latent = self.mlp_integrate(latent)
        initial = x[:, -1:, :]  # observed history only, never future labels
        residual = self.decoder(latent, x - initial)
        return {'residual': residual, 'prediction': residual + initial, 'continuity': continuity}


def objective(result, labels, lookback, continuity_weight=1., difference_weight=1.):
    residual = result['residual']
    target_residual = labels - labels[:, lookback - 1:lookback]
    lp = F.smooth_l1_loss(residual[:, lookback:], target_residual[:, lookback:])
    lr = F.smooth_l1_loss(residual[:, :lookback], target_residual[:, :lookback])
    lf = F.smooth_l1_loss(residual[:, lookback:] - residual[:, lookback - 1:-1],
                          labels[:, lookback:] - labels[:, lookback - 1:-1])
    lc = result['continuity']
    return {'loss': lp + continuity_weight * lc + difference_weight * lf,
            'prediction_loss': lp, 'continuity_loss': lc, 'difference_loss': lf,
            'reconstruction_diagnostic': lr}


def loader(dataset, config, shuffle):
    generator = torch.Generator().manual_seed(config['seed']) if shuffle else None
    return DataLoader(dataset, batch_size=config['batch_size'], shuffle=shuffle,
                      generator=generator, num_workers=config['num_workers'],
                      pin_memory=config['device'] == 'cuda')


def move_batch(batch, device):
    return [item.to(device=device, dtype=torch.float32, non_blocking=True) for item in batch]


@torch.no_grad()
def evaluate(model, dataset, config, prepared, collect_example=False):
    model.eval()
    count, samples = 0, 0
    sums = {key: 0. for key in ('loss', 'prediction_loss', 'continuity_loss',
                              'difference_loss', 'reconstruction_diagnostic')}
    channels = len(prepared.columns)
    squares = torch.zeros(channels, dtype=torch.float64, device=config['device'])
    absolutes, raw_squares, raw_absolutes = squares.clone(), squares.clone(), squares.clone()
    scale = torch.tensor(prepared.scale, device=config['device'], dtype=torch.float32)
    example = None
    for index, batch in enumerate(loader(dataset, config, False)):
        if config['max_eval_batches'] is not None and index >= config['max_eval_batches']:
            break
        x, temporal, tau, labels = move_batch(batch, config['device'])
        result = model(x, temporal, tau)
        losses = objective(result, labels, config['lookback'], config['continuity_weight'], config['difference_weight'])
        error = result['prediction'][:, config['lookback']:] - labels[:, config['lookback']:]
        raw_error = error * scale
        squares += error.square().sum((0, 1), dtype=torch.float64)
        absolutes += error.abs().sum((0, 1), dtype=torch.float64)
        raw_squares += raw_error.square().sum((0, 1), dtype=torch.float64)
        raw_absolutes += raw_error.abs().sum((0, 1), dtype=torch.float64)
        count += error.shape[0] * error.shape[1]
        samples += len(x)
        for key, value in losses.items():
            sums[key] += value.item() * len(x)
        if collect_example and example is None:
            column = config['plot_column'] or ('OT' if 'OT' in prepared.columns else prepared.columns[0])
            if column not in prepared.columns:
                raise ValueError(f'Unknown plot column {column!r}')
            c = prepared.columns.index(column)
            actual = labels[0, :, c].cpu().numpy() * prepared.scale[c] + prepared.mean[c]
            predicted = result['prediction'][0, config['lookback']:, c].cpu().numpy() * prepared.scale[c] + prepared.mean[c]
            example = (column, actual, predicted)
    if not count:
        raise ValueError('No evaluation samples processed')
    metric = {key: value / samples for key, value in sums.items()}
    metric.update({
        'mse': (squares / count).mean().item(), 'mae': (absolutes / count).mean().item(),
        'raw_mse': (raw_squares / count).mean().item(), 'raw_mae': (raw_absolutes / count).mean().item(),
        'metric_scale': config['value_scaler'], 'num_forecast_values': count * channels,
        'evaluated_windows': samples,
        'per_channel': {name: {'mse': squares[c].item() / count, 'mae': absolutes[c].item() / count,
                               'raw_mse': raw_squares[c].item() / count,
                               'raw_mae': raw_absolutes[c].item() / count}
                        for c, name in enumerate(prepared.columns)},
    })
    if not all(math.isfinite(metric[k]) for k in ('loss', 'mse', 'mae', 'raw_mse', 'raw_mae')):
        raise FloatingPointError('Evaluation produced NaN/Inf')
    return metric, example


def save_plot(example, lookback, path):
    if example is None:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    column, actual, prediction = example
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(actual, label='actual ' + column)
    ax.plot(np.arange(lookback, len(actual)), prediction, label='forecast ' + column)
    ax.axvline(lookback - 1, linestyle='--', color='grey')
    ax.set(xlabel='time step', ylabel=column + ' (raw scale)', title='PDETime paper-derived test example')
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train_one(config, prepared, directory, test_after_selection):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    seed_everything(config['seed'], config['deterministic'])
    train_data, val_data = prepared.dataset(config, 'train'), prepared.dataset(config, 'val')
    write_json(directory / 'config.json', config)
    write_json(directory / 'parameter_sources.json', parameter_sources(config))
    write_json(directory / 'provenance.json', provenance(config))
    manifest = deepcopy(prepared.manifest)
    manifest['windows'] = {'train': train_data.manifest, 'val': val_data.manifest}
    # Test dataset/labels are not constructed or read by any search trial.
    write_json(directory / 'data_manifest.json', manifest)
    model = PaperPDETime(config, len(prepared.columns), len(prepared.temporal_names)).to(config['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    if config['device'] == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    best, best_epoch, history = float('inf'), 0, []
    train_loader = loader(train_data, config, True)
    checkpoint_path = directory / 'best.pt'
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        totals, samples = {}, 0
        for index, batch in enumerate(train_loader):
            if config['max_train_batches'] is not None and index >= config['max_train_batches']:
                break
            x, temporal, tau, labels = move_batch(batch, config['device'])
            optimizer.zero_grad(set_to_none=True)
            result = model(x, temporal, tau)
            losses = objective(result, labels, config['lookback'], config['continuity_weight'], config['difference_weight'])
            if not torch.isfinite(losses['loss']):
                raise FloatingPointError('Non-finite training loss')
            losses['loss'].backward()
            optimizer.step()
            samples += len(x)
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.) + value.item() * len(x)
        if not samples:
            raise ValueError('No training samples processed')
        validation, _ = evaluate(model, val_data, config, prepared)
        score = validation[config['selection_metric']]
        row = {'epoch': epoch, **{'train_' + k: v / samples for k, v in totals.items()},
               **{'val_' + k: validation[k] for k in totals},
               'val_mse': validation['mse'], 'val_mae': validation['mae'],
               'elapsed_seconds': time.perf_counter() - start}
        history.append(row)
        with (directory / 'history.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        print(json.dumps({'run': str(directory), **row}), flush=True)
        if score < best:
            best, best_epoch = score, epoch
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                        'config': config, 'data_sha256': prepared.data_hash,
                        'data_manifest': prepared.manifest, 'epoch': epoch, 'best_validation': best}, checkpoint_path)
    if config['device'] == 'cuda':
        torch.cuda.synchronize()
    summary = {'config': config, 'best_epoch': best_epoch, 'best_validation': best,
               'selection_metric': config['selection_metric'],
               'partial': bool(config['max_train_batches'] or config['max_eval_batches']),
               'fit_seconds': time.perf_counter() - start,
               'peak_vram_bytes': torch.cuda.max_memory_allocated() if config['device'] == 'cuda' else None,
               'parameter_count': sum(p.numel() for p in model.parameters()),
               'test': None, 'status': 'paper-derived; not verified paper-table reproduction'}
    if test_after_selection:
        state = torch.load(checkpoint_path, map_location=config['device'], weights_only=True)
        model.load_state_dict(state['model_state_dict'])
        test_data = prepared.dataset(config, 'test')
        summary['test'], example = evaluate(model, test_data, config, prepared, not config['no_plot'])
        manifest['windows']['test'] = test_data.manifest
        write_json(directory / 'data_manifest.json', manifest)
        if not config['no_plot']:
            save_plot(example, config['lookback'], directory / 'prediction.png')
    write_json(directory / 'summary.json', summary)
    del model, optimizer
    if config['device'] == 'cuda':
        torch.cuda.empty_cache()
    return summary


def repeat_config(config, repeat):
    result = deepcopy(config)
    result['repeat_id'] = repeat
    result['seed'] = config['seed'] + (repeat if config['repeat_seed_mode'] == 'offset' else 0)
    return result


def aggregate_runs(runs, seed_mode):
    result = {'repeat_seed_mode': seed_mode, 'independent_seeds': seed_mode == 'offset',
              'seeds': [r['config']['seed'] for r in runs], 'runs': runs,
              'partial': any(r['partial'] for r in runs)}
    result['aggregate'] = {}
    for key in ('mse', 'mae', 'raw_mse', 'raw_mae'):
        values = [r['test'][key] for r in runs]
        result['aggregate'][key] = {'mean': float(np.mean(values)),
                                   'std': float(np.std(values, ddof=1)) if len(values) > 1 else 0.}
    return result


def run_search(config, prepared, directory):
    records, selected = [], []
    for horizon in config['horizons']:
        best_config, best_score = None, float('inf')
        for mu, lr, n in itertools.product(config['mu_grid'], config['lr_grid'], config['aggregation_grid']):
            trial = deepcopy(config)
            trial.update(horizon=horizon, lookback=horizon * mu, lookback_multiplier=mu, lr=lr, aggregation_layers=n)
            validate_config(trial)
            trial_dir = directory / f'H{horizon}' / f'mu{mu}_lr{lr:g}_N{n}'
            record = {'horizon': horizon, 'mu': mu, 'lr': lr, 'aggregation_layers': n, 'run_dir': str(trial_dir)}
            try:
                summary = train_one(trial, prepared, trial_dir, test_after_selection=False)
                score = summary['best_validation']
                record.update(status='complete', validation=score)
                if score < best_score:
                    best_score, best_config = score, trial
            except (ValueError, FloatingPointError, torch.cuda.OutOfMemoryError) as exc:
                record.update(status='failed', error=str(exc))
                write_json(trial_dir.parent / (trial_dir.name + '_failure.json'), record)
                if config['device'] == 'cuda':
                    torch.cuda.empty_cache()
                print(json.dumps(record), flush=True)
            records.append(record)
            write_json(directory / 'trials.json', records)
        if best_config is None:
            raise RuntimeError(f'All validation trials failed at H={horizon}; inspect trials.json')
        write_json(directory / f'selected_H{horizon}.json', best_config)
        # Freeze the selected configuration before any final test evaluation.
        runs = [train_one(repeat_config(best_config, r), prepared,
                          directory / f'H{horizon}' / f'final_repeat{r}', True)
                for r in range(config['repeats'])]
        report = aggregate_runs(runs, config['repeat_seed_mode'])
        write_json(directory / f'results_H{horizon}.json', report)
        selected.append({'horizon': horizon, 'selected_validation': best_score, 'results': report})
    write_json(directory / 'summary.json', {'horizons': selected, 'trials': records})


def resolve_data(dataset, root):
    names = {'Electricity': ['electricity.csv', 'Electricity.csv'],
             'Traffic': ['traffic.csv', 'Traffic.csv'], 'Weather': ['weather.csv', 'Weather.csv']}.get(dataset, [dataset + '.csv'])
    folders = [Path(root), Path(root) / 'ETT-small', Path(root) / dataset,
               Path(root) / dataset.lower(), Path(root) / 'electricity' if dataset == 'Electricity' else Path(root)]
    for folder in folders:
        for name in names:
            if (folder / name).is_file():
                return str((folder / name).resolve())
    raise FileNotFoundError(f'Cannot locate {dataset} CSV below {root}; pass --data explicitly')


def validate_config(config):
    positive = ('horizon', 'lookback', 'lookback_multiplier', 'batch_size', 'epochs', 'repeats', 'inr_layers',
                'aggregation_layers', 'patch_length', 'hidden_features', 'latent_features',
                'temporal_latent_features', 'mlp_hidden_layers', 'mlp_hidden_features', 'cff_scales')
    if any(config[k] <= 0 for k in positive):
        raise ValueError('Lengths, widths, epochs, repeats and layer counts must be positive')
    if config['inr_layers'] < 2 or config['latent_features'] % 2:
        raise ValueError('INR layers must be >=2; latent features must be even for CFF')
    if (config['lookback'] + config['horizon']) % config['patch_length']:
        raise ValueError('lookback + horizon must be divisible by --patch-length')
    if config['num_workers'] < 0 or config['seed'] < 0:
        raise ValueError('num-workers and seed must be nonnegative')
    for key in ('lr', 'ridge_lambda', 'continuity_weight', 'difference_weight'):
        if not math.isfinite(config[key]) or config[key] < 0 or (key == 'lr' and config[key] == 0):
            raise ValueError(f'Invalid {key}')
    for key in ('max_train_batches', 'max_eval_batches'):
        if config[key] is not None and config[key] < 1:
            raise ValueError(f'{key} must be positive')
    if config['split_ratios'] and (any(x <= 0 or not math.isfinite(x) for x in config['split_ratios'])
                                   or not math.isclose(sum(config['split_ratios']), 1.)):
        raise ValueError('--split-ratios must contain three positive values summing to 1')
    for key in ('horizons', 'mu_grid', 'lr_grid', 'aggregation_grid'):
        if len(set(config[key])) != len(config[key]) or any(x <= 0 or not math.isfinite(x) for x in config[key]):
            raise ValueError(f'{key} must contain unique positive finite values')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode', choices=('train', 'search', 'evaluate'), default='train')
    parser.add_argument('--dataset', choices=tuple(DATASETS), default='ETTm1')
    parser.add_argument('--data', help='CSV path; otherwise locate dataset under --data-root')
    parser.add_argument('--data-root', default='data')
    parser.add_argument('--run-dir', required=True, help='New output directory; existing output is never overwritten')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    for flag, default in [('horizon', 96), ('lookback-multiplier', 5), ('epochs', 10), ('batch-size', 32),
                          ('seed', 2024), ('num-workers', 0), ('repeats', 1), ('inr-layers', 5),
                          ('aggregation-layers', 1), ('patch-length', 32), ('hidden-features', 64),
                          ('latent-features', 64), ('temporal-latent-features', 32),
                          ('mlp-hidden-layers', 3), ('mlp-hidden-features', 64), ('cff-scales', 1)]:
        parser.add_argument('--' + flag, type=int, default=default)
    parser.add_argument('--lookback', type=int, help='Explicit L, overrides H times multiplier; not used in search')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--ridge-lambda', type=float, default=.1)
    parser.add_argument('--continuity-weight', type=float, default=1.)
    parser.add_argument('--difference-weight', type=float, default=1.)
    parser.add_argument('--split-ratios', type=float, nargs=3, metavar=('TRAIN', 'VAL', 'TEST'))
    parser.add_argument('--context-policy', choices=('carry', 'isolated'), default='carry')
    parser.add_argument('--value-scaler', choices=('none', 'standard'), default='none')
    parser.add_argument('--calendar-scaling', choices=('split', 'canonical'), default='split')
    parser.add_argument('--time-index-mode', choices=('split', 'train', 'window'), default='split')
    parser.add_argument('--solver-weight-sharing', choices=('trainpy', 'independent'), default='trainpy')
    parser.add_argument('--aggregation-weight-sharing', choices=('trainpy', 'independent'), default='independent')
    parser.add_argument('--selection-metric', choices=('prediction_loss', 'loss', 'mse'), default='prediction_loss')
    parser.add_argument('--repeat-seed-mode', choices=('fixed', 'offset'), default='fixed')
    parser.add_argument('--horizons', type=int, nargs='+', default=PAPER_HORIZONS)
    parser.add_argument('--mu-grid', type=int, nargs='+', default=PAPER_MU)
    parser.add_argument('--lr-grid', type=float, nargs='+', default=PAPER_LR)
    parser.add_argument('--aggregation-grid', type=int, nargs='+', default=PAPER_N)
    parser.add_argument('--max-train-batches', type=int)
    parser.add_argument('--max-eval-batches', type=int)
    parser.add_argument('--checkpoint', help='Required for --mode evaluate; restores saved configuration')
    parser.add_argument('--eval-split', choices=('val', 'test'), default='test')
    parser.add_argument('--plot-column', help='Default OT for ETT, otherwise first real target')
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--strict-paper-shape', action='store_true', help='Reject row counts different from PDF Table 5')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Print resolved parameter sources; no data/GPU required')
    args = parser.parse_args(argv)
    explicit_lookback = args.lookback
    if args.mode == 'search' and explicit_lookback is not None:
        parser.error('search uses --mu-grid; do not pass --lookback')
    config = vars(args)
    config['lookback'] = args.lookback if args.lookback is not None else args.horizon * args.lookback_multiplier
    try:
        validate_config(config)
    except ValueError as exc:
        parser.error(str(exc))
    return config


def evaluate_checkpoint(cli):
    if not cli['checkpoint']:
        raise ValueError('--mode evaluate requires --checkpoint')
    state = torch.load(cli['checkpoint'], map_location='cpu', weights_only=True)
    config = deepcopy(state['config'])
    for key in ('device', 'batch_size', 'num_workers', 'max_eval_batches', 'no_plot', 'plot_column'):
        config[key] = cli[key]
    config['data'] = str(Path(cli['data'] or config['data']).resolve())
    if file_hash(config['data']) != state['data_sha256']:
        raise ValueError('Evaluation CSV hash differs from the training CSV')
    prepared = PreparedData(config)
    if prepared.manifest != state['data_manifest']:
        # JSON serializes tuples as lists, while checkpoints preserve tuples.
        if json.dumps(prepared.manifest, sort_keys=True) != json.dumps(state['data_manifest'], sort_keys=True):
            raise ValueError('Evaluation preprocessing differs from checkpoint')
    seed_everything(config['seed'], config['deterministic'])
    model = PaperPDETime(config, len(prepared.columns), len(prepared.temporal_names)).to(config['device'])
    model.load_state_dict(state['model_state_dict'])
    metric, example = evaluate(model, prepared.dataset(config, cli['eval_split']), config, prepared, not config['no_plot'])
    directory = Path(cli['run_dir'])
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / 'metrics.json', {'split': cli['eval_split'], 'metrics': metric,
               'partial': bool(config['max_train_batches'] or config['max_eval_batches']),
               'checkpoint': str(Path(cli['checkpoint']).resolve()), 'config': config})
    if not config['no_plot']:
        save_plot(example, config['lookback'], directory / 'prediction.png')
    print(json.dumps(metric), flush=True)


def main(argv=None):
    config = parse_args(argv)
    if config['dry_run']:
        plan = parameter_sources(config)
        plan['search_trials'] = (len(config['horizons']) * len(config['mu_grid']) * len(config['lr_grid']) * len(config['aggregation_grid'])) if config['mode'] == 'search' else 0
        plan['final_fits'] = (len(config['horizons']) if config['mode'] == 'search' else 1) * config['repeats']
        print(json.dumps(plan, indent=2))
        return
    if config['device'] == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; choose --device cpu for a local smoke test')
    if config['mode'] == 'evaluate':
        evaluate_checkpoint(config)
        return
    config['data'] = str(Path(config['data']).resolve()) if config['data'] else resolve_data(config['dataset'], config['data_root'])
    prepared = PreparedData(config)
    config['data_sha256'] = prepared.data_hash
    directory = Path(config['run_dir'])
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / 'parameter_sources.json', parameter_sources(config))
    if config['mode'] == 'search':
        run_search(config, prepared, directory)
    else:
        runs = [train_one(repeat_config(config, r), prepared, directory / f'repeat{r}', True)
                for r in range(config['repeats'])]
        write_json(directory / 'summary.json', aggregate_runs(runs, config['repeat_seed_mode']))
    print('Saved results:', directory.resolve(), flush=True)


if __name__ == '__main__':
    main()
