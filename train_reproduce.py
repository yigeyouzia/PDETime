#!/usr/bin/env python3
"""TSLib-style CLI around the unchanged source PDETime and PDETimeLoss.

This entry retains train.py's model/loss/forecast/optimizer coverage, including
unregistered and shared aggregation layers. See docs/experiments/03_tslib_entry.md.
"""
import argparse
import csv
import hashlib
import json
import math
import platform
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn import __version__ as sklearn_version

from src.PDETime.models.pdetime_model import PDETime, PDETimeLoss
from src.PDETime.preprocess.dataloader import PDETimeDataset

ROOT = Path(__file__).resolve().parent
DATASETS = {'ETTh1': (7, 17420), 'ETTh2': (7, 17420), 'ETTm1': (7, 69680),
            'ETTm2': (7, 69680), 'Electricity': (321, 26304),
            'Traffic': (862, 52696), 'Weather': (21, 17544)}
ETT_COLUMNS = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
FORMAT = 'pdetime-source-train-v1'


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def source_hashes():
    files = [ROOT / 'train.py'] + sorted((ROOT / 'src/PDETime').rglob('*.py'))
    return {str(p.relative_to(ROOT)): sha256(p) for p in files}


def git_read(*args):
    result = subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def parameter_sources(args):
    return {
        'actual': vars(args),
        'paper': {'H': [96, 192, 336, 720], 'mu_candidates': [1, 3, 5, 7, 9],
                  'lr_candidates': [1e-4, 1e-3, 5e-3], 'N_candidates': [1, 2, 3, 5],
                  'batch': 32, 'seed': 2024, 'repeats': 3, 'INR_layers': 5,
                  'splits': 'ETT 6:2:2, others 7:1:2', 'time_features': 'Appendix A.2: 5 or 4'},
        'trainpy_fallbacks': {'mu_starting_value': 5, 'lr_starting_value': 1e-4,
                             'epochs': 10, 'hidden_features': 64, 'latent_features': 64,
                             'temporal_latent_features': 32, 'aggregation_layers': 1,
                             'patch_length': 32, 'MLP_hidden_layers': 3, 'MLP_hidden_features': 64,
                             's_cff': 1, 'ridge_lambda': .1, 'target_scaling': 'none',
                             'scheduler': None, 'early_stopping': None,
                             'window_context': 'isolated splits', 'time_scaling': 'per split MinMax'},
        'preserved_source_semantics': ['original PDETime and PDETimeLoss', 'raw-history ridge fit',
            'forecast offset from last reconstructed output', 'original Lr+Lp+Lf objective, no added Lc',
            'unregistered aggregation omitted from optimizer, as train.py',
            'aggregation/MLP shared layers retained'],
        'added_engineering': ['seeded execution', 'validation checkpoint selection',
            'complete auxiliary checkpoint for unregistered layers', 'test metrics and named-column plot'],
        'unresolved': ['author-official repository identity', 'paper/code objective and initial-value conflict',
            'exact target/metric scaling', 'best per-task hyperparameters', 'training budget and repeat independence'],
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--task_name', choices=['long_term_forecast'], default='long_term_forecast')
    p.add_argument('--is_training', type=int, choices=[0, 1], default=1)
    p.add_argument('--model', choices=['PDETime'], default='PDETime')
    p.add_argument('--model_id', help='Experiment name; defaults to dataset_L_H')
    p.add_argument('--des', default='source_train_semantics')
    p.add_argument('--data', choices=list(DATASETS), default='ETTm1')
    p.add_argument('--root_path', help='CSV directory; training default ./data/')
    p.add_argument('--data_path', help='CSV filename; inferred from --data when omitted')
    p.add_argument('--features', choices=['M'], default='M')
    p.add_argument('--protocol', choices=['paper', 'trainpy'], default='paper',
                   help='paper: reported splits and 5/4 time features; trainpy: 7:1:2 and auto time features')
    p.add_argument('--seq_len', type=int, help='L; if omitted use 5*pred_len as train.py')
    p.add_argument('--pred_len', type=int, default=96, help='H')
    p.add_argument('--enc_in', type=int, help='Check real target-column count; inferred otherwise')
    p.add_argument('--c_out', type=int, help='Must equal enc_in: source model predicts all real variables')
    for name, default in [('train_epochs', 10), ('batch_size', 32), ('itr', 3), ('seed', 2024),
                          ('num_workers', 0), ('gpu', 0), ('inr_layers', 5), ('aggregation_layers', 1),
                          ('patch_length', 32), ('hidden_features', 64), ('latent_features', 64),
                          ('temporal_latent_features', 32), ('mlp_hidden_layers', 3),
                          ('mlp_hidden_features', 64), ('s_cff', 1)]:
        p.add_argument('--' + name, type=int, default=default, help=name.replace('_', ' '))
    p.add_argument('--learning_rate', type=float, default=1e-4, help='Adam initial learning rate')
    p.add_argument('--ridge_lambda', type=float, default=.1, help='Source decoder regularization_strength')
    p.add_argument('--smooth_l1_beta', type=float, default=1., help='Source SmoothL1Loss beta')
    p.add_argument('--val_ratio', type=float, help='Optional explicit split override')
    p.add_argument('--test_ratio', type=float, help='Optional explicit split override')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--selection_metric', choices=['loss', 'mse'], default='loss',
                   help='Validation checkpoint selector; loss is the original PDETimeLoss')
    p.add_argument('--evaluate_test', type=int, choices=[0, 1], default=1,
                   help='Set 0 for validation-only hyperparameter trials')
    p.add_argument('--checkpoints', default='./runs/tslib', help='New experiment directories below this root')
    p.add_argument('--checkpoint', help='Required for is_training=0; restores saved model/data configuration')
    p.add_argument('--plot_column', help='OT if present, otherwise first real variable')
    p.add_argument('--no_plot', action='store_true')
    p.add_argument('--max_train_batches', type=int, help='Smoke test only')
    p.add_argument('--max_eval_batches', type=int, help='Partial evaluation only')
    p.add_argument('--dry_run', action='store_true', help='Print resolved configuration, without data/GPU/training')
    a = p.parse_args(argv)
    a.seq_len = a.seq_len if a.seq_len is not None else 5 * a.pred_len
    if a.model_id is None:
        a.model_id = f'{a.data}_{a.seq_len}_{a.pred_len}'
    for name in ['seq_len', 'pred_len', 'train_epochs', 'batch_size', 'itr', 'inr_layers',
                 'aggregation_layers', 'patch_length', 'hidden_features', 'latent_features',
                 'temporal_latent_features', 'mlp_hidden_layers', 'mlp_hidden_features', 's_cff']:
        if getattr(a, name) <= 0:
            p.error(name + ' must be positive')
    if a.inr_layers < 2 or a.latent_features % 2:
        p.error('inr_layers must be >=2 and latent_features even for the source CFF')
    if (a.seq_len + a.pred_len) % a.patch_length:
        p.error('seq_len + pred_len must be divisible by patch_length')
    if a.num_workers < 0 or a.gpu < 0 or not 0 <= a.seed < 2**32:
        p.error('num_workers/gpu must be nonnegative; seed must fit NumPy uint32')
    for name in ['enc_in', 'c_out', 'max_train_batches', 'max_eval_batches']:
        if getattr(a, name) is not None and getattr(a, name) <= 0:
            p.error(name + ' must be positive')
    for name in ['learning_rate', 'ridge_lambda', 'smooth_l1_beta']:
        value = getattr(a, name)
        if not math.isfinite(value) or value < 0 or (name != 'ridge_lambda' and value == 0):
            p.error(name + ' is invalid')
    if a.val_ratio is None:
        a.val_ratio = .2 if a.protocol == 'paper' and a.data.startswith('ETT') else .1
    if a.test_ratio is None:
        a.test_ratio = .2
    if (not all(math.isfinite(x) and x > 0 for x in [a.val_ratio, a.test_ratio])
            or a.val_ratio + a.test_ratio >= 1):
        p.error('val/test ratios must be positive and sum to less than 1')
    return a


class QuietSourceDataset(PDETimeDataset):
    """Only removes original per-sample printing; all data construction stays upstream."""
    def __getitem__(self, i):
        if i < 0 or i >= len(self):
            raise IndexError(i)
        left, right = i + self.lookback, i + self.lookback + self.horizon
        return self.features[i:left], self.temporal_features[i:right], self.time_index[i:right], self.features[i:right]


def load_data(a):
    default_name = a.data + '.csv' if a.data.startswith('ETT') else a.data.lower() + '.csv'
    path = (Path(a.root_path or './data') / (a.data_path or default_name)).resolve()
    frame = pd.read_csv(path)
    if 'date' not in frame:
        raise ValueError('CSV must have date + numeric target columns')
    dates = pd.DatetimeIndex(pd.to_datetime(frame['date'], errors='raise'))
    if dates.hasnans or not dates.is_monotonic_increasing or not dates.is_unique:
        raise ValueError('date must be valid, strictly chronological and unique')
    columns = [c for c in frame.columns if c != 'date']
    expected, expected_rows = DATASETS[a.data]
    if len(columns) != expected or (a.enc_in is not None and len(columns) != a.enc_in):
        raise ValueError(f'Target columns={len(columns)}, expected {expected}; enc_in={a.enc_in}')
    if a.data.startswith('ETT') and columns != ETT_COLUMNS:
        raise ValueError('ETT column order must be date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT')
    if not np.isfinite(frame[columns].to_numpy(dtype=float)).all():
        raise ValueError('NaN/Inf targets; no automatic filling or target scaling is applied')
    a.enc_in = len(columns)
    if a.c_out is not None and a.c_out != a.enc_in:
        raise ValueError('Source PDETime c_out must equal enc_in')
    a.c_out, a.root_path, a.data_path = a.enc_in, str(path.parent), path.name
    temporal = None
    if a.protocol == 'paper':
        temporal = ['day_of_year', 'month', 'day_of_week', 'hour']
        if a.data in ['ETTm1', 'ETTm2', 'Weather']:
            temporal.append('minute')
    work = frame.copy()
    if temporal is None:
        temporal = PDETimeDataset.choose_temporal_features(work)
    n = len(frame)
    # Exact source split expression, including its integer rounding convention.
    train_end = int(n * (1 - a.val_ratio - a.test_ratio))
    val_end = train_end + int(n * a.val_ratio)
    bounds = {'train': (0, train_end), 'val': (train_end, val_end), 'test': (val_end, n)}
    datasets, manifests = {}, {}
    for name, (start, end) in bounds.items():
        if end - start < a.seq_len + a.pred_len:
            raise ValueError(f'{name}: {end-start} rows cannot fit isolated L={a.seq_len}, H={a.pred_len}; '
                             'choose a shorter --seq_len; context is not silently changed')
        ds = QuietSourceDataset(work.iloc[start:end].copy(), horizon=a.pred_len, lookback=a.seq_len,
                                temporal_feature_names=temporal)
        if ds.features.shape[1] != a.enc_in:
            raise ValueError('Derived calendar columns leaked into targets')
        datasets[name] = ds
        manifests[name] = {'rows': [start, end], 'windows': len(ds),
                           'first_forecast_row': start + a.seq_len, 'forecast_end_exclusive': end,
                           'date_begin': str(dates[start]), 'date_end': str(dates[end - 1]),
                           'time_scaler_min': ds.scaler.data_min_.tolist(),
                           'time_scaler_max': ds.scaler.data_max_.tolist()}
    if n != expected_rows:
        print(f'DATA VERSION NOTE: {a.data} has {n} rows; PDF Table 5 says {expected_rows}; no truncation.', flush=True)
    manifest = {'data_sha256': sha256(path), 'rows': n, 'paper_table5_rows': expected_rows,
                'columns': columns, 'temporal_columns': temporal, 'target_scaling': 'none',
                'context': 'isolated, same as source train.py', 'splits': manifests}
    return datasets, manifest


def model_kwargs(a, temporal_count):
    return dict(dimension=a.enc_in, temporal_features=temporal_count,
                temporal_latent_features=a.temporal_latent_features, lookback=a.seq_len,
                horizon=a.pred_len, s_cff=a.s_cff, hidden_features=a.hidden_features,
                INR_layers=a.inr_layers, aggregation_layers=a.aggregation_layers,
                latent_features=a.latent_features, patch_length=a.patch_length,
                MLP_hidden_layers=a.mlp_hidden_layers, MLP_hidden_features=a.mlp_hidden_features,
                regularization_strength=a.ridge_lambda)


def device_name(a):
    return f'cuda:{a.gpu}' if a.device == 'cuda' else 'cpu'


def build_model(a, temporal_count):
    # Match source allocation device without changing the process-wide DataLoader default.
    with torch.device(device_name(a)):
        model = PDETime(**model_kwargs(a, temporal_count))
    return model  # Do not convert agg_layers to ModuleList or change weight sharing.


def aggregation_layers(model):
    unique, pattern, ids = [], [], {}
    for layer in model.encoder.agg_layers:
        if id(layer) not in ids:
            ids[id(layer)] = len(unique)
            unique.append(layer)
        pattern.append(ids[id(layer)])
    return unique, pattern


def complete_state(model):
    layers, pattern = aggregation_layers(model)
    return {'model': model.state_dict(), 'aggregation': [m.state_dict() for m in layers], 'sharing': pattern}


def restore_state(model, state):
    layers, pattern = aggregation_layers(model)
    if pattern != state['sharing'] or len(layers) != len(state['aggregation']):
        raise ValueError('Aggregation sharing differs from checkpoint')
    model.load_state_dict(state['model'])
    for layer, saved in zip(layers, state['aggregation']):
        layer.load_state_dict(saved)


def source_forecast(outputs, lookback):
    return outputs[:, lookback:] + outputs[:, lookback - 1:lookback]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(ds, a, shuffle):
    return DataLoader(ds, batch_size=a.batch_size, shuffle=shuffle, num_workers=a.num_workers,
                      generator=torch.Generator().manual_seed(a.seed) if shuffle else None,
                      pin_memory=a.device == 'cuda')


def to_device(batch, a):
    return [x.to(device=device_name(a), dtype=torch.float32, non_blocking=True) for x in batch]


@torch.no_grad()
def evaluate(model, ds, criterion, a, collect=False):
    model.eval()
    squared, absolute = np.zeros(a.enc_in), np.zeros(a.enc_in)
    samples, elements_per_channel, loss_sum, example = 0, 0, 0., None
    for i, batch in enumerate(make_loader(ds, a, False)):
        if a.max_eval_batches is not None and i >= a.max_eval_batches:
            break
        x, t, tau, labels = to_device(batch, a)
        outputs = model(x, t, tau)
        prediction = source_forecast(outputs, a.seq_len)
        target = labels[:, a.seq_len:]
        loss_sum += criterion(outputs, labels).item() * len(x)
        delta = prediction - target
        squared += delta.square().sum((0, 1), dtype=torch.float64).cpu().numpy()
        absolute += delta.abs().sum((0, 1), dtype=torch.float64).cpu().numpy()
        elements_per_channel += delta.shape[0] * delta.shape[1]
        samples += len(x)
        if collect and example is None:
            column = a.plot_column or ('OT' if 'OT' in a.target_columns else a.target_columns[0])
            if column not in a.target_columns:
                raise ValueError('Unknown plot_column: ' + column)
            c = a.target_columns.index(column)
            example = (column, labels[0, :, c].cpu().numpy(), prediction[0, :, c].cpu().numpy())
    if not samples:
        raise ValueError('No evaluation windows processed')
    metric = {'loss': loss_sum / samples, 'mse': float(squared.mean() / elements_per_channel),
              'mae': float(absolute.mean() / elements_per_channel), 'scale': 'raw',
              'evaluated_windows': samples, 'num_forecast_values': elements_per_channel * a.enc_in,
              'per_channel': {c: {'mse': float(squared[i] / elements_per_channel),
                                  'mae': float(absolute[i] / elements_per_channel)}
                              for i, c in enumerate(a.target_columns)}}
    if not all(math.isfinite(metric[k]) for k in ['loss', 'mse', 'mae']):
        raise FloatingPointError('Non-finite evaluation metrics')
    return metric, example


def plot_example(example, lookback, path):
    if example is None:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    name, actual, prediction = example
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(actual, label='actual ' + name)
    ax.plot(np.arange(lookback, len(actual)), prediction, label='forecast ' + name)
    ax.axvline(lookback - 1, color='grey', linestyle='--')
    ax.set(xlabel='time step', ylabel=name + ' (raw)', title='Source PDETime test forecast')
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_identity(a, manifest):
    ignored = {'checkpoints', 'checkpoint', 'no_plot', 'plot_column', 'dry_run', 'is_training'}
    identity = {k: v for k, v in vars(a).items() if k not in ignored}
    identity.update(data_sha256=manifest['data_sha256'], source_sha256=source_hashes())
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    slug = re.sub(r'[^A-Za-z0-9_.-]', '_', a.model_id)
    return slug + '_' + digest


def train_once(a, datasets, manifest, directory):
    directory.mkdir(parents=True, exist_ok=False)
    seed_all(a.seed)  # fixed seed for each itr, matching the paper's literal wording
    model = build_model(a, len(manifest['temporal_columns']))
    criterion = PDETimeLoss(nn.SmoothL1Loss(beta=a.smooth_l1_beta), a.seq_len, a.pred_len)
    optimizer = torch.optim.Adam(model.parameters(), lr=a.learning_rate)
    unique, sharing = aggregation_layers(model)
    coverage = {'optimizer_parameters': sum(p.numel() for p in model.parameters()),
                'unregistered_aggregation_parameters': sum(p.numel() for m in unique for p in m.parameters()),
                'aggregation_sharing_pattern': sharing,
                'policy': 'Source optimizer coverage preserved; aggregation is NOT newly registered'}
    save_json(directory / 'config.json', vars(a))
    save_json(directory / 'parameter_sources.json', parameter_sources(a))
    save_json(directory / 'data_manifest.json', manifest)
    save_json(directory / 'provenance.json', {
        'git_sha': git_read('rev-parse', 'HEAD'), 'source_sha256': source_hashes(),
        'entry_sha256': sha256(Path(__file__).resolve()), 'data_sha256': manifest['data_sha256'],
        'tracked_status': git_read('status', '--short', '--untracked-files=no'),
        'command': sys.argv, 'python': sys.version, 'platform': platform.platform(),
        'torch': str(torch.__version__), 'numpy': str(np.__version__), 'pandas': str(pd.__version__),
        'scikit_learn': sklearn_version,
        'cuda_runtime': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(a.gpu) if a.device == 'cuda' else None,
        'model_coverage': coverage})
    train_loader = make_loader(datasets['train'], a, True)
    best_score, best_epoch, history = float('inf'), 0, []
    checkpoint = directory / 'best.pt'
    if a.device == 'cuda':
        torch.cuda.synchronize(a.gpu)
        torch.cuda.reset_peak_memory_stats(a.gpu)
    start = time.perf_counter()
    for epoch in range(1, a.train_epochs + 1):
        model.train()
        loss_sum, batches = 0., 0
        for i, batch in enumerate(train_loader):
            if a.max_train_batches is not None and i >= a.max_train_batches:
                break
            x, t, tau, labels = to_device(batch, a)
            optimizer.zero_grad()
            outputs = model(x, t, tau)
            loss = criterion(outputs, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss')
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            batches += 1
        if not batches:
            raise ValueError('No training batches processed')
        val, _ = evaluate(model, datasets['val'], criterion, a)
        score = val[a.selection_metric]
        row = {'epoch': epoch, 'train_loss': loss_sum / batches,
               'val_loss': val['loss'], 'val_mse': val['mse'], 'val_mae': val['mae']}
        history.append(row)
        with (directory / 'history.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        print(json.dumps({'run': str(directory), **row}), flush=True)
        if score < best_score:
            best_score, best_epoch = score, epoch
            torch.save({'format': FORMAT, 'config': vars(a).copy(), 'state': complete_state(model),
                        'optimizer': optimizer.state_dict(), 'epoch': epoch,
                        'best_score': score, 'source_sha256': source_hashes(),
                        'data_manifest': manifest}, checkpoint)
    if a.device == 'cuda':
        torch.cuda.synchronize(a.gpu)
    summary = {'config': vars(a).copy(), 'partial': bool(a.max_train_batches or a.max_eval_batches),
               'best_epoch': best_epoch, 'best_validation': best_score, 'selection_metric': a.selection_metric,
               'fit_seconds': time.perf_counter() - start,
               'peak_vram_bytes': torch.cuda.max_memory_allocated(a.gpu) if a.device == 'cuda' else None,
               'model_coverage': coverage, 'test': None,
               'status': 'source-code semantics preserved; paper/code differences remain unresolved'}
    if a.evaluate_test:
        state = torch.load(checkpoint, map_location=device_name(a), weights_only=True)
        restore_state(model, state['state'])
        summary['test'], example = evaluate(model, datasets['test'], criterion, a, not a.no_plot)
        if not a.no_plot:
            plot_example(example, a.seq_len, directory / 'prediction.png')
    save_json(directory / 'summary.json', summary)
    return summary


def evaluate_saved(cli):
    if not cli.checkpoint:
        raise ValueError('--is_training 0 requires --checkpoint')
    state = torch.load(cli.checkpoint, map_location='cpu', weights_only=True)
    if state.get('format') != FORMAT:
        raise ValueError('Checkpoint must be produced by this source-preserving entry')
    if state['source_sha256'] != source_hashes():
        raise ValueError('Source model/data code differs from checkpoint; use its recorded revision')
    a = SimpleNamespace(**state['config'])
    a.is_training, a.evaluate_test = 0, 1
    a.checkpoint, a.checkpoints = str(Path(cli.checkpoint).resolve()), cli.checkpoints
    for key in ['device', 'gpu', 'batch_size', 'num_workers', 'max_eval_batches', 'no_plot', 'plot_column']:
        setattr(a, key, getattr(cli, key))
    if cli.root_path is not None:
        a.root_path = cli.root_path
    if cli.data_path is not None:
        a.data_path = cli.data_path
    datasets, manifest = load_data(a)
    if manifest != state['data_manifest']:
        raise ValueError('Evaluation data/preprocessing differs from training checkpoint')
    a.target_columns = manifest['columns']
    seed_all(a.seed)
    model = build_model(a, len(manifest['temporal_columns']))
    restore_state(model, state['state'])
    criterion = PDETimeLoss(nn.SmoothL1Loss(beta=a.smooth_l1_beta), a.seq_len, a.pred_len)
    metric, example = evaluate(model, datasets['test'], criterion, a, not a.no_plot)
    output = Path(cli.checkpoints) / (run_identity(a, manifest) + '_evaluation')
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / 'metrics.json', {'checkpoint': str(Path(cli.checkpoint).resolve()),
              'test': metric, 'config': vars(a), 'partial': bool(a.max_train_batches or a.max_eval_batches)})
    if not a.no_plot:
        plot_example(example, a.seq_len, output / 'prediction.png')
    print(json.dumps({'output': str(output), 'test': metric}), flush=True)


def main(argv=None):
    a = parse_args(argv)
    if a.dry_run:
        print(json.dumps(parameter_sources(a), indent=2, ensure_ascii=False))
        return
    if a.device == 'cuda' and (not torch.cuda.is_available() or a.gpu >= torch.cuda.device_count()):
        raise RuntimeError('Requested CUDA device unavailable; use --device cpu for local smoke tests')
    if not a.is_training:
        evaluate_saved(a)
        return
    datasets, manifest = load_data(a)
    a.target_columns = manifest['columns']
    output = Path(a.checkpoints) / run_identity(a, manifest)
    output.mkdir(parents=True, exist_ok=False)
    reports = []
    for i in range(a.itr):
        a.repeat_id = i
        reports.append(train_once(a, datasets, manifest, output / f'repeat{i}'))
    aggregate = {}
    if a.evaluate_test:
        for key in ['mse', 'mae']:
            values = [r['test'][key] for r in reports]
            aggregate[key] = {'mean': float(np.mean(values)),
                              'std': float(np.std(values, ddof=1)) if len(values) > 1 else 0.}
    save_json(output / 'summary.json', {'runs': reports, 'aggregate': aggregate,
              'seed_policy': 'same fixed seed per itr; not independent seeds',
              'partial': any(r['partial'] for r in reports)})
    print('Saved:', output.resolve(), flush=True)


if __name__ == '__main__':
    main()
