#!/usr/bin/env python3
"""Reproducible ETTm1 train/validation/test runner for the PDETime fork.

Keeps the source model and loss, but registers Encoder.agg_layers so its
parameters are trainable and checkpointable. This is an intentional fix,
not a byte-for-byte reproduction of the original train.py behavior.
"""

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.PDETime.models.pdetime_model import PDETime, PDETimeLoss
from src.PDETime.preprocess.dataloader import PDETimeDataset


class QuietPDETimeDataset(PDETimeDataset):
    """Same indexing as upstream without one print() per sample."""

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Invalid dataset index: {idx}")
        left = idx + self.lookback
        right = left + self.horizon
        return (
            self.features[idx:left],
            self.temporal_features[idx:right],
            self.time_index[idx:right],
            self.features[idx:right],
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/ETTm1.csv")
    parser.add_argument("--run-dir", default="runs/a5000_ettm1_h96_seed2026")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--lookback", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=None,
                        help="Only for smoke tests; changes training")
    parser.add_argument("--max-eval-batches", type=int, default=None,
                        help="Only for smoke tests; yields partial metrics")
    args = parser.parse_args()
    args.lookback = args.lookback if args.lookback is not None else args.horizon * 5
    if args.horizon <= 0 or args.lookback <= 0 or args.batch_size <= 0 or args.epochs <= 0:
        parser.error("horizon, lookback, batch-size and epochs must all be positive")
    if (args.lookback + args.horizon) % 32 != 0:
        parser.error("lookback + horizon must be divisible by patch_length=32")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available; check the NVIDIA driver/PyTorch wheel")
    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(features, temporal_features, lookback, horizon, device):
    model = PDETime(
        dimension=features,
        temporal_features=temporal_features,
        temporal_latent_features=32,
        lookback=lookback,
        horizon=horizon,
        s_cff=1,
        hidden_features=64,
        INR_layers=5,
        aggregation_layers=1,
        latent_features=64,
        patch_length=32,
        MLP_hidden_layers=3,
        MLP_hidden_features=64,
    )
    # Critical: upstream stores these in a plain Python list, so its optimizer,
    # .to(device), and state_dict() otherwise do not include the aggregation layers.
    model.encoder.agg_layers = nn.ModuleList(model.encoder.agg_layers)
    return model.to(device)


def make_loader(dataset, batch_size, shuffle, seed, num_workers, device):
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, generator=generator,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )


def on_device(batch, device):
    return [tensor.to(device=device, dtype=torch.float32, non_blocking=True)
            for tensor in batch]


def forecast(model, batch, device, lookback):
    x, t, tau, labels = on_device(batch, device)
    outputs = model(x, t, tau)
    # The original train.py adds the last reconstructed value to future residuals.
    prediction = outputs[:, lookback:, :] + outputs[:, lookback - 1:lookback, :]
    target = labels[:, lookback:, :]
    return outputs, labels, prediction, target


@torch.no_grad()
def evaluate(model, loader, device, lookback, max_batches=None, collect_example=False):
    model.eval()
    squared_error = 0.0
    absolute_error = 0.0
    n = 0
    example = None
    for idx, batch in enumerate(loader):
        if max_batches is not None and idx >= max_batches:
            break
        _, labels, prediction, target = forecast(model, batch, device, lookback)
        delta = prediction - target
        squared_error += delta.square().sum().item()
        absolute_error += delta.abs().sum().item()
        n += delta.numel()
        if collect_example and example is None:
            example = (labels[0, :, -1].detach().cpu().numpy(),
                       prediction[0, :, -1].detach().cpu().numpy())
    if n == 0:
        raise ValueError("Empty evaluation split; reduce lookback/horizon")
    return {"mse": squared_error / n, "mae": absolute_error / n,
            "num_forecast_values": n}, example


def save_plot(example, lookback, destination):
    actual, prediction = example
    steps = np.arange(len(actual))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(steps, actual, label="actual OT", linewidth=1.3)
    ax.plot(steps[lookback:], prediction, label="forecast OT", linewidth=1.3)
    ax.axvline(lookback - 1, linestyle="--", color="grey", label="forecast start")
    ax.set(xlabel="time step", ylabel="OT (raw scale)", title="PDETime ETTm1 test example")
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data)
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset missing: {data_path}. See docs/experiments/01_a5000_reproduction.md")
    frame = pd.read_csv(data_path)
    if "date" not in frame or "OT" not in frame:
        raise ValueError("ETTm1.csv must have date and OT columns")
    train_ds, test_ds, val_ds = QuietPDETimeDataset.train_test_val_split(
        frame, horizon=args.horizon, lookback=args.lookback,
        val_ratio=0.1, test_ratio=0.2,
    )
    for name, ds in (("train", train_ds), ("validation", val_ds), ("test", test_ds)):
        if len(ds) <= 0:
            raise ValueError(f"{name} split has no windows; reduce lookback/horizon")
    feature_count = train_ds.features.shape[1]
    temporal_count = train_ds.temporal_features.shape[1]
    print(f"device={device}; data_rows={len(frame)}; spatial_features={feature_count}; "
          f"temporal_features={temporal_count}", flush=True)
    print(f"windows: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}", flush=True)
    train_loader = make_loader(train_ds, args.batch_size, True, args.seed, args.num_workers, device)
    val_loader = make_loader(val_ds, args.batch_size, False, args.seed, args.num_workers, device)
    test_loader = make_loader(test_ds, args.batch_size, False, args.seed, args.num_workers, device)
    model = make_model(feature_count, temporal_count, args.lookback, args.horizon, device)
    criterion = PDETimeLoss(nn.SmoothL1Loss(), args.lookback, args.horizon)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val_mse = float("inf")
    best_epoch = 0
    history = []
    checkpoint_path = output_dir / "best.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        train_batches = 0
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", mininterval=5)):
            if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
                break
            x, t, tau, labels = on_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(x, t, tau)
            loss = criterion(outputs, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss: try lower --lr, check input values")
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            train_batches += 1
        if train_batches == 0:
            raise ValueError("No training batches processed")
        val_metric, _ = evaluate(model, val_loader, device, args.lookback, args.max_eval_batches)
        entry = {"epoch": epoch, "train_loss": loss_sum / train_batches,
                 "val_mse": val_metric["mse"], "val_mae": val_metric["mae"]}
        history.append(entry)
        print(json.dumps(entry, ensure_ascii=False), flush=True)
        if val_metric["mse"] < best_val_mse:
            best_val_mse = val_metric["mse"]
            best_epoch = epoch
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "args": vars(args), "spatial_features": feature_count,
                        "temporal_features": temporal_count,
                        "best_val_mse": best_val_mse}, checkpoint_path)
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("epoch", "train_loss", "val_mse", "val_mae"))
        writer.writeheader()
        writer.writerows(history)
    # PyTorch >=2.5 supports safe loading with weights_only=True.
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    test_metric, example = evaluate(model, test_loader, device, args.lookback,
                                    args.max_eval_batches, collect_example=True)
    save_plot(example, args.lookback, output_dir / "prediction_ot.png")
    summary = {"config": vars(args), "data_rows": len(frame),
               "train_windows": len(train_ds), "val_windows": len(val_ds),
               "test_windows": len(test_ds), "spatial_features": feature_count,
               "temporal_features": temporal_count, "best_epoch": best_epoch,
               "best_val_mse": best_val_mse, "test": test_metric,
               "note": "Adjusted pipeline: aggregation layers registered via ModuleList; "
                       "forecast MSE/MAE on raw future values. Not directly paper-comparable."}
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Best epoch={best_epoch}; test MSE={test_metric['mse']:.6f}; "
          f"test MAE={test_metric['mae']:.6f}", flush=True)
    print(f"Saved results in {output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
