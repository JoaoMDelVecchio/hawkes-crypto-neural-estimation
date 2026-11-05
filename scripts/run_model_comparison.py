#!/usr/bin/env python
"""Executa a comparação sintética mínima entre MLE, LSTM e Transformer."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from hawkes_crypto.estimate.mle import fit_mle
from hawkes_crypto.models.attention_encoder import HawkesTransformer
from hawkes_crypto.models.lstm_baseline import LeeLSTM
from hawkes_crypto.simulate.hawkes_sim import HawkesParameters, simulate_path


def make_dataset(
    size: int, sequence_length: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    inputs = np.empty((size, sequence_length, 2), dtype=np.float32)
    targets = np.empty((size, 4), dtype=np.float32)
    for index in range(size):
        mu = rng.uniform(0.2, 0.6)
        alpha_self = rng.uniform(0.1, 0.5)
        alpha_cross = rng.uniform(0.1, 0.5)
        beta = alpha_self + alpha_cross + rng.uniform(0.4, 1.3)
        parameters = HawkesParameters(mu, alpha_self, alpha_cross, beta)
        path = simulate_path(parameters, sequence_length, seed=int(rng.integers(2**32)))
        inputs[index, :, 0] = path.inter_arrivals
        inputs[index, :, 1] = path.event_types
        targets[index] = [mu, alpha_self, alpha_cross, beta]
    return torch.from_numpy(inputs), torch.from_numpy(targets)


def train(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    epochs: int,
    batch_size: int,
) -> float:
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    started = time.perf_counter()
    model.train()
    for _ in range(epochs):
        permutation = torch.randperm(len(inputs))
        for batch in permutation.split(batch_size):
            loss = criterion(model(inputs[batch]), targets[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/model_comparison"))
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--test-size", type=int, default=256)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    train_x, train_y = make_dataset(args.train_size, args.sequence_length, args.seed)
    test_x, test_y = make_dataset(args.test_size, args.sequence_length, args.seed + 1)
    models: dict[str, nn.Module] = {
        "lstm_lee": LeeLSTM(),
        "transformer": HawkesTransformer(d_model=16, nhead=4, num_layers=1),
    }
    results: dict[str, dict[str, float]] = {}
    for name, model in models.items():
        train_seconds = train(model, train_x, train_y, args.epochs, 64)
        model.eval()
        with torch.inference_mode():
            started = time.perf_counter()
            predictions = model(test_x)
            inference_seconds = time.perf_counter() - started
        results[name] = {
            "test_mse": float(nn.functional.mse_loss(predictions, test_y)),
            "train_seconds": train_seconds,
            "inference_ms_per_sequence": inference_seconds * 1000 / len(test_x),
            "valid_predictions": float(
                torch.isfinite(predictions).all(dim=1).logical_and(
                    predictions[:, 3] > predictions[:, 1] + predictions[:, 2]
                ).float().mean()
            ),
        }

    path = simulate_path(HawkesParameters(0.3, 0.4, 0.7, 1.5), 300, seed=args.seed)
    mle_started = time.perf_counter()
    fit = fit_mle(
        path.times,
        path.event_types,
        initial=HawkesParameters(0.3, 0.4, 0.7, 1.5),
        n_starts=2,
    )
    results["mle_fixed_path"] = {
        "log_likelihood": fit.log_likelihood,
        "seconds": time.perf_counter() - mle_started,
        "converged": float(fit.converged),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(results, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
