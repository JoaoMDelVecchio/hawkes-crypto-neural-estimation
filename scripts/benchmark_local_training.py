#!/usr/bin/env python
"""Microbenchmark reproduzível para selecionar dispositivo e batch local."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from hawkes_crypto.config import load_experiment_config
from hawkes_crypto.synthetic import manifest_records
from hawkes_crypto.training import (
    build_model,
    selected_device,
    set_deterministic_seed,
    synchronize,
)


def benchmark(
    *,
    config,
    model_name: str,
    length: int,
    device_name: str,
    batch_size: int,
    steps: int,
    compile_model: bool,
) -> dict[str, object]:
    device = selected_device(device_name)
    record = manifest_records(config.synthetic_root, length, "train")[0]
    with np.load(Path(str(record["path"]))) as shard:
        inputs = torch.from_numpy(shard["inputs"][:batch_size].copy()).to(device)
        targets = torch.from_numpy(shard["targets"][:batch_size].copy()).to(device)
    set_deterministic_seed(config.seed)
    model = build_model(model_name, config).to(device)
    if compile_model:
        model = torch.compile(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    for _ in range(2):
        loss = nn.functional.mse_loss(model(inputs), targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    synchronize(device)
    started = time.perf_counter()
    last_loss = 0.0
    for _ in range(steps):
        loss = nn.functional.mse_loss(model(inputs), targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach())
    synchronize(device)
    seconds = time.perf_counter() - started
    return {
        "model": model_name,
        "length": length,
        "device": device_name,
        "batch_size": batch_size,
        "compiled": compile_model,
        "steps": steps,
        "seconds": seconds,
        "samples_per_second": batch_size * steps / seconds,
        "finite_loss": bool(np.isfinite(last_loss)),
        "last_loss": last_loss,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/lee_local_optimized.yaml")
    )
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/benchmarks/local_training.json")
    )
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    cases = [
        ("lstm_lee", 1000, "cpu", 64),
        ("lstm_lee", 1000, "mps", 64),
        ("transformer", 1000, "mps", 8),
    ]
    results = []
    for model, length, device, batch in cases:
        try:
            result = benchmark(
                config=config,
                model_name=model,
                length=length,
                device_name=device,
                batch_size=batch,
                steps=args.steps,
                compile_model=args.compile,
            )
        except (RuntimeError, NotImplementedError) as error:
            result = {
                "model": model,
                "length": length,
                "device": device,
                "batch_size": batch,
                "compiled": args.compile,
                "error": str(error),
            }
        results.append(result)
        print(json.dumps(result), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
