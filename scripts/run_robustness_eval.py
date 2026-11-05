#!/usr/bin/env python
"""Avalia checkpoint congelado na janela temporal independente de outubro."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hawkes_crypto.config import load_experiment_config
from hawkes_crypto.empirical import PARAMETER_NAMES
from hawkes_crypto.estimate.mle import fit_mle_batch
from hawkes_crypto.eval.metrics import parameter_metrics
from hawkes_crypto.training import build_model, selected_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/local_smoke.yaml"))
    parser.add_argument("--model", choices=["lstm_lee", "transformer"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data-root", type=Path, default=Path("data/processed_robustness/um_futures")
    )
    parser.add_argument("--length", type=int, default=1000)
    parser.add_argument("--max-windows", type=int, default=12)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/robustness/summary.json")
    )
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    inputs, metadata, paths = [], [], []
    symbols = sorted({path.parent.name for path in args.data_root.glob("*/*.parquet")})
    per_symbol_limit = max(1, args.max_windows // max(len(symbols), 1))
    symbol_counts = {symbol: 0 for symbol in symbols}
    for path in sorted(args.data_root.glob("*/*.parquet")):
        symbol = path.parent.name
        if symbol_counts[symbol] >= per_symbol_limit:
            continue
        events = pd.read_parquet(
            path, columns=["inter_arrival", "event_type", "mid_price"]
        )
        for start in range(0, len(events) - args.length + 1, args.length):
            window = events.iloc[start : start + args.length]
            array = window[["inter_arrival", "event_type"]].to_numpy(np.float32)
            inputs.append(array)
            prices = window["mid_price"].to_numpy(np.float64)
            metadata.append(
                {
                    "symbol": symbol,
                    "date": path.stem,
                    "start_event": start,
                    "duration_seconds": float(array[:, 0].sum()),
                    "realized_volatility": float(np.std(np.diff(np.log(prices)))),
                }
            )
            paths.append(
                (
                    np.cumsum(array[:, 0], dtype=np.float64),
                    array[:, 1].astype(np.int8),
                )
            )
            symbol_counts[symbol] += 1
            if symbol_counts[symbol] >= per_symbol_limit:
                break
    if not inputs:
        raise RuntimeError(f"Nenhuma janela encontrada em {args.data_root}")
    fits = fit_mle_batch(
        paths,
        n_starts=config.mle.n_starts,
        seed=config.seed,
        maxiter=config.mle.maxiter,
        workers=config.mle.workers,
    )
    targets = np.asarray(
        [
            [getattr(fit.parameters, name) for name in PARAMETER_NAMES]
            for fit in fits
        ],
        dtype=np.float32,
    )
    device = selected_device()
    model = build_model(args.model, config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    with torch.inference_mode():
        predictions = model(torch.from_numpy(np.stack(inputs)).to(device)).cpu().numpy()
    volatility = np.asarray([item["realized_volatility"] for item in metadata])
    thresholds = np.quantile(volatility, [1 / 3, 2 / 3])
    regimes = np.digitize(volatility, thresholds)
    per_regime = {}
    for regime, label in enumerate(("low", "medium", "high")):
        selected = np.flatnonzero(regimes == regime)
        if len(selected):
            per_regime[label] = parameter_metrics(
                predictions[selected], targets[selected]
            )
    per_symbol = {}
    for symbol in symbols:
        selected = np.asarray(
            [index for index, item in enumerate(metadata) if item["symbol"] == symbol]
        )
        if len(selected):
            per_symbol[symbol] = parameter_metrics(predictions[selected], targets[selected])
    payload = {
        "period": "2023-10-23/2023-10-29",
        "retrained": False,
        "model": args.model,
        "checkpoint": str(args.checkpoint),
        "sequence_length": args.length,
        "metrics": parameter_metrics(predictions, targets),
        "by_symbol": per_symbol,
        "by_volatility_regime": per_regime,
        "mle_convergence_rate": float(np.mean([fit.converged for fit in fits])),
        "windows": metadata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "windows": len(inputs)}, indent=2))


if __name__ == "__main__":
    main()
