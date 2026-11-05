#!/usr/bin/env python
"""Escopo A na janela de robustez (23--29/out/2023), sem retreino (etapa 8).

Corta cada dia em janelas de 1.000 eventos sem sobreposição, rotula-as com o MLE (5 inícios)
e avalia, contra esse rótulo, todas as variantes do ``summary.json`` do escopo A com os pesos
gravados na etapa 4: redes sintéticas sem treino, redes treinadas do zero e com ajuste fino
(ensemble das 5 seeds) e a média dos rótulos de treino de janeiro. Nada é ajustado em outubro.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hawkes_crypto.config import load_experiment_config
from hawkes_crypto.empirical import PARAMETER_NAMES, labeled_arrays
from hawkes_crypto.estimate.mle import fit_mle_batch
from hawkes_crypto.eval.metrics import parameter_metrics
from hawkes_crypto.training import build_model, selected_device


def predict(model: torch.nn.Module, inputs: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        return np.concatenate([
            model(torch.from_numpy(inputs[i : i + 64]).to(device)).cpu().numpy()
            for i in range(0, len(inputs), 64)
        ])


def load(model: torch.nn.Module, path: Path, device: torch.device) -> torch.nn.Module:
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload.get("state_dict", payload))
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/lee_local_optimized.yaml"))
    parser.add_argument("--data-root", type=Path,
                        default=Path("data/processed_robustness/um_futures"))
    parser.add_argument("--models-dir", type=Path,
                        default=Path("artifacts/empirical_models/scope_a"))
    parser.add_argument("--labels", type=Path,
                        default=Path("artifacts/empirical_mle/calibration_validation.json"))
    parser.add_argument("--train-data-root", type=Path,
                        default=Path("data/processed/um_futures"))
    parser.add_argument("--length", type=int, default=1000)
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/robustness/scope_a_october.json"))
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    frozen = json.loads((args.models_dir / "summary.json").read_text())
    seeds = frozen["seeds"]

    inputs, metadata, paths = [], [], []
    for path in sorted(args.data_root.glob("*/*.parquet")):
        events = pd.read_parquet(path, columns=["inter_arrival", "event_type", "mid_price"])
        for start in range(0, len(events) - args.length + 1, args.length):
            window = events.iloc[start : start + args.length]
            array = window[["inter_arrival", "event_type"]].to_numpy(np.float32)
            prices = window["mid_price"].to_numpy(np.float64)
            inputs.append(array)
            paths.append((np.cumsum(array[:, 0], dtype=np.float64),
                          array[:, 1].astype(np.int8)))
            metadata.append({
                "symbol": path.parent.name, "date": path.stem, "start_event": start,
                "realized_volatility": float(np.std(np.diff(np.log(prices)))),
            })
    started = time.perf_counter()
    fits = fit_mle_batch(paths, n_starts=config.mle.n_starts, seed=config.seed,
                         maxiter=config.mle.maxiter, workers=config.mle.workers)
    mle_seconds = time.perf_counter() - started
    converged = np.asarray([fit.converged for fit in fits])
    targets = np.asarray([[getattr(fit.parameters, n) for n in PARAMETER_NAMES] for fit in fits],
                         dtype=np.float32)[converged]
    x = np.stack(inputs)[converged]
    meta = [m for m, ok in zip(metadata, converged, strict=True) if ok]

    records = json.loads(args.labels.read_text())["records"]
    _, train_y, _ = labeled_arrays(records, args.train_data_root, split="calibration_train",
                                   length=args.length)
    predictions = {"label_mean": np.repeat(train_y.mean(axis=0, keepdims=True), len(x), axis=0)}
    device = selected_device()
    for key in frozen["variants"]:
        if key == "label_mean":
            continue
        name, init = key.split("/")
        outputs = []
        for seed in seeds:
            model = build_model(name, config).to(device)
            weights = (config.artifacts_root / f"length_{args.length}" / name / f"seed_{seed}"
                       / "best.pt" if init == "zero_shot"
                       else args.models_dir / name / init / f"seed_{seed}" / "best.pt")
            outputs.append(predict(load(model, weights, device), x, device))
        predictions[key] = np.mean(outputs, axis=0)

    symbols = sorted({m["symbol"] for m in meta})
    volatility = np.asarray([m["realized_volatility"] for m in meta])
    regimes = np.digitize(volatility, np.quantile(volatility, [1 / 3, 2 / 3]))
    results = {}
    for key, values in predictions.items():
        results[key] = {
            "all": parameter_metrics(values, targets),
            "by_symbol": {
                s: parameter_metrics(values[idx], targets[idx])
                for s in symbols
                if len(idx := np.flatnonzero([m["symbol"] == s for m in meta]))
            },
            "by_volatility_tercile": {
                label: parameter_metrics(values[regimes == r], targets[regimes == r])
                for r, label in enumerate(("low", "medium", "high"))
            },
        }
    payload = {
        "period": "2023-10-23/2023-10-29",
        "retrained": False,
        "selected_variant": frozen["selected_variant"],
        "windows": int(len(x)),
        "mle_convergence_rate": float(converged.mean()),
        "mle_seconds": mle_seconds,
        "label_quantiles": {
            n: [float(np.quantile(targets[:, i], q)) for q in (0.05, 0.5, 0.95)]
            for i, n in enumerate(PARAMETER_NAMES)
        },
        "variants": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: round(v["all"]["macro_mse"], 5) for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
