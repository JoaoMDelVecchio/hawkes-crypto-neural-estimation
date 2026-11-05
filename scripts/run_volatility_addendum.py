#!/usr/bin/env python
"""ADR 011, parte 2: volatilidade de Hawkes com as redes do adendo (*post hoc*).

Mesma fórmula e mesmo benchmark do critério C4 (``run_hawkes_volatility.py``), nas 711 janelas
da semana de fevereiro/2024 e nas 476 de ``validation``. Comparação principal: Spearman da
mistura LSTM+Transformer menos o da LSTM congelada, em fevereiro, com IC bootstrap.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_addendum_finetune import ensemble  # noqa: E402
from run_hawkes_volatility import hawkes_variance_rate, realized_variance  # noqa: E402

from hawkes_crypto.config import load_experiment_config  # noqa: E402
from hawkes_crypto.empirical import PARAMETER_NAMES, window_events  # noqa: E402
from hawkes_crypto.estimate.mle import fit_mle_batch  # noqa: E402
from hawkes_crypto.training import selected_device  # noqa: E402

STEP, N_BOOT = 10.0, 2000
COLUMNS = ["inter_arrival", "event_type", "mid_price"]


def describe(events: pd.DataFrame) -> dict:
    seconds = events["inter_arrival"].cumsum().to_numpy(np.float64)
    log_price = np.log(events["mid_price"].to_numpy(np.float64))
    marks = np.abs(np.diff(log_price))
    return {"duration": float(seconds[-1] - seconds[0]), "mark_mean": float(marks.mean()),
            "mark_var": float(marks.var()), "rv": realized_variance(seconds, log_price, STEP)}


def february(root: Path, length: int, config) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Mesmas janelas e mesmos rótulos de ``october_windows`` (ADR 009), com preço e marcas."""
    inputs, paths, rows = [], [], []
    for path in sorted(root.glob("*/*.parquet")):
        events = pd.read_parquet(path, columns=COLUMNS)
        for start in range(0, len(events) - length + 1, length):
            window = events.iloc[start : start + length].reset_index(drop=True)
            array = window[["inter_arrival", "event_type"]].to_numpy(np.float32)
            inputs.append(array)
            paths.append((np.cumsum(array[:, 0], dtype=np.float64), array[:, 1].astype(np.int8)))
            rows.append(describe(window))
    fits = fit_mle_batch(paths, n_starts=config.mle.n_starts, seed=config.seed,
                         maxiter=config.mle.maxiter, workers=config.mle.workers)
    ok = np.asarray([fit.converged for fit in fits])
    labels = np.asarray([[getattr(f.parameters, n) for n in PARAMETER_NAMES] for f in fits])
    return np.stack(inputs)[ok], labels[ok], [r for r, keep in zip(rows, ok, strict=True) if keep]


def validation(labels: Path, data_root: Path, length: int):
    records = [r for r in json.loads(labels.read_text())["records"]
               if r["split"] == "validation" and r.get("converged") and int(r["length"]) == length]
    inputs, targets, rows = [], [], []
    for record in records:
        events = window_events(record, data_root, columns=COLUMNS)
        inputs.append(events[["inter_arrival", "event_type"]].to_numpy(np.float32))
        targets.append([float(record["parameters"][n]) for n in PARAMETER_NAMES])
        rows.append(describe(events))
    return np.stack(inputs), np.asarray(targets), rows


def volatility(params: np.ndarray, rows: list[dict]) -> np.ndarray:
    return np.asarray([np.sqrt(hawkes_variance_rate(p, r["mark_mean"], r["mark_var"])
                               * r["duration"]) for p, r in zip(params, rows, strict=True)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/lee_local_optimized.yaml"))
    parser.add_argument("--labels", type=Path,
                        default=Path("artifacts/empirical_mle/calibration_validation.json"))
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/um_futures"))
    parser.add_argument("--feb-root", type=Path,
                        default=Path("data/processed_holdout_009/um_futures"))
    parser.add_argument("--frozen-dir", type=Path,
                        default=Path("artifacts/empirical_models/scope_a"))
    parser.add_argument("--addendum-dir", type=Path, default=Path("artifacts/addendum_009"))
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/addendum_009/volatility_addendum.json"))
    parser.add_argument("--length", type=int, default=1000)
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    seeds = list(config.training.seeds)
    device = selected_device("cpu")
    blocks = {"holdout_2024_02": february(args.feb_root, args.length, config),
              "validation": validation(args.labels, args.data_root, args.length)}
    report = {}
    for block, (x, labels, rows) in blocks.items():
        estimates = {
            "MLE": labels,
            "lstm_lee/finetune": ensemble("lstm_lee", args.frozen_dir / "lstm_lee" / "finetune",
                                          x, config, seeds, device),
            "lstm_lee/finetune_long": ensemble("lstm_lee", args.addendum_dir / "lstm_lee"
                                               / "finetune_long", x, config, seeds, device),
            "transformer/finetune": ensemble("transformer", args.frozen_dir / "transformer"
                                             / "finetune", x, config, seeds, device),
        }
        estimates["mix/lstm+transformer"] = (estimates["lstm_lee/finetune_long"]
                                             + estimates["transformer/finetune"]) / 2
        realized = np.sqrt([r["rv"] for r in rows])
        vols = {k: volatility(v, rows) for k, v in estimates.items()}
        block_report = {"windows": len(rows), "step_seconds": STEP, "estimators": {}}
        for name, vol in vols.items():
            valid = np.isfinite(vol) & (vol > 0)
            block_report["estimators"][name] = {
                "spearman": float(stats.spearmanr(vol[valid], realized[valid]).statistic),
                "median_ratio": float(np.median(vol[valid] / realized[valid])),
                "valid_rate": float(valid.mean()),
            }
        if block == "holdout_2024_02":
            rng = np.random.default_rng(config.seed)
            a, b = vols["mix/lstm+transformer"], vols["lstm_lee/finetune"]
            diff = []
            for _ in range(N_BOOT):
                idx = rng.integers(0, len(realized), len(realized))
                diff.append(stats.spearmanr(a[idx], realized[idx]).statistic
                            - stats.spearmanr(b[idx], realized[idx]).statistic)
            point = (block_report["estimators"]["mix/lstm+transformer"]["spearman"]
                     - block_report["estimators"]["lstm_lee/finetune"]["spearman"])
            block_report["primary"] = {
                "a": "mix/lstm+transformer", "b": "lstm_lee/finetune",
                "spearman_difference": float(point),
                "bootstrap_95ci": np.quantile(diff, [0.025, 0.975]).tolist(),
            }
        report[block] = block_report
        print(json.dumps({block: {k: round(v["spearman"], 3)
                                  for k, v in block_report["estimators"].items()}}), flush=True)
    report["adr"] = "011, parte 2 (post hoc)"
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["holdout_2024_02"]["primary"], indent=2))


if __name__ == "__main__":
    main()
