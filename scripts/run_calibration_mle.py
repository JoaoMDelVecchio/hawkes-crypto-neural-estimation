#!/usr/bin/env python
"""Rotula janelas empíricas por MLE sem abrir o teste inadvertidamente."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from hawkes_crypto.config import load_experiment_config
from hawkes_crypto.empirical import PARAMETER_NAMES, read_window_index, window_events
from hawkes_crypto.estimate.mle import fit_mle_batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/local_smoke.yaml"))
    parser.add_argument("--index", type=Path, default=Path("data/processed/window_index.jsonl"))
    parser.add_argument(
        "--data-root", type=Path, default=Path("data/processed/um_futures")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/empirical_mle/calibration.json")
    )
    parser.add_argument("--splits", nargs="+", default=["calibration_train"])
    parser.add_argument("--length", type=int, default=1000)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--open-test", action="store_true")
    args = parser.parse_args()
    if "test_closed" in args.splits and not args.open_test:
        raise SystemExit("Recusado: use --open-test para avaliar test_closed explicitamente")
    config = load_experiment_config(args.config)
    selected = [
        record
        for record in read_window_index(args.index)
        if record["split"] in args.splits and int(record["length"]) == args.length
    ]
    if args.max_windows > 0:
        selected = [
            record
            for split in args.splits
            for record in [item for item in selected if item["split"] == split][
                : args.max_windows
            ]
        ]
    paths = []
    for record in selected:
        events = window_events(
            record, args.data_root, columns=["inter_arrival", "event_type"]
        )
        paths.append(
            (
                events["inter_arrival"].cumsum().to_numpy(np.float64),
                events["event_type"].to_numpy(np.int8),
            )
        )
    started = time.perf_counter()
    fits = fit_mle_batch(
        paths,
        n_starts=config.mle.n_starts,
        seed=config.seed,
        maxiter=config.mle.maxiter,
        workers=config.mle.workers,
    )
    records = []
    for source, fit in zip(selected, fits, strict=True):
        records.append(
            {
                **source,
                "converged": fit.converged,
                "iterations": fit.iterations,
                "log_likelihood": fit.log_likelihood,
                "message": fit.message,
                "parameters": {
                    name: value
                    for name, value in zip(
                        PARAMETER_NAMES,
                        (
                            fit.parameters.mu,
                            fit.parameters.alpha_self,
                            fit.parameters.alpha_cross,
                            fit.parameters.beta,
                        ),
                        strict=True,
                    )
                },
            }
        )
    converged = [record for record in records if record["converged"]]
    ranges = {}
    if converged:
        for name in PARAMETER_NAMES:
            values = np.asarray([record["parameters"][name] for record in converged])
            ranges[name] = {
                "q05": float(np.quantile(values, 0.05)),
                "median": float(np.median(values)),
                "q95": float(np.quantile(values, 0.95)),
            }
    payload = {
        "splits": args.splits,
        "test_explicitly_opened": args.open_test,
        "length": args.length,
        "seconds": time.perf_counter() - started,
        "convergence_rate": len(converged) / len(records) if records else 0.0,
        "calibration_quantiles": ranges,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "fits": len(records)}, indent=2))


if __name__ == "__main__":
    main()
