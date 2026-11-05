#!/usr/bin/env python
"""Ajusta MLE em uma amostra estratificada do dataset primário."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from hawkes_crypto.estimate.mle import fit_mle
from hawkes_crypto.simulate.hawkes_sim import HawkesParameters


def split_for(day: str) -> str:
    if day <= "2024-01-22":
        return "calibration_train"
    if day <= "2024-01-25":
        return "validation"
    return "test_closed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/processed/um_futures"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/empirical_mle/summary.json"))
    parser.add_argument("--max-per-split", type=int, default=2)
    args = parser.parse_args()
    records: list[dict[str, object]] = []
    for path in sorted(args.root.glob("*/*.parquet")):
        day, symbol = path.stem, path.parent.name
        split = split_for(day)
        selected = sum(
            r["split"] == split and r["symbol"] == symbol for r in records
        )
        if selected >= args.max_per_split:
            continue
        events = pd.read_parquet(path, columns=["inter_arrival", "event_type"])
        length = 1000
        if len(events) < length:
            continue
        window = events.iloc[:length]
        times = window["inter_arrival"].cumsum().to_numpy()
        types = window["event_type"].to_numpy(dtype="int8")
        started = time.perf_counter()
        try:
            fit = fit_mle(
                times,
                types,
                initial=HawkesParameters(0.3, 0.4, 0.7, 1.5),
                n_starts=2,
                seed=20260831,
                maxiter=300,
            )
            records.append(
                {
                    "symbol": symbol,
                    "date": day,
                    "split": split,
                    "length": length,
                    "seconds": time.perf_counter() - started,
                    "converged": fit.converged,
                    "log_likelihood": fit.log_likelihood,
                    "parameters": fit.parameters.__dict__,
                }
            )
        except (ValueError, FloatingPointError) as error:
            records.append(
                {
                    "symbol": symbol,
                    "date": day,
                    "split": split,
                    "length": length,
                    "status": "error",
                    "error": str(error),
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps({"fits": len(records), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
