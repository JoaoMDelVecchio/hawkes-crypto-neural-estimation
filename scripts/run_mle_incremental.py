#!/usr/bin/env python
"""Executa MLE em chunks persistentes e retomáveis."""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np  # noqa: E402

from hawkes_crypto.config import load_experiment_config  # noqa: E402
from hawkes_crypto.estimate.mle import MLETask, fit_mle_task  # noqa: E402
from hawkes_crypto.eval.metrics import (  # noqa: E402
    group_indices_by_branching_ratio,
    parameter_metrics,
)
from hawkes_crypto.synthetic import load_split  # noqa: E402


def load_completed_records(records_path: Path) -> dict[int, dict[str, object]]:
    completed: dict[int, dict[str, object]] = {}
    if not records_path.exists():
        return completed
    for line_number, line in enumerate(
        records_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            index = int(record["index"])
            if index in completed:
                raise ValueError(f"índice duplicado: {index}")
            completed[index] = record
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Registro MLE inválido em {records_path}:{line_number}"
            ) from error
    return completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/lee_local_optimized.yaml")
    )
    parser.add_argument("--length", type=int, choices=[1000], required=True)
    parser.add_argument("--chunk-size", type=int, default=100)
    args = parser.parse_args()
    if args.chunk_size <= 0:
        parser.error("--chunk-size deve ser positivo")
    config = load_experiment_config(args.config)
    output_dir = config.artifacts_root / f"length_{args.length}" / "mle"
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    completed = load_completed_records(records_path)
    inputs, targets = load_split(config.synthetic_root, args.length, "test")
    count = min(len(inputs), config.mle.max_paths)
    invalid_indices = sorted(index for index in completed if not 0 <= index < count)
    if invalid_indices:
        raise ValueError(f"records.jsonl contém índices fora do protocolo: {invalid_indices[:5]}")
    started = time.perf_counter()
    completed_at_start = len(completed)
    executor = (
        ProcessPoolExecutor(max_workers=config.mle.workers)
        if config.mle.workers > 1
        else None
    )
    try:
        for chunk_start in range(0, count, args.chunk_size):
            indices = [
                index
                for index in range(
                    chunk_start, min(chunk_start + args.chunk_size, count)
                )
                if index not in completed
            ]
            if not indices:
                continue
            tasks = [
                MLETask(
                    np.cumsum(inputs[index, :, 0], dtype=np.float64),
                    inputs[index, :, 1].astype(np.int8),
                    config.mle.n_starts,
                    config.seed + index,
                    config.mle.maxiter,
                )
                for index in indices
            ]
            fits = (
                list(executor.map(fit_mle_task, tasks))
                if executor is not None
                else [fit_mle_task(task) for task in tasks]
            )
            with records_path.open("a", encoding="utf-8") as handle:
                for index, fit in zip(indices, fits, strict=True):
                    record = {
                        "index": index,
                        "parameters": [
                            fit.parameters.mu,
                            fit.parameters.alpha_self,
                            fit.parameters.alpha_cross,
                            fit.parameters.beta,
                        ],
                        "converged": fit.converged,
                        "iterations": fit.iterations,
                        "log_likelihood": fit.log_likelihood,
                        "message": fit.message,
                    }
                    completed[index] = record
                    handle.write(json.dumps(record) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            elapsed = time.perf_counter() - started
            completed_now = len(completed) - completed_at_start
            rate = completed_now / elapsed if elapsed else 0.0
            remaining = count - len(completed)
            print(
                json.dumps(
                    {
                        "length": args.length,
                        "completed": len(completed),
                        "total": count,
                        "paths_per_second": rate,
                        "eta_seconds": remaining / rate if rate else None,
                    }
                ),
                flush=True,
            )
    finally:
        if executor is not None:
            executor.shutdown()
    missing = sorted(set(range(count)).difference(completed))
    if missing:
        raise RuntimeError(f"MLE incompleto; índices ausentes: {missing[:10]}")
    ordered = [completed[index] for index in range(count)]
    predictions = np.asarray([record["parameters"] for record in ordered])
    metrics = parameter_metrics(predictions, targets[:count])
    metrics["by_branching_ratio"] = {
        name: parameter_metrics(predictions[indices], targets[:count][indices])
        for name, indices in group_indices_by_branching_ratio(targets[:count]).items()
        if len(indices)
    }
    metrics["convergence_rate"] = float(
        np.mean([bool(record["converged"]) for record in ordered])
    )
    elapsed = time.perf_counter() - started
    processed_this_run = count - completed_at_start
    summary_path = output_dir / "summary.json"
    if processed_this_run:
        metrics["seconds_this_process"] = elapsed
        metrics["seconds_per_path_wall_clock"] = elapsed / processed_this_run
    elif summary_path.exists():
        previous_metrics = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics["seconds_this_process"] = previous_metrics["seconds_this_process"]
        metrics["seconds_per_path_wall_clock"] = previous_metrics[
            "seconds_per_path_wall_clock"
        ]
    else:
        metrics["seconds_this_process"] = 0.0
        metrics["seconds_per_path_wall_clock"] = 0.0
    metrics["processed_paths_this_process"] = processed_this_run
    metrics["median_iterations"] = float(
        np.median([int(record["iterations"]) for record in ordered])
    )
    metrics["workers"] = config.mle.workers
    metrics["n_starts"] = config.mle.n_starts
    summary_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
