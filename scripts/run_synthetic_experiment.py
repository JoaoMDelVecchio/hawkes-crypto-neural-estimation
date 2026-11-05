#!/usr/bin/env python
"""Treina e avalia LSTM, Transformer e MLE no protocolo congelado."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from hawkes_crypto.config import load_experiment_config
from hawkes_crypto.estimate.mle import fit_mle_batch
from hawkes_crypto.eval.metrics import (
    aggregate_seed_metrics,
    group_indices_by_branching_ratio,
    parameter_metrics,
)
from hawkes_crypto.synthetic import load_split
from hawkes_crypto.training import evaluate_model, execution_metadata, train_model


def run_mle(config, length: int) -> dict[str, object]:
    inputs, targets = load_split(config.synthetic_root, length, "test")
    count = min(len(inputs), config.mle.max_paths)
    paths = [
        (
            np.cumsum(inputs[index, :, 0], dtype=np.float64),
            inputs[index, :, 1].astype(np.int8),
        )
        for index in range(count)
    ]
    started = time.perf_counter()
    fits = fit_mle_batch(
        paths,
        n_starts=config.mle.n_starts,
        seed=config.seed,
        maxiter=config.mle.maxiter,
        workers=config.mle.workers,
    )
    predictions = np.asarray(
        [
            [
                fit.parameters.mu,
                fit.parameters.alpha_self,
                fit.parameters.alpha_cross,
                fit.parameters.beta,
            ]
            for fit in fits
        ]
    )
    result = parameter_metrics(predictions, targets[:count])
    result["seconds"] = time.perf_counter() - started
    result["convergence_rate"] = float(np.mean([fit.converged for fit in fits]))
    result["median_iterations"] = float(np.median([fit.iterations for fit in fits]))
    result["initialization"] = "empirical_rate_heuristic_with_random_multistart"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/local_smoke.yaml"))
    parser.add_argument("--skip-mle", action="store_true")
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    config.artifacts_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "execution": execution_metadata(config),
        "lengths": {},
    }
    for length in config.data.sequence_lengths:
        length_result: dict[str, object] = {"models": {}}
        _, test_targets = load_split(config.synthetic_root, length, "test")
        groups = group_indices_by_branching_ratio(test_targets)
        for model_name in ("lstm_lee", "transformer"):
            seed_results = []
            for seed in config.training.seeds:
                model, training, checkpoint = train_model(
                    model_name, config, sequence_length=length, seed=seed
                )
                evaluation = evaluate_model(
                    model,
                    config,
                    sequence_length=length,
                    batch_size=int(training["batch_size"]),
                )
                evaluation["training"] = training
                evaluation["checkpoint"] = str(checkpoint)
                seed_results.append(evaluation)
            aggregate = aggregate_seed_metrics(seed_results)
            aggregate["runs"] = seed_results
            aggregate["test_g_strata_samples"] = {
                name: int(len(indices)) for name, indices in groups.items()
            }
            length_result["models"][model_name] = aggregate
        if config.mle.enabled and not args.skip_mle:
            length_result["mle"] = run_mle(config, length)
        summary["lengths"][str(length)] = length_result
        (config.artifacts_root / "summary.partial.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
    output = config.artifacts_root / "summary.json"
    output.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"output": str(output), "lengths": list(summary["lengths"])}, indent=2))


if __name__ == "__main__":
    main()
