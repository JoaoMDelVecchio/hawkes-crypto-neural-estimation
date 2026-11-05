#!/usr/bin/env python
"""Adendo, parte 1 (ADR 009): erro das redes com ajuste fino quando β se afasta de janeiro.

Simula o Hawkes com μ, g e α_c fixos no centro dos rótulos de ``calibration_train`` e β em
uma grade, leva cada trajetória à grade de 100 ms como no pré-processamento e corta os
primeiros 1.000 eventos. O alvo principal é o MLE da trajetória (o que as redes aprenderam a
imitar); o secundário, o θ verdadeiro. As redes são os pesos congelados de
``lstm_lee/finetune`` e ``transformer/finetune`` (ensemble das 5 seeds), sem retreino.

H-β: Δ = d̄(4,5) − d̄(2,7) > 0, com d = perda_LSTM − perda_Transformer por trajetória; IC
bootstrap 95% independente nos dois β.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_scope_a_finetune import VARIANTS, ensemble  # noqa: E402
from run_real_data_diagnostics import to_grid  # noqa: E402

from hawkes_crypto.config import load_experiment_config  # noqa: E402
from hawkes_crypto.empirical import PARAMETER_NAMES  # noqa: E402
from hawkes_crypto.estimate.mle import fit_mle_batch  # noqa: E402
from hawkes_crypto.simulate.hawkes_sim import HawkesParameters, simulate_path  # noqa: E402
from hawkes_crypto.training import selected_device  # noqa: E402

BETAS = (1.2, 2.0, 2.7, 3.5, 4.5)
MU, BRANCHING, ALPHA_CROSS = 0.23, 0.50, 0.001
SEED, LENGTH, N_BOOT = 20261102, 1000, 2000


def gridded_path(parameters: HawkesParameters, seed: int) -> tuple[np.ndarray, np.ndarray]:
    n_events = 2 * LENGTH
    while True:
        path = simulate_path(parameters, n_events, seed=seed)
        times, types = to_grid(path.times, path.event_types)
        if len(times) >= LENGTH:
            return times[:LENGTH], types[:LENGTH]
        n_events *= 2


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    idx = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    return values[idx].mean(axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/lee_local_optimized.yaml"))
    parser.add_argument("--models-dir", type=Path,
                        default=Path("artifacts/empirical_models/scope_a"))
    parser.add_argument("--paths", type=int, default=400)
    parser.add_argument("--betas", nargs="+", type=float, default=list(BETAS))
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/addendum_009/beta_sweep.json"))
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    seeds = json.loads((args.models_dir / "summary.json").read_text())["seeds"]
    device = selected_device()
    rng = np.random.default_rng(SEED)

    results, losses = {}, {}
    started = time.perf_counter()
    for beta in args.betas:
        i = BETAS.index(beta) if beta in BETAS else int(10 * beta)
        truth = HawkesParameters(MU, BRANCHING * beta - ALPHA_CROSS, ALPHA_CROSS, beta)
        paths = [gridded_path(truth, SEED + 1000 * i + j) for j in range(args.paths)]
        fits = fit_mle_batch(paths, n_starts=config.mle.n_starts, seed=SEED + 1000 * i,
                             maxiter=config.mle.maxiter, workers=config.mle.workers)
        ok = np.asarray([fit.converged for fit in fits])
        labels = np.asarray([[getattr(f.parameters, n) for n in PARAMETER_NAMES] for f in fits],
                            dtype=np.float64)[ok]
        x = np.stack([np.column_stack([np.diff(t, prepend=0.0), s]) for t, s in paths])
        x = x.astype(np.float32)[ok]
        true = np.repeat([[MU, truth.alpha_self, ALPHA_CROSS, beta]], len(x), axis=0)
        block = {"windows": int(ok.sum()), "mle_convergence_rate": float(ok.mean()),
                 "mle_median": dict(zip(PARAMETER_NAMES, np.median(labels, axis=0).tolist(),
                                        strict=True))}
        per_window = {}
        for name in VARIANTS:
            prediction = ensemble(name, x, config, args.models_dir, seeds, device)
            per_window[name] = np.mean((prediction - labels) ** 2, axis=1)
            block[name] = {
                "mse_vs_mle": float(per_window[name].mean()),
                "mse_vs_truth": float(np.mean((prediction - true) ** 2)),
                "per_parameter_mse_vs_mle": dict(zip(
                    PARAMETER_NAMES, np.mean((prediction - labels) ** 2, axis=0).tolist(),
                    strict=True)),
            }
        d = per_window["lstm_lee"] - per_window["transformer"]
        losses[beta] = d
        boot = bootstrap_mean(d, rng)
        block["d_lstm_minus_transformer"] = {
            "mean": float(d.mean()),
            "bootstrap_95ci": [float(np.quantile(boot, 0.025)),
                               float(np.quantile(boot, 0.975))],
        }
        results[str(beta)] = block
        print(json.dumps({"beta": beta, "windows": block["windows"],
                          **{n: round(block[n]["mse_vs_mle"], 4) for n in VARIANTS}}),
              flush=True)

    h_beta = None
    if 2.7 in losses and 4.5 in losses:
        delta = bootstrap_mean(losses[4.5], rng) - bootstrap_mean(losses[2.7], rng)
        ci = [float(np.quantile(delta, 0.025)), float(np.quantile(delta, 0.975))]
        h_beta = {"delta": float(losses[4.5].mean() - losses[2.7].mean()),
                  "bootstrap_95ci": ci, "supported": ci[0] > 0}
    payload = {
        "adr": "009, parte 1",
        "fixed": {"mu": MU, "branching_ratio": BRANCHING, "alpha_cross": ALPHA_CROSS},
        "loss": "per-window mean squared error over (mu, alpha_self, alpha_cross, beta)",
        "betas": results,
        "h_beta": h_beta,
        "seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["h_beta"], indent=2))


if __name__ == "__main__":
    main()
