#!/usr/bin/env python
"""Viés da grade de 100 ms no MLE de β (adendo, *post hoc*, depois do teste de β do ADR 009).

O teste de β mostrou que levar a trajetória à grade (um evento por célula) puxa o β do MLE
para baixo. Aqui o mapa "parâmetro verdadeiro → MLE na grade" é medido numa malha de (μ, β)
com g = 0,5 e α_c ≈ 0, como nas janelas reais, e invertido por inferência indireta: para cada
rótulo real (μ̂, β̂) procura-se o (μ, β) em tempo contínuo cuja mediana do MLE na grade
coincide com ele (interpolação bilinear em log). Usa só ``calibration_train``, ``validation``
e a mediana de outubro; o ``test_closed`` não é lido.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_beta_sweep import gridded_path  # noqa: E402

from hawkes_crypto.config import load_experiment_config  # noqa: E402
from hawkes_crypto.estimate.mle import fit_mle_batch  # noqa: E402
from hawkes_crypto.simulate.hawkes_sim import HawkesParameters  # noqa: E402

MUS = (0.05, 0.1, 0.23, 0.5, 1.0, 2.0)
BETAS = (1.2, 2.0, 2.7, 3.5, 4.5, 6.0, 8.0, 10.0, 13.0)
BRANCHING, ALPHA_CROSS, SEED = 0.50, 0.001, 20261103


def invert(forward_mu, forward_beta, mu_hat: float, beta_hat: float) -> tuple[float, float, bool]:
    """(μ, β) contínuos cuja mediana do MLE na grade é (μ̂, β̂); busca numa malha fina."""
    log_mu = np.linspace(np.log(MUS[0]), np.log(MUS[-1]), 240)
    log_beta = np.linspace(np.log(BETAS[0]), np.log(BETAS[-1]), 240)
    mesh = np.stack(np.meshgrid(log_mu, log_beta, indexing="ij"), axis=-1).reshape(-1, 2)
    error = ((forward_mu(mesh) - np.log(mu_hat)) ** 2
             + (forward_beta(mesh) - np.log(beta_hat)) ** 2)
    best = int(np.argmin(error))
    inside = bool(np.sqrt(error[best]) < 0.05)
    return float(np.exp(mesh[best, 0])), float(np.exp(mesh[best, 1])), inside


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/lee_local_optimized.yaml"))
    parser.add_argument("--labels", type=Path,
                        default=Path("artifacts/empirical_mle/calibration_validation.json"))
    parser.add_argument("--october", type=Path,
                        default=Path("artifacts/robustness/scope_a_october.json"))
    parser.add_argument("--paths", type=int, default=100)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/addendum_009/grid_bias.json"))
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    started = time.perf_counter()

    cells, paths = [], []
    for i, mu in enumerate(MUS):
        for j, beta in enumerate(BETAS):
            truth = HawkesParameters(mu, BRANCHING * beta - ALPHA_CROSS, ALPHA_CROSS, beta)
            for k in range(args.paths):
                cells.append((i, j))
                paths.append(gridded_path(truth, SEED + 100_000 * i + 1000 * j + k))
    fits = fit_mle_batch(paths, n_starts=config.mle.n_starts, seed=SEED,
                         maxiter=config.mle.maxiter, workers=args.workers or config.mle.workers)
    estimates = np.full((len(MUS), len(BETAS), args.paths, 3), np.nan)
    counters = np.zeros((len(MUS), len(BETAS)), dtype=int)
    for (i, j), fit in zip(cells, fits, strict=True):
        if fit.converged:
            p = fit.parameters
            estimates[i, j, counters[i, j]] = (p.mu, p.beta,
                                               (p.alpha_self + p.alpha_cross) / p.beta)
            counters[i, j] += 1
    median = np.nanmedian(estimates, axis=2)
    grid = (np.log(MUS), np.log(BETAS))
    forward_mu = RegularGridInterpolator(grid, np.log(median[..., 0]))
    forward_beta = RegularGridInterpolator(grid, np.log(median[..., 1]))

    records = json.loads(args.labels.read_text())["records"]
    blocks = {}
    for split in ("calibration_train", "validation"):
        labels = [r["parameters"] for r in records if r["split"] == split and r.get("converged")]
        inverted = [invert(forward_mu, forward_beta, r["mu"], r["beta"]) for r in labels]
        beta_hat = np.asarray([r["beta"] for r in labels])
        beta_cont = np.asarray([b for _, b, _ in inverted])
        mu_cont = np.asarray([m for m, _, _ in inverted])
        blocks[split] = {
            "windows": len(labels),
            "inside_map_rate": float(np.mean([ok for *_, ok in inverted])),
            "beta_grid_mle_quantiles": np.quantile(beta_hat, [0.05, 0.5, 0.95]).tolist(),
            "beta_continuous_quantiles": np.quantile(beta_cont, [0.05, 0.5, 0.95]).tolist(),
            "mu_continuous_quantiles": np.quantile(mu_cont, [0.05, 0.5, 0.95]).tolist(),
            "median_ratio_grid_to_continuous": float(np.median(beta_hat / beta_cont)),
        }
    october = json.loads(args.october.read_text())["label_quantiles"]
    mu_c, beta_c, ok = invert(forward_mu, forward_beta, october["mu"][1], october["beta"][1])
    blocks["october_2023_median"] = {"beta_grid_mle": october["beta"][1], "mu_grid_mle":
                                     october["mu"][1], "beta_continuous": beta_c,
                                     "mu_continuous": mu_c, "inside_map": ok}
    payload = {
        "post_hoc": True,
        "design": {"mu": MUS, "beta": BETAS, "branching_ratio": BRANCHING,
                   "alpha_cross": ALPHA_CROSS, "paths_per_cell": args.paths,
                   "grid_seconds": 0.1, "events": 1000},
        "forward_map_median": {
            f"mu={mu}": {f"beta={beta}": {"mu_hat": float(median[i, j, 0]),
                                          "beta_hat": float(median[i, j, 1]),
                                          "g_hat": float(median[i, j, 2]),
                                          "converged": int(counters[i, j])}
                         for j, beta in enumerate(BETAS)}
            for i, mu in enumerate(MUS)},
        "inversion": blocks,
        "seconds": time.perf_counter() - started,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(blocks, indent=2))


if __name__ == "__main__":
    main()
