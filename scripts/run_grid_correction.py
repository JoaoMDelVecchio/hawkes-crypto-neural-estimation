#!/usr/bin/env python
"""ADR 011, parte 1: a rede aprende a desfazer o viés da grade de 100 ms?

Treina a LSTM nas trajetórias sintéticas do escopo B (entrada na grade) com o θ verdadeiro em
tempo contínuo como alvo, e compara num lote novo do mesmo prior: rede, MLE na grade e MLE
corrigido pelo mapa do viés (``grid_bias.json``), todos contra o θ verdadeiro.

``--generate-test`` cria o lote de teste; sem flag, o processo é um worker da fila de seeds;
``--evaluate`` faz as comparações e aplica a rede às janelas reais (exploratório).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_posthoc import paired_comparison  # noqa: E402
from compare_scope_a_finetune import october_windows  # noqa: E402
from run_addendum_finetune import ensemble, fit_arm, holm  # noqa: E402
from run_beta_sweep import gridded_path  # noqa: E402
from run_grid_bias import invert  # noqa: E402
from run_scope_b import claim, forward_maps, sample_prior  # noqa: E402

from hawkes_crypto.config import load_experiment_config  # noqa: E402
from hawkes_crypto.empirical import PARAMETER_NAMES, labeled_arrays  # noqa: E402
from hawkes_crypto.estimate.mle import fit_mle_batch  # noqa: E402
from hawkes_crypto.training import (  # noqa: E402
    build_model,
    selected_device,
    set_deterministic_seed,
)

TEST_SEED, TEST_PATHS = 20261105, 2_000
TRAIN_SPEC = {"lr": 1e-3, "epochs": 150, "patience": 15}
NAME = "lstm_lee"


def generate_test(args, config) -> None:
    rng = np.random.default_rng(TEST_SEED)
    records = json.loads(args.labels.read_text())["records"]
    labels = [r["parameters"] for r in records
              if r["split"] == "calibration_train" and r.get("converged")]
    maps = forward_maps(json.loads(args.grid_bias.read_text()))
    prior = sample_prior(labels, maps, args.paths, rng)
    paths = [gridded_path(p, TEST_SEED + i) for i, p in enumerate(prior)]
    fits = fit_mle_batch(paths, n_starts=config.mle.n_starts, seed=TEST_SEED,
                         maxiter=config.mle.maxiter, workers=config.mle.workers)
    x = np.stack([np.column_stack([np.diff(t, prepend=0.0), s]) for t, s in paths])
    mle = np.asarray([[getattr(f.parameters, n) for n in PARAMETER_NAMES] for f in fits],
                     dtype=np.float64)
    truth = np.asarray([[p.mu, p.alpha_self, p.alpha_cross, p.beta] for p in prior])
    converged = np.asarray([f.converged for f in fits])
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "test.npz", x=x.astype(np.float32), mle=mle,
                        truth=truth, converged=converged)
    print(json.dumps({"paths": len(paths), "converged": int(converged.sum())}))


def corrected_mle(mle: np.ndarray, maps) -> tuple[np.ndarray, np.ndarray]:
    """(μ, β) invertidos pelo mapa; g e fração cruzada do MLE aplicados ao β corrigido."""
    out, inside = np.empty_like(mle), np.empty(len(mle), dtype=bool)
    for k, (mu, alpha_self, alpha_cross, beta) in enumerate(mle):
        mu_c, beta_c, inside[k] = invert(*maps, mu, beta)
        g, total = (alpha_self + alpha_cross) / beta, alpha_self + alpha_cross
        cross = alpha_cross / total if total > 0 else 0.0
        out[k] = (mu_c, g * beta_c * (1 - cross), g * beta_c * cross, beta_c)
    return out, inside


def worker(args, config, seeds, device) -> None:
    if device.type == "cpu":
        torch.set_num_threads(args.threads)
    synthetic = np.load(args.synthetic)
    is_train = synthetic["split"] == "train"
    data = {"train_x": synthetic["x"][is_train], "train_y": synthetic["truth"][is_train],
            "val_x": synthetic["x"][~is_train], "val_y": synthetic["truth"][~is_train]}
    ones = torch.ones(4, device=device)
    for seed in seeds:
        run_dir = args.output / "train" / f"seed_{seed}"
        if not claim(run_dir):
            continue
        set_deterministic_seed(seed)
        model = build_model(NAME, config).to(device)
        run = fit_arm(model, data, learning_rate=TRAIN_SPEC["lr"],
                      batch_size=config.training.batch_size, patience=TRAIN_SPEC["patience"],
                      seed=seed, device=device, checkpoint=run_dir / "best.pt", schedule=False,
                      weights=ones, epochs=TRAIN_SPEC["epochs"])
        run.update(seed=seed, device=device.type, target="continuous-time truth")
        (run_dir / "complete.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
        print(json.dumps({"seed": seed,
                          "best_validation_mse": round(run["best_validation_mse"], 5),
                          "best_epoch": run["best_epoch"],
                          "seconds": round(float(run["train_seconds"]), 1)}), flush=True)


def summarize(estimate: np.ndarray, truth: np.ndarray) -> dict:
    error = estimate - truth
    return {
        "macro_mse": float(np.mean(error**2)),
        "mse": dict(zip(PARAMETER_NAMES, np.mean(error**2, axis=0).tolist(), strict=True)),
        "median_abs_error": dict(zip(PARAMETER_NAMES, np.median(np.abs(error), axis=0).tolist(),
                                     strict=True)),
        "median_relative_error_beta": float(np.median(error[:, 3] / truth[:, 3])),
        "median_abs_relative_error_beta": float(np.median(np.abs(error[:, 3]) / truth[:, 3])),
    }


def evaluate(args, config, seeds, device) -> None:
    test = np.load(args.output / "test.npz")
    ok = test["converged"]
    x, mle, truth = test["x"][ok], test["mle"][ok], test["truth"][ok]
    maps = forward_maps(json.loads(args.grid_bias.read_text()))
    corrected, inside = corrected_mle(mle, maps)
    estimates = {"network": ensemble(NAME, args.output / "train", x, config, seeds, device),
                 "mle_grid": mle, "mle_corrected": corrected}
    loss = {k: np.mean((v - truth) ** 2, axis=1) for k, v in estimates.items()}
    rng = np.random.default_rng(config.seed)
    primary = paired_comparison(loss["network"], loss["mle_corrected"], rng)
    secondary = {
        "network vs mle_grid": paired_comparison(loss["network"], loss["mle_grid"], rng),
        "mle_corrected vs mle_grid": paired_comparison(loss["mle_corrected"], loss["mle_grid"],
                                                       rng),
    }
    adjusted = holm({k: v["wilcoxon_p"] for k, v in secondary.items()})
    for k, v in secondary.items():
        v["holm_p"] = adjusted[k]

    real = {}
    records = json.loads(args.labels.read_text())["records"]
    val_x, val_y, _ = labeled_arrays(records, args.data_root, split="validation",
                                     length=args.length)
    feb_x, feb_y = october_windows(args.feb_root, args.length, config)
    blocks = {"validation": (val_x, val_y), "holdout_2024_02": (feb_x, feb_y)}
    for block, (bx, by) in blocks.items():
        predicted = ensemble(NAME, args.output / "train", bx, config, seeds, device)
        inverted = np.asarray([invert(*maps, m, b)[1] for m, b in zip(by[:, 0], by[:, 3],
                                                                     strict=True)])
        real[block] = {
            "windows": int(len(bx)),
            "beta_grid_mle_quantiles": np.quantile(by[:, 3], [0.05, 0.5, 0.95]).tolist(),
            "beta_network_quantiles": np.quantile(predicted[:, 3], [0.05, 0.5, 0.95]).tolist(),
            "beta_inverted_quantiles": np.quantile(inverted, [0.05, 0.5, 0.95]).tolist(),
            "spearman_beta_network_vs_inverted": float(
                stats.spearmanr(predicted[:, 3], inverted).statistic),
        }
    report = {
        "adr": "011, parte 1",
        "test_paths": int(len(x)), "mle_converged": int(ok.sum()),
        "corrected_inside_map_rate": float(inside.mean()),
        "estimators": {k: summarize(v, truth) for k, v in estimates.items()},
        "primary": {"a": "network", "b": "mle_corrected", **primary,
                    "favourable": primary["bootstrap_95ci"][1] < 0},
        "secondary": secondary,
        "real_windows": real,
        "runs": [{k: v for k, v in json.loads((args.output / "train" / f"seed_{s}"
                                                / "complete.json").read_text()).items()
                  if k != "history"} for s in seeds],
    }
    (args.output / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v["macro_mse"] for k, v in report["estimators"].items()}, indent=2))
    print(json.dumps(report["primary"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/lee_local_optimized.yaml"))
    parser.add_argument("--labels", type=Path,
                        default=Path("artifacts/empirical_mle/calibration_validation.json"))
    parser.add_argument("--grid-bias", type=Path,
                        default=Path("artifacts/addendum_009/grid_bias.json"))
    parser.add_argument("--synthetic", type=Path,
                        default=Path("artifacts/scope_b_010/synthetic.npz"))
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/um_futures"))
    parser.add_argument("--feb-root", type=Path,
                        default=Path("data/processed_holdout_009/um_futures"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/grid_correction_011"))
    parser.add_argument("--length", type=int, default=1000)
    parser.add_argument("--paths", type=int, default=TEST_PATHS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--generate-test", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    seeds = list(config.training.seeds)
    device = selected_device(args.device)
    started = time.perf_counter()
    if args.generate_test:
        generate_test(args, config)
    elif args.evaluate:
        evaluate(args, config, seeds, device)
    else:
        worker(args, config, seeds, device)
    print(f"{time.perf_counter() - started:.0f} s", file=sys.stderr)


if __name__ == "__main__":
    main()
