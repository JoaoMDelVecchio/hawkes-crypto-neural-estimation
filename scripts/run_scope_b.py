#!/usr/bin/env python
"""Escopo B reduzido (ADR 010): pré-treino sintético na região dos dados e na grade de 100 ms.

``--generate`` sorteia o prior (mistura de janelas de ``calibration_train`` levadas ao tempo
contínuo pelo mapa do viés da grade e de um prior largo), simula, leva à grade, corta em 1.000
eventos e rotula cada trajetória com o MLE. Sem flag, o processo é um worker de uma fila
(pré-treino e depois ajuste fino da LSTM, por seed; reserva por ``mkdir`` atômico).
``--select`` grava as métricas de validação; ``--open-holdout`` lê 4--10/mar/2024 uma vez.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_posthoc import paired_comparison  # noqa: E402
from compare_scope_a_finetune import october_windows  # noqa: E402
from run_addendum_finetune import ensemble, fit_arm, holm, r_squared  # noqa: E402
from run_beta_sweep import gridded_path  # noqa: E402
from run_empirical_models import evaluate_variant, load_weights, predict  # noqa: E402
from run_grid_bias import invert  # noqa: E402

from hawkes_crypto.config import load_experiment_config  # noqa: E402
from hawkes_crypto.empirical import PARAMETER_NAMES, labeled_arrays  # noqa: E402
from hawkes_crypto.estimate.mle import fit_mle_batch  # noqa: E402
from hawkes_crypto.eval.metrics import parameter_metrics  # noqa: E402
from hawkes_crypto.simulate.hawkes_sim import HawkesParameters  # noqa: E402
from hawkes_crypto.training import (  # noqa: E402
    build_model,
    selected_device,
    set_deterministic_seed,
)

SEED, TRAIN, VALIDATION, MAX_BETA = 20261104, 20_000, 2_000, 20.0
EMPIRICAL_SHARE, JITTER, ALPHA_FLOOR = 0.75, 0.2, 1e-4
PRETRAIN = {"lr": 1e-3, "epochs": 150, "patience": 15}
FINETUNE = {"lr": 1e-4, "epochs": 1000, "patience": 30}
NAME = "lstm_lee"


def forward_maps(bias: dict) -> tuple[RegularGridInterpolator, RegularGridInterpolator]:
    design = bias["design"]
    grid = (np.log(design["mu"]), np.log(design["beta"]))
    rows = [list(row.values()) for row in bias["forward_map_median"].values()]
    mu = np.log([[cell["mu_hat"] for cell in row] for row in rows])
    beta = np.log([[cell["beta_hat"] for cell in row] for row in rows])
    return RegularGridInterpolator(grid, mu), RegularGridInterpolator(grid, beta)


def sample_prior(labels: list[dict], maps, n: int, rng: np.random.Generator) -> list:
    """Parâmetros em tempo contínuo; ver o ADR 010 para a mistura."""
    continuous = [invert(*maps, r["mu"], r["beta"])[:2] for r in labels]
    out = []
    for _ in range(n):
        if rng.random() < EMPIRICAL_SHARE:
            k = int(rng.integers(len(labels)))
            r, (mu, beta) = labels[k], continuous[k]
            mu *= np.exp(rng.normal(0, JITTER))
            beta *= np.exp(rng.normal(0, JITTER))
            total = r["alpha_self"] + r["alpha_cross"]
            g = float(np.clip(total / r["beta"] * rng.uniform(1.0, 1.25), 0.05, 0.95))
            cross = r["alpha_cross"] / total
        else:
            mu = float(np.exp(rng.uniform(np.log(0.05), np.log(2.0))))
            beta = float(np.exp(rng.uniform(np.log(1.2), np.log(13.0))))
            g = float(rng.uniform(0.2, 0.9))
            cross = 0.0 if rng.random() < 0.5 else float(rng.uniform(0.0, 0.5))
        alpha_cross = max(g * beta * cross, ALPHA_FLOOR)
        out.append(HawkesParameters(float(mu), g * beta - alpha_cross, alpha_cross, float(beta)))
    return out


def generate(args, config) -> None:
    rng = np.random.default_rng(SEED)
    records = json.loads(args.labels.read_text())["records"]
    labels = [r["parameters"] for r in records
              if r["split"] == "calibration_train" and r.get("converged")]
    prior = sample_prior(labels, forward_maps(json.loads(args.grid_bias.read_text())),
                         args.train + args.validation, rng)
    started = time.perf_counter()
    paths = [gridded_path(p, SEED + i) for i, p in enumerate(prior)]
    fits = fit_mle_batch(paths, n_starts=config.mle.n_starts, seed=SEED,
                         maxiter=config.mle.maxiter, workers=config.mle.workers)
    y = np.asarray([[getattr(f.parameters, n) for n in PARAMETER_NAMES] for f in fits],
                   dtype=np.float32)
    keep = np.asarray([f.converged for f in fits]) & (y[:, 3] <= MAX_BETA)
    x = np.stack([np.column_stack([np.diff(t, prepend=0.0), s]) for t, s in paths])
    truth = np.asarray([[p.mu, p.alpha_self, p.alpha_cross, p.beta] for p in prior],
                       dtype=np.float32)
    split = np.array(["train"] * args.train + ["validation"] * args.validation)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "synthetic.npz", x=x[keep].astype(np.float32),
                        y=y[keep], truth=truth[keep], split=split[keep])
    real = np.asarray([[r[n] for n in PARAMETER_NAMES] for r in labels])
    low, high = np.quantile(y[keep], [0.01, 0.99], axis=0)
    report = {
        "paths": len(paths), "kept": int(keep.sum()),
        "dropped_not_converged": int((~np.asarray([f.converged for f in fits])).sum()),
        "dropped_beta_above_max": int((y[:, 3] > MAX_BETA).sum()),
        "synthetic_label_quantiles": {n: np.quantile(y[keep][:, i], [0.05, 0.5, 0.95]).tolist()
                                      for i, n in enumerate(PARAMETER_NAMES)},
        "real_label_quantiles": {n: np.quantile(real[:, i], [0.05, 0.5, 0.95]).tolist()
                                 for i, n in enumerate(PARAMETER_NAMES)},
        "real_inside_synthetic_1_99": {n: float(np.mean((real[:, i] >= low[i])
                                                        & (real[:, i] <= high[i])))
                                       for i, n in enumerate(PARAMETER_NAMES)},
        "seconds": time.perf_counter() - started,
    }
    (args.output / "generate.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def real_data(args) -> dict[str, np.ndarray]:
    records = json.loads(args.labels.read_text())["records"]
    if any(r["split"] == "test_closed" for r in records):
        raise SystemExit("Recusado: o arquivo de rótulos contém test_closed")
    data = {}
    data["train_x"], data["train_y"], _ = labeled_arrays(
        records, args.data_root, split="calibration_train", length=args.length)
    data["val_x"], data["val_y"], _ = labeled_arrays(
        records, args.data_root, split="validation", length=args.length)
    return data


def claim(run_dir: Path) -> bool:
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        run_dir.mkdir()
    except FileExistsError:
        return False
    return True


def worker(args, config, seeds, device) -> None:
    if device.type == "cpu":
        torch.set_num_threads(args.threads)
    synthetic = np.load(args.output / "synthetic.npz")
    is_train = synthetic["split"] == "train"
    pre_data = {"train_x": synthetic["x"][is_train], "train_y": synthetic["y"][is_train],
                "val_x": synthetic["x"][~is_train], "val_y": synthetic["y"][~is_train]}
    fine_data = real_data(args)
    ones = torch.ones(4, device=device)
    stage_specs = (("pretrain", pre_data, PRETRAIN), ("finetune", fine_data, FINETUNE))
    while True:
        pending = False
        for seed in seeds:
            for stage, data, spec in stage_specs:
                run_dir = args.output / stage / f"seed_{seed}"
                if (run_dir / "complete.json").exists():
                    continue
                pending = True
                start = args.output / "pretrain" / f"seed_{seed}"
                if stage == "finetune" and not (start / "complete.json").exists():
                    continue
                if not claim(run_dir):
                    continue
                set_deterministic_seed(seed)
                model = build_model(NAME, config).to(device)
                if stage == "finetune":
                    load_weights(model, start / "best.pt", device)
                run = fit_arm(model, data, learning_rate=spec["lr"],
                              batch_size=config.training.batch_size,
                              patience=spec["patience"], seed=seed, device=device,
                              checkpoint=run_dir / "best.pt", schedule=False, weights=ones,
                              epochs=spec["epochs"])
                run.update(seed=seed, device=device.type, stage=stage)
                (run_dir / "complete.json").write_text(json.dumps(run, indent=2),
                                                       encoding="utf-8")
                print(json.dumps({"stage": stage, "seed": seed,
                                  "best_validation_mse": round(run["best_validation_mse"], 5),
                                  "best_epoch": run["best_epoch"],
                                  "seconds": round(float(run["train_seconds"]), 1)}),
                      flush=True)
        if not pending:
            return
        time.sleep(30)


def candidates(args, config, seeds, device, x: np.ndarray) -> dict[str, np.ndarray]:
    stage = {"B/zero": "pretrain", "B/finetune": "finetune"}
    out = {k: ensemble(NAME, args.output / s, x, config, seeds, device) for k, s in stage.items()}
    out["transformer/finetune"] = ensemble("transformer", args.frozen_dir / "transformer"
                                           / "finetune", x, config, seeds, device)
    out["mix/B+transformer"] = (out["B/finetune"] + out["transformer/finetune"]) / 2
    return out


def select(args, config, seeds, device) -> None:
    data = real_data(args)
    variants = {}
    for key, value in candidates(args, config, seeds, device, data["val_x"]).items():
        variants[key] = {"validation": parameter_metrics(value, data["val_y"])}
    for stage in ("pretrain", "finetune"):
        predictions = []
        for seed in seeds:
            model = build_model(NAME, config).to(device)
            load_weights(model, args.output / stage / f"seed_{seed}" / "best.pt", device)
            predictions.append(predict(model, data["val_x"], device))
        key = "B/zero" if stage == "pretrain" else "B/finetune"
        variants[key]["validation_seeds"] = evaluate_variant(predictions, data["val_y"])
        variants[key]["runs"] = [
            {k: v for k, v in json.loads((args.output / stage / f"seed_{s}" / "complete.json")
                                         .read_text()).items() if k != "history"}
            for s in seeds]
    summary = {"adr": "010", "holdout_opened": False, "variants": variants,
               "validation_macro_mse": {k: v["validation"]["macro_mse"]
                                        for k, v in variants.items()}}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["validation_macro_mse"], indent=2))


def open_holdout(args, config, seeds, device) -> None:
    summary = json.loads((args.output / "summary.json").read_text())
    if summary.get("holdout_opened"):
        raise SystemExit("Recusado: a semana de teste do ADR 010 já foi aberta")
    train_y = real_data(args)["train_y"]
    blocks = {"holdout_2024_03": args.holdout_root, "holdout_2024_02": args.feb_root,
              "october_2023": args.october_root}
    report = {}
    for block, root in blocks.items():
        x, y = october_windows(root, args.length, config)
        predictions = {"label_mean": np.repeat(train_y.mean(axis=0, keepdims=True), len(x),
                                               axis=0)}
        predictions["lstm_lee/finetune"] = ensemble(NAME, args.frozen_dir / NAME / "finetune",
                                                    x, config, seeds, device)
        predictions["lstm_lee/finetune_long"] = ensemble(
            NAME, args.addendum_dir / NAME / "finetune_long", x, config, seeds, device)
        predictions.update(candidates(args, config, seeds, device, x))
        predictions["mix/lstm+transformer"] = (predictions["lstm_lee/finetune_long"]
                                               + predictions["transformer/finetune"]) / 2
        loss = {k: np.mean((v - y) ** 2, axis=1) for k, v in predictions.items()}
        rng = np.random.default_rng(config.seed)
        primary = paired_comparison(loss["B/finetune"], loss["lstm_lee/finetune_long"], rng)
        pairs = (("mix/B+transformer", "mix/lstm+transformer"), ("B/zero", "label_mean"),
                 ("B/finetune", "lstm_lee/finetune"))
        secondary = {f"{a} vs {b}": paired_comparison(loss[a], loss[b], rng) for a, b in pairs}
        adjusted = holm({k: v["wilcoxon_p"] for k, v in secondary.items()})
        for k, v in secondary.items():
            v["holm_p"] = adjusted[k]
        report[block] = {
            "windows": int(len(x)),
            "variants": {k: {**parameter_metrics(v, y), "r2": r_squared(v, y)}
                         for k, v in predictions.items()},
            "primary": {"a": "B/finetune", "b": "lstm_lee/finetune_long", **primary,
                        "favourable": primary["bootstrap_95ci"][1] < 0},
            "secondary": secondary,
        }
        print(json.dumps({block: {k: round(v["macro_mse"], 4)
                                  for k, v in report[block]["variants"].items()}}), flush=True)
    (args.output / "holdout.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary["holdout_opened"] = True
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/lee_local_optimized.yaml"))
    parser.add_argument("--labels", type=Path,
                        default=Path("artifacts/empirical_mle/calibration_validation.json"))
    parser.add_argument("--grid-bias", type=Path,
                        default=Path("artifacts/addendum_009/grid_bias.json"))
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/um_futures"))
    parser.add_argument("--frozen-dir", type=Path,
                        default=Path("artifacts/empirical_models/scope_a"))
    parser.add_argument("--addendum-dir", type=Path, default=Path("artifacts/addendum_009"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/scope_b_010"))
    parser.add_argument("--holdout-root", type=Path,
                        default=Path("data/processed_holdout_010/um_futures"))
    parser.add_argument("--feb-root", type=Path,
                        default=Path("data/processed_holdout_009/um_futures"))
    parser.add_argument("--october-root", type=Path,
                        default=Path("data/processed_robustness/um_futures"))
    parser.add_argument("--length", type=int, default=1000)
    parser.add_argument("--train", type=int, default=TRAIN)
    parser.add_argument("--validation", type=int, default=VALIDATION)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--select", action="store_true")
    parser.add_argument("--open-holdout", action="store_true")
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    seeds = list(config.training.seeds)
    device = selected_device(args.device)
    if args.generate:
        generate(args, config)
    elif args.select:
        select(args, config, seeds, device)
    elif args.open_holdout:
        open_holdout(args, config, seeds, device)
    else:
        worker(args, config, seeds, device)


if __name__ == "__main__":
    main()
