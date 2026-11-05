#!/usr/bin/env python
"""Adendo, partes 2 e 3 (ADR 009): receita de ajuste fino e semana nova.

Treino (padrão): braços novos de ajuste fino nos mesmos dados, seeds e ponto de partida do
escopo A (``run_empirical_models.py``). Cada braço grava os ``best.pt`` e as métricas do
ensemble na validação; ``--select`` junta os braços e registra a escolha (menor MSE macro do
ensemble na validação, escala original) em ``summary.json``.

Semana nova (``--open-holdout``, uma única vez): rotula 5--11/fev/2024 com o MLE e compara,
janela a janela, os braços com o ``lstm_lee/finetune`` congelado. Também avalia outubro/2023
(exploratório). Nada é treinado nesse modo.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_posthoc import paired_comparison  # noqa: E402
from compare_scope_a_finetune import october_windows  # noqa: E402
from run_empirical_models import (  # noqa: E402
    evaluate_variant,
    load_weights,
    predict,
    synthetic_checkpoint,
)

from hawkes_crypto.config import load_experiment_config  # noqa: E402
from hawkes_crypto.empirical import PARAMETER_NAMES, labeled_arrays  # noqa: E402
from hawkes_crypto.eval.metrics import parameter_metrics  # noqa: E402
from hawkes_crypto.training import (  # noqa: E402
    _loader,
    _loss_over_loader,
    build_model,
    selected_device,
    set_deterministic_seed,
    synchronize,
)

ARMS = {
    "finetune_long": {"models": ("lstm_lee",), "schedule": False, "standardize": False},
    "finetune_sched": {"models": ("lstm_lee", "transformer"), "schedule": True,
                       "standardize": False},
    "finetune_std": {"models": ("lstm_lee", "transformer"), "schedule": True,
                     "standardize": True},
}
EPOCHS, WARMUP, PLATEAU, FACTOR, CLIP, VAR_FLOOR = 1000, 5, 10, 0.5, 1.0, 0.01
FROZEN = ("lstm_lee/finetune", "transformer/finetune")
MIX = "mix/lstm+transformer"


def fit_arm(
    model: torch.nn.Module,
    data: dict[str, np.ndarray],
    *,
    learning_rate: float,
    batch_size: int,
    patience: int,
    seed: int,
    device: torch.device,
    checkpoint: Path,
    schedule: bool,
    weights: torch.Tensor,
    epochs: int = EPOCHS,
) -> dict[str, object]:
    """Como ``fit_on_windows``, com teto de 1.000 épocas e, se pedido, aquecimento, redução
    no platô e corte do gradiente. Parada e checkpoint seguem o MSE na escala original."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    validation_loader = _loader(data["val_x"], data["val_y"], batch_size=batch_size,
                                shuffle=False, seed=seed)
    best, best_epoch, stale, plateau = _loss_over_loader(model, validation_loader, device), 0, 0, 0
    torch.save(model.state_dict(), checkpoint)
    history = [{"epoch": 0, "validation_mse": best, "lr": learning_rate}]
    lr = learning_rate
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        if schedule and epoch <= WARMUP:
            lr = learning_rate * epoch / WARMUP
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        for batch_x, batch_y in _loader(data["train_x"], data["train_y"],
                                        batch_size=batch_size, shuffle=True,
                                        seed=seed + epoch):
            error = (model(batch_x.to(device)) - batch_y.to(device)) ** 2
            loss = torch.mean(error * weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if schedule:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()
        synchronize(device)
        validation = _loss_over_loader(model, validation_loader, device)
        history.append({"epoch": epoch, "validation_mse": validation, "lr": lr})
        if validation < best:
            best, best_epoch, stale, plateau = validation, epoch, 0, 0
            torch.save(model.state_dict(), checkpoint)
            continue
        stale += 1
        plateau += 1
        if stale >= patience:
            break
        if schedule and epoch > WARMUP and plateau >= PLATEAU:
            lr, plateau = max(lr * FACTOR, learning_rate / 100), 0
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    return {
        "best_validation_mse": best,
        "best_epoch": best_epoch,
        "epochs_completed": len(history) - 1,
        "train_seconds": time.perf_counter() - started,
        "history": history,
    }


def ensemble(name: str, arm_dir: Path, x: np.ndarray, config, seeds, device) -> np.ndarray:
    outputs = []
    for seed in seeds:
        model = build_model(name, config).to(device)
        load_weights(model, arm_dir / f"seed_{seed}" / "best.pt", device)
        outputs.append(predict(model, x, device))
    return np.mean(outputs, axis=0)


def r_squared(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    residual = np.sum((prediction - target) ** 2, axis=0)
    total = np.sum((target - target.mean(axis=0)) ** 2, axis=0)
    return {n: float(1 - r / t) for n, r, t in zip(PARAMETER_NAMES, residual, total,
                                                       strict=True)}


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    order = sorted(pvalues, key=pvalues.get)
    adjusted, running = {}, 0.0
    for rank, key in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * pvalues[key]))
        adjusted[key] = running
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/lee_local_optimized.yaml"))
    parser.add_argument("--labels", type=Path,
                        default=Path("artifacts/empirical_mle/calibration_validation.json"))
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/um_futures"))
    parser.add_argument("--frozen-dir", type=Path,
                        default=Path("artifacts/empirical_models/scope_a"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/addendum_009"))
    parser.add_argument("--models", nargs="+", default=["lstm_lee", "transformer"])
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--length", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--select", action="store_true")
    parser.add_argument("--open-holdout", action="store_true")
    parser.add_argument("--holdout-root", type=Path,
                        default=Path("data/processed_holdout_009/um_futures"))
    parser.add_argument("--october-root", type=Path,
                        default=Path("data/processed_robustness/um_futures"))
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    seeds = list(config.training.seeds)
    device = selected_device(args.device)
    keys = [f"{m}/{a}" for a in ARMS for m in ARMS[a]["models"]]

    if args.open_holdout:
        summary = json.loads((args.output / "summary.json").read_text())
        if summary.get("holdout_opened"):
            raise SystemExit("Recusado: a semana nova já foi aberta")
        selected = summary["selected_variant"]
        blocks = {"holdout_2024_02": october_windows(args.holdout_root, args.length, config),
                  "october_2023": october_windows(args.october_root, args.length, config)}
        records = json.loads(args.labels.read_text())["records"]
        _, train_y, _ = labeled_arrays(records, args.data_root, split="calibration_train",
                                       length=args.length)
        report = {}
        for block, (x, y) in blocks.items():
            predictions = {"label_mean": np.repeat(train_y.mean(axis=0, keepdims=True),
                                                   len(x), axis=0)}
            for key in FROZEN:
                name, init = key.split("/")
                predictions[key] = ensemble(name, args.frozen_dir / name / init, x, config,
                                            seeds, device)
            for key in keys:
                name, arm = key.split("/")
                predictions[key] = ensemble(name, args.output / name / arm, x, config,
                                            seeds, device)
            predictions[MIX] = np.mean([predictions[k] for k in summary["mix_components"]],
                                       axis=0)
            loss = {k: np.mean((v - y) ** 2, axis=1) for k, v in predictions.items()}
            rng = np.random.default_rng(config.seed)
            primary = paired_comparison(loss[selected], loss["lstm_lee/finetune"], rng)
            secondary = {
                f"{k} vs lstm_lee/finetune": paired_comparison(loss[k],
                                                               loss["lstm_lee/finetune"], rng)
                for k in [*keys, MIX] if k != selected
            }
            for arm in ("finetune_sched", "finetune_std"):
                secondary[f"lstm_lee/{arm} vs transformer/{arm}"] = paired_comparison(
                    loss[f"lstm_lee/{arm}"], loss[f"transformer/{arm}"], rng)
            adjusted = holm({k: v["wilcoxon_p"] for k, v in secondary.items()})
            for k, v in secondary.items():
                v["holm_p"] = adjusted[k]
            report[block] = {
                "windows": int(len(x)),
                "label_quantiles": {n: np.quantile(y[:, i], [0.05, 0.5, 0.95]).tolist()
                                    for i, n in enumerate(PARAMETER_NAMES)},
                "variants": {k: {**parameter_metrics(v, y), "r2": r_squared(v, y)}
                             for k, v in predictions.items()},
                "primary": {"a": selected, "b": "lstm_lee/finetune", **primary},
                "secondary": secondary,
            }
            print(json.dumps({block: {k: round(v["macro_mse"], 4)
                                      for k, v in report[block]["variants"].items()}}),
                  flush=True)
        (args.output / "holdout.json").write_text(json.dumps(report, indent=2),
                                                  encoding="utf-8")
        summary["holdout_opened"] = True
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2),
                                                  encoding="utf-8")
        return

    if args.select:
        records = json.loads(args.labels.read_text())["records"]
        val_x, val_y, _ = labeled_arrays(records, args.data_root, split="validation",
                                         length=args.length)
        variants, val_predictions = {}, {}
        for key in keys:
            name, arm = key.split("/")
            arm_dir = args.output / name / arm
            runs = [json.loads((arm_dir / f"seed_{seed}" / "complete.json").read_text())
                    for seed in seeds]
            predictions = []
            for seed in seeds:
                model = build_model(name, config).to(device)
                load_weights(model, arm_dir / f"seed_{seed}" / "best.pt", device)
                predictions.append(predict(model, val_x, device))
            val_predictions[key] = predictions
            variants[key] = {
                "validation": evaluate_variant(predictions, val_y),
                "runs": [{k: v for k, v in r.items() if k != "history"} for r in runs],
            }
        frozen = json.loads((args.frozen_dir / "summary.json").read_text())["variants"]
        # Mistura (emenda do ADR 009, antes de abrir a semana nova): média do melhor LSTM e do
        # melhor Transformer na validação, cada um escolhido entre o congelado e os braços novos.
        val_predictions = {k: np.mean(v, axis=0) for k, v in val_predictions.items()}
        for key in FROZEN:
            name, init = key.split("/")
            val_predictions[key] = ensemble(name, args.frozen_dir / name / init, val_x, config,
                                            seeds, device)
        pool = {k: frozen[k]["validation"]["ensemble"]["macro_mse"] for k in FROZEN}
        pool.update({k: v["validation"]["ensemble"]["macro_mse"] for k, v in variants.items()})
        components = [min((k for k in pool if k.startswith(name)), key=pool.get)
                      for name in ("lstm_lee/", "transformer/")]
        mix = np.mean([val_predictions[k] for k in components], axis=0)
        variants[MIX] = {"validation": {"ensemble": parameter_metrics(mix, val_y)},
                         "components": components}
        selected = min(variants, key=lambda k: variants[k]["validation"]["ensemble"]["macro_mse"])
        payload = {
            "adr": "009, parte 2",
            "selection_rule": "menor MSE macro do ensemble na validação (escala original)",
            "selected_variant": selected,
            "mix_components": components,
            "holdout_opened": False,
            "validation_macro_mse": {
                **{k: frozen[k]["validation"]["ensemble"]["macro_mse"] for k in FROZEN},
                **{k: v["validation"]["ensemble"]["macro_mse"] for k, v in variants.items()},
            },
            "variants": variants,
        }
        (args.output / "summary.json").write_text(json.dumps(payload, indent=2),
                                                  encoding="utf-8")
        print(json.dumps({"selected": selected, **payload["validation_macro_mse"]}, indent=2))
        return

    # Fila de tarefas: vários processos podem rodar ao mesmo tempo; cada (rede, braço, seed)
    # é reservada por um ``mkdir`` atômico e pulada se já tiver ``complete.json``.
    if device.type == "cpu":
        torch.set_num_threads(args.threads)
    records = json.loads(args.labels.read_text())["records"]
    if any(record["split"] == "test_closed" for record in records):
        raise SystemExit("Recusado: o arquivo de rótulos contém test_closed")
    data: dict[str, np.ndarray] = {}
    data["train_x"], data["train_y"], _ = labeled_arrays(
        records, args.data_root, split="calibration_train", length=args.length)
    data["val_x"], data["val_y"], _ = labeled_arrays(
        records, args.data_root, split="validation", length=args.length)
    variance = np.maximum(data["train_y"].var(axis=0), VAR_FLOOR)
    for seed in seeds:
        for arm in args.arms:
            spec = ARMS[arm]
            weights = torch.ones(4) if not spec["standardize"] else torch.from_numpy(
                (1 / variance).astype(np.float32))
            for name in [m for m in args.models if m in spec["models"]]:
                run_dir = args.output / name / arm / f"seed_{seed}"
                run_dir.parent.mkdir(parents=True, exist_ok=True)
                try:
                    run_dir.mkdir()
                except FileExistsError:
                    continue
                batch_size = (int(config.transformer.get("batch_size",
                                                         config.training.batch_size))
                              if name == "transformer" else config.training.batch_size)
                set_deterministic_seed(seed)
                model = build_model(name, config).to(device)
                load_weights(model, synthetic_checkpoint(config.artifacts_root, name,
                                                         args.length, seed), device)
                run = fit_arm(model, data, learning_rate=config.training.learning_rate / 10,
                              batch_size=batch_size, patience=config.training.patience,
                              seed=seed, device=device, checkpoint=run_dir / "best.pt",
                              schedule=spec["schedule"], weights=weights.to(device),
                              epochs=args.epochs)
                run.update(seed=seed, device=device.type, loss_weights=weights.tolist())
                (run_dir / "complete.json").write_text(json.dumps(run, indent=2),
                                                       encoding="utf-8")
                print(json.dumps({"variant": f"{name}/{arm}", "seed": seed,
                                  "device": device.type,
                                  "best_validation_mse": round(run["best_validation_mse"], 5),
                                  "best_epoch": run["best_epoch"],
                                  "epochs": run["epochs_completed"],
                                  "seconds": round(float(run["train_seconds"]), 1)}),
                      flush=True)

if __name__ == "__main__":
    main()
