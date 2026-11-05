#!/usr/bin/env python
"""Escopo empírico A (etapa 4 do roadmap): redes aprendem o MLE em janelas reais.

O alvo de cada janela é a estimativa do MLE (o "rótulo"), não um parâmetro verdadeiro, que
não existe nos dados reais. A pergunta é se a rede reproduz o MLE com custo de inferência
menor. Treina em ``calibration_train`` (15--22/jan/2024), seleciona em ``validation``
(23--25/jan) e só lê ``test_closed`` com ``--open-test`` (etapa 8, uma única vez).

Variantes, cada uma com as cinco seeds do protocolo:

- ``scratch``: rede iniciada do zero;
- ``finetune``: parte do ``best.pt`` sintético da mesma seed, com taxa de aprendizado 10×
  menor;
- referências sem treino: ``zero_shot`` (rede sintética como está) e ``label_mean`` (média
  dos rótulos de treino, o melhor preditor constante).

Com ``--open-test`` nada é treinado: os ``best.pt`` gravados na etapa 4 são carregados e
avaliados também no teste.

A variante escolhida é a de menor MSE macro do ensemble das cinco seeds na validação; a
escolha fica registrada antes de o teste ser aberto.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from hawkes_crypto.config import load_experiment_config
from hawkes_crypto.empirical import labeled_arrays
from hawkes_crypto.eval.metrics import aggregate_seed_metrics, parameter_metrics
from hawkes_crypto.training import (
    _loader,
    _loss_over_loader,
    build_model,
    selected_device,
    set_deterministic_seed,
    synchronize,
)

MODELS = ("lstm_lee", "transformer")
INITS = ("scratch", "finetune")


def predict(model: torch.nn.Module, inputs: np.ndarray, device: torch.device) -> np.ndarray:
    loader = _loader(inputs, np.zeros((len(inputs), 4), np.float32), batch_size=64,
                     shuffle=False, seed=0)
    model.eval()
    outputs = []
    with torch.inference_mode():
        for batch_x, _ in loader:
            outputs.append(model(batch_x.to(device)).cpu().numpy())
    return np.concatenate(outputs)


def synthetic_checkpoint(root: Path, name: str, length: int, seed: int) -> Path:
    return root / f"length_{length}" / name / f"seed_{seed}" / "best.pt"


def load_weights(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload.get("state_dict", payload))


def fit_on_windows(
    model: torch.nn.Module,
    data: dict[str, np.ndarray],
    *,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    patience: int,
    seed: int,
    device: torch.device,
    checkpoint: Path,
) -> dict[str, object]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    validation_loader = _loader(data["val_x"], data["val_y"], batch_size=batch_size,
                                shuffle=False, seed=seed)
    best, best_epoch, stale = _loss_over_loader(model, validation_loader, device), 0, 0
    torch.save(model.state_dict(), checkpoint)
    history = [{"epoch": 0, "validation_mse": best}]
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_x, batch_y in _loader(data["train_x"], data["train_y"],
                                        batch_size=batch_size, shuffle=True,
                                        seed=seed + epoch):
            loss = torch.nn.functional.mse_loss(model(batch_x.to(device)), batch_y.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        synchronize(device)
        validation = _loss_over_loader(model, validation_loader, device)
        history.append({"epoch": epoch, "validation_mse": validation})
        if validation < best:
            best, best_epoch, stale = validation, epoch, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    return {
        "best_validation_mse": best,
        "best_epoch": best_epoch,
        "epochs_completed": len(history) - 1,
        "train_seconds": time.perf_counter() - started,
        "history": history,
    }


def evaluate_variant(
    predictions: list[np.ndarray], targets: np.ndarray
) -> dict[str, object]:
    per_seed = [parameter_metrics(values, targets) for values in predictions]
    result: dict[str, object] = {
        "ensemble": parameter_metrics(np.mean(predictions, axis=0), targets)
    }
    if len(per_seed) > 1:
        result["seeds"] = aggregate_seed_metrics(per_seed)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mle-summary",
        type=Path,
        default=Path("artifacts/empirical_mle/calibration_validation.json"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/um_futures"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/empirical_models/scope_a"))
    parser.add_argument("--config", type=Path, default=Path("configs/lee_local_optimized.yaml"))
    parser.add_argument("--length", type=int, default=1000)
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    parser.add_argument("--inits", nargs="+", default=list(INITS), choices=INITS)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--open-test", action="store_true")
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    seeds = args.seeds or list(config.training.seeds)
    epochs = args.epochs or config.training.epochs
    payload = json.loads(args.mle_summary.read_text())
    records = payload["records"] if isinstance(payload, dict) else payload
    if any(record["split"] == "test_closed" for record in records) and not args.open_test:
        raise SystemExit("Recusado: o arquivo de rótulos contém test_closed; use --open-test")

    data: dict[str, np.ndarray] = {}
    data["train_x"], data["train_y"], train_records = labeled_arrays(
        records, args.data_root, split="calibration_train", length=args.length
    )
    data["val_x"], data["val_y"], val_records = labeled_arrays(
        records, args.data_root, split="validation", length=args.length
    )
    splits = {"validation": (data["val_x"], data["val_y"])}
    windows = {"calibration_train": len(train_records), "validation": len(val_records)}
    if args.open_test:
        test_x, test_y, test_records = labeled_arrays(
            records, args.data_root, split="test_closed", length=args.length
        )
        splits["test"] = (test_x, test_y)
        windows["test_closed"] = len(test_records)

    device = selected_device()
    args.output.mkdir(parents=True, exist_ok=True)
    label_mean = data["train_y"].mean(axis=0, keepdims=True)
    variants: dict[str, dict[str, object]] = {
        "label_mean": {
            split: evaluate_variant([np.repeat(label_mean, len(y), axis=0)], y)
            for split, (_, y) in splits.items()
        }
    }
    runs: dict[str, list[dict[str, object]]] = {}
    for name in args.models:
        batch_size = (
            int(config.transformer.get("batch_size", config.training.batch_size))
            if name == "transformer"
            else config.training.batch_size
        )
        zero_shot: dict[str, list[np.ndarray]] = {split: [] for split in splits}
        for seed in seeds:
            model = build_model(name, config).to(device)
            load_weights(model, synthetic_checkpoint(config.artifacts_root, name,
                                                     args.length, seed), device)
            for split, (x, _) in splits.items():
                zero_shot[split].append(predict(model, x, device))
        variants[f"{name}/zero_shot"] = {
            split: evaluate_variant(zero_shot[split], splits[split][1]) for split in splits
        }
        for init in args.inits:
            key = f"{name}/{init}"
            predictions: dict[str, list[np.ndarray]] = {split: [] for split in splits}
            runs[key] = []
            for seed in seeds:
                set_deterministic_seed(seed)
                model = build_model(name, config).to(device)
                learning_rate = config.training.learning_rate
                if init == "finetune":
                    load_weights(model, synthetic_checkpoint(config.artifacts_root, name,
                                                             args.length, seed), device)
                    learning_rate /= 10
                run_dir = args.output / name / init / f"seed_{seed}"
                if args.open_test:
                    # Etapa 8: avalia os pesos congelados na etapa 4, sem retreinar.
                    load_weights(model, run_dir / "best.pt", device)
                    for split, (x, _) in splits.items():
                        predictions[split].append(predict(model, x, device))
                    continue
                run_dir.mkdir(parents=True, exist_ok=True)
                run = fit_on_windows(
                    model,
                    data,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    epochs=epochs,
                    patience=config.training.patience,
                    seed=seed,
                    device=device,
                    checkpoint=run_dir / "best.pt",
                )
                run["seed"] = seed
                runs[key].append(run)
                (run_dir / "complete.json").write_text(json.dumps(run, indent=2),
                                                       encoding="utf-8")
                for split, (x, _) in splits.items():
                    predictions[split].append(predict(model, x, device))
                print(json.dumps({"variant": key, "seed": seed,
                                  "best_validation_mse": run["best_validation_mse"],
                                  "best_epoch": run["best_epoch"],
                                  "seconds": round(float(run["train_seconds"]), 1)}),
                      flush=True)
            variants[key] = {
                split: evaluate_variant(predictions[split], splits[split][1])
                for split in splits
            }

    trained = [key for key in variants if key.split("/")[-1] in INITS]
    selected = min(
        trained, key=lambda key: variants[key]["validation"]["ensemble"]["macro_mse"]
    ) if trained else None
    if args.open_test:
        frozen = json.loads((args.output / "summary.json").read_text())["selected_variant"]
        if frozen != selected:
            raise SystemExit(f"Seleção mudou: {frozen} na etapa 4, {selected} agora")
    results = {
        "question": "a rede reproduz o MLE em janelas reais?",
        "split_policy": {
            "trained_on": "calibration_train",
            "selected_on": "validation",
            "test_opened": args.open_test,
        },
        "windows": windows,
        "seeds": seeds,
        "selected_variant": selected,
        "selection_rule": "menor MSE macro do ensemble na validação",
        "variants": variants,
        "runs": runs,
    }
    name = "summary_test.json" if args.open_test else "summary.json"
    (args.output / name).write_text(json.dumps(results, indent=2), encoding="utf-8")
    table = {
        key: {split: round(float(value[split]["ensemble"]["macro_mse"]), 5) for split in value}
        for key, value in variants.items()
    }
    print(json.dumps({"selected": selected, "ensemble_macro_mse": table}, indent=2))


if __name__ == "__main__":
    main()
