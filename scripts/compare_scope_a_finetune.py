#!/usr/bin/env python
"""Comparação pareada pós-teste: LSTM × Transformer com ajuste fino (escopo A).

Análise post hoc, não pré-registrada no ADR 008: a seleção já foi feita na validação
(``lstm_lee/finetune``), e aqui só se mede a incerteza da diferença entre as duas variantes
com ajuste fino, janela a janela, nos mesmos pesos gravados. Usa o bloco de teste já aberto
(etapa 8) e a janela de outubro/2023 (rotulada de novo pelo MLE, como em
``run_robustness_scope_a.py``). Perda por janela: erro quadrático médio nos 4 parâmetros, na
escala original (a mesma do MSE macro). Diferença < 0 favorece a LSTM.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_posthoc import paired_comparison  # noqa: E402
from run_robustness_scope_a import load, predict  # noqa: E402

from hawkes_crypto.config import load_experiment_config  # noqa: E402
from hawkes_crypto.empirical import PARAMETER_NAMES, labeled_arrays  # noqa: E402
from hawkes_crypto.estimate.mle import fit_mle_batch  # noqa: E402
from hawkes_crypto.training import build_model, selected_device  # noqa: E402

VARIANTS = ("lstm_lee", "transformer")


def ensemble(name: str, x: np.ndarray, config, models_dir: Path, seeds, device) -> np.ndarray:
    outputs = []
    for seed in seeds:
        model = build_model(name, config).to(device)
        path = models_dir / name / "finetune" / f"seed_{seed}" / "best.pt"
        outputs.append(predict(load(model, path, device), x, device))
    return np.mean(outputs, axis=0)


def october_windows(data_root: Path, length: int, config) -> tuple[np.ndarray, np.ndarray]:
    inputs, paths = [], []
    for path in sorted(data_root.glob("*/*.parquet")):
        events = pd.read_parquet(path, columns=["inter_arrival", "event_type"])
        for start in range(0, len(events) - length + 1, length):
            array = events.iloc[start : start + length].to_numpy(np.float32)
            inputs.append(array)
            paths.append((np.cumsum(array[:, 0], dtype=np.float64), array[:, 1].astype(np.int8)))
    fits = fit_mle_batch(paths, n_starts=config.mle.n_starts, seed=config.seed,
                         maxiter=config.mle.maxiter, workers=config.mle.workers)
    ok = np.asarray([fit.converged for fit in fits])
    targets = np.asarray([[getattr(f.parameters, n) for n in PARAMETER_NAMES] for f in fits],
                         dtype=np.float32)
    return np.stack(inputs)[ok], targets[ok]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/lee_local_optimized.yaml"))
    parser.add_argument("--labels", type=Path,
                        default=Path("artifacts/empirical_mle/calibration_validation_test.json"))
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/um_futures"))
    parser.add_argument("--october-root", type=Path,
                        default=Path("data/processed_robustness/um_futures"))
    parser.add_argument("--models-dir", type=Path,
                        default=Path("artifacts/empirical_models/scope_a"))
    parser.add_argument("--length", type=int, default=1000)
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/empirical_models/scope_a/paired_finetune.json"))
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    seeds = json.loads((args.models_dir / "summary.json").read_text())["seeds"]
    records = json.loads(args.labels.read_text())["records"]
    device = selected_device()
    torch.manual_seed(config.seed)

    blocks = {}
    test_x, test_y, _ = labeled_arrays(records, args.data_root, split="test_closed",
                                       length=args.length)
    blocks["test_closed"] = (test_x, test_y)
    blocks["october_2023"] = october_windows(args.october_root, args.length, config)

    results = {}
    for block, (x, y) in blocks.items():
        losses = [np.mean((ensemble(n, x, config, args.models_dir, seeds, device) - y) ** 2,
                          axis=1) for n in VARIANTS]
        rng = np.random.default_rng(config.seed)
        results[block] = {"windows": int(len(x)), **paired_comparison(*losses, rng)}
    payload = {
        "post_hoc": True,
        "a": "lstm_lee/finetune (selected)",
        "b": "transformer/finetune",
        "loss": "per-window mean squared error over (mu, alpha_self, alpha_cross, beta)",
        "blocks": results,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
