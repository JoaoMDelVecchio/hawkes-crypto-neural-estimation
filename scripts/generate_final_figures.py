#!/usr/bin/env python
"""Gera figuras finais reproduzíveis a partir do resumo consolidado.

``--lang pt`` (padrão) escreve ``<nome>.pdf``; ``--lang en`` escreve ``<nome>_en.pdf`` no
mesmo diretório, com títulos e rótulos em inglês.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PARAMETERS = ("mu", "alpha_self", "alpha_cross", "beta")
PARAMETER_LABELS = (r"$\mu$", r"$\alpha_{\mathrm{self}}$", r"$\alpha_{\mathrm{cross}}$", r"$\beta$")
REGIMES = ("low", "intermediate", "near_critical")
LANGUAGES = ("pt", "en")
TEXT = {
    "pt": {
        "regimes": ("Baixo", "Intermediário", "Quase crítico"),
        "parameter_ylabel": "Erro quadrático médio (MSE, escala log)",
        "parameter_xlabel": "Parâmetro Hawkes",
        "parameter_title": "MSE por parâmetro no teste sintético de 1.000 eventos",
        "regime_ylabel": "MSE macro (escala log)",
        "regime_xlabel": "Regime da razão de ramificação",
        "regime_title": "Erro por regime de estabilidade",
        "cost_ylabel": "Tempo por trajetória (ms, escala log)",
        "cost_xlabel": "Estimador",
        "cost_title": "Custo de inferência no hardware local",
        "suffix": "",
    },
    "en": {
        "regimes": ("Low", "Intermediate", "Near-critical"),
        "parameter_ylabel": "Mean squared error (MSE, log scale)",
        "parameter_xlabel": "Hawkes parameter",
        "parameter_title": "MSE by parameter on the 1,000-event synthetic test set",
        "regime_ylabel": "Macro MSE (log scale)",
        "regime_xlabel": "Branching-ratio regime",
        "regime_title": "Error by stability regime",
        "cost_ylabel": "Time per path (ms, log scale)",
        "cost_xlabel": "Estimator",
        "cost_title": "Inference cost on the local hardware",
        "suffix": "_en",
    },
}
REGIME_LABELS = TEXT["pt"]["regimes"]
MODEL_SPECS = (
    ("lstm_lee", "LSTM"),
    ("transformer", "Transformer"),
)


def _save_figure(figure: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def generate_figures(
    summary: dict[str, object], output_dir: Path, lang: str = "pt"
) -> list[Path]:
    if lang not in LANGUAGES:
        raise ValueError(f"Língua não suportada: {lang}")
    text = TEXT[lang]
    suffix = text["suffix"]
    lengths = summary["lengths"]
    if set(lengths) != {"1000"}:
        raise ValueError("As figuras finais exigem somente o protocolo de 1.000 eventos")
    result = lengths["1000"]
    models = result["models"]
    mle = result["mle"]
    outputs: list[Path] = []

    x = np.arange(len(PARAMETERS))
    width = 0.25
    figure, axis = plt.subplots(figsize=(8.2, 4.4))
    for offset, (key, label) in zip((-width, 0.0), MODEL_SPECS, strict=True):
        values = [models[key]["per_parameter"][name]["mse_mean"] for name in PARAMETERS]
        errors = [models[key]["per_parameter"][name]["mse_std"] for name in PARAMETERS]
        axis.bar(
            x + offset,
            values,
            width,
            yerr=errors,
            capsize=3,
            label=label,
        )
    axis.bar(
        x + width,
        [mle["per_parameter"][name]["mse"] for name in PARAMETERS],
        width,
        label="MLE",
    )
    axis.set_xticks(x, PARAMETER_LABELS)
    axis.set_yscale("log")
    axis.set_ylabel(text["parameter_ylabel"])
    axis.set_xlabel(text["parameter_xlabel"])
    axis.set_title(text["parameter_title"])
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    path = output_dir / f"parameter_mse{suffix}.pdf"
    _save_figure(figure, path)
    outputs.append(path)

    x = np.arange(len(REGIMES))
    figure, axis = plt.subplots(figsize=(8.2, 4.4))
    for offset, (key, label) in zip((-width, 0.0), MODEL_SPECS, strict=True):
        values = [
            models[key]["by_branching_ratio"][name]["macro_mse_mean"]
            for name in REGIMES
        ]
        errors = [
            models[key]["by_branching_ratio"][name].get("macro_mse_std", 0.0)
            for name in REGIMES
        ]
        axis.bar(
            x + offset,
            values,
            width,
            yerr=errors,
            capsize=3,
            label=label,
        )
    axis.bar(
        x + width,
        [mle["by_branching_ratio"][name]["macro_mse"] for name in REGIMES],
        width,
        label="MLE",
    )
    axis.set_xticks(x, text["regimes"])
    axis.set_yscale("log")
    axis.set_ylabel(text["regime_ylabel"])
    axis.set_xlabel(text["regime_xlabel"])
    axis.set_title(text["regime_title"])
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    path = output_dir / f"branching_regime_mse{suffix}.pdf"
    _save_figure(figure, path)
    outputs.append(path)

    labels = ["LSTM", "Transformer", "MLE"]
    latency = [
        models["lstm_lee"]["latency_batch_1_median_ms_mean"],
        models["transformer"]["latency_batch_1_median_ms_mean"],
        mle["seconds_per_path_wall_clock"] * 1000.0,
    ]
    figure, axis = plt.subplots(figsize=(7.3, 4.4))
    bars = axis.bar(labels, latency)
    axis.bar_label(bars, fmt="%.2f ms", padding=3)
    axis.set_yscale("log")
    axis.set_ylabel(text["cost_ylabel"])
    axis.set_xlabel(text["cost_xlabel"])
    axis.set_title(text["cost_title"])
    axis.grid(axis="y", alpha=0.25)
    path = output_dir / f"inference_cost{suffix}.pdf"
    _save_figure(figure, path)
    outputs.append(path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("artifacts/final_hawkes/lee_local_optimized/summary.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("thesis/build/figures")
    )
    parser.add_argument("--lang", choices=LANGUAGES, default="pt")
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    if summary.get("missing"):
        raise RuntimeError("Resumo ainda possui jobs neurais ausentes")
    outputs = generate_figures(summary, args.output_dir, args.lang)
    print(json.dumps({"figures": [str(path) for path in outputs]}, indent=2))


if __name__ == "__main__":
    main()
