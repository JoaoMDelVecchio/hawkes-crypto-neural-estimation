"""Métricas comuns aos estimadores neurais e de máxima verossimilhança."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import t as student_t

PARAMETER_NAMES = ("mu", "alpha_self", "alpha_cross", "beta")


def parameter_metrics(predictions: ArrayLike, targets: ArrayLike) -> dict[str, object]:
    predicted = np.asarray(predictions, dtype=np.float64)
    expected = np.asarray(targets, dtype=np.float64)
    if predicted.shape != expected.shape or predicted.ndim != 2 or predicted.shape[1] != 4:
        raise ValueError("predictions e targets devem ter formato [n, 4]")
    error = predicted - expected
    per_parameter: dict[str, dict[str, float]] = {}
    for index, name in enumerate(PARAMETER_NAMES):
        parameter_error = error[:, index]
        denominator = np.maximum(np.abs(expected[:, index]), 1e-12)
        per_parameter[name] = {
            "mse": float(np.mean(parameter_error**2)),
            "mae": float(np.mean(np.abs(parameter_error))),
            "rmse": float(np.sqrt(np.mean(parameter_error**2))),
            "bias": float(np.mean(parameter_error)),
            "median_absolute_relative_error": float(
                np.median(np.abs(parameter_error) / denominator)
            ),
        }
    predicted_g = (predicted[:, 1] + predicted[:, 2]) / predicted[:, 3]
    expected_g = (expected[:, 1] + expected[:, 2]) / expected[:, 3]
    valid = (
        np.isfinite(predicted).all(axis=1)
        & (predicted > 0).all(axis=1)
        & (predicted_g < 1)
    )
    return {
        "samples": int(len(predicted)),
        "macro_mse": float(np.mean(error**2)),
        "macro_mae": float(np.mean(np.abs(error))),
        "per_parameter": per_parameter,
        "branching_ratio": {
            "mse": float(np.mean((predicted_g - expected_g) ** 2)),
            "bias": float(np.mean(predicted_g - expected_g)),
        },
        "validity_rate": float(np.mean(valid)),
    }


def latency_metrics(samples_ms: Sequence[float]) -> dict[str, float]:
    values = np.asarray(samples_ms, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Amostras de latência devem ser finitas e não vazias")
    return {
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.quantile(values, 0.95)),
        "mean_ms": float(np.mean(values)),
    }


def aggregate_seed_metrics(metrics: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not metrics:
        raise ValueError("É necessária pelo menos uma seed")
    macro = np.asarray([float(item["macro_mse"]) for item in metrics])
    macro_mae = np.asarray([float(item["macro_mae"]) for item in metrics])
    validity = np.asarray([float(item["validity_rate"]) for item in metrics])
    standard_error = macro.std(ddof=1) / np.sqrt(len(macro)) if len(macro) > 1 else 0.0
    critical_value = (
        float(student_t.ppf(0.975, df=len(macro) - 1)) if len(macro) > 1 else 0.0
    )
    macro_mse_95ci = [
        float(macro.mean() - critical_value * standard_error),
        float(macro.mean() + critical_value * standard_error),
    ]
    per_parameter = {}
    for name in PARAMETER_NAMES:
        values = np.asarray(
            [float(item["per_parameter"][name]["mse"]) for item in metrics]
        )
        mae_values = np.asarray(
            [float(item["per_parameter"][name]["mae"]) for item in metrics]
        )
        per_parameter[name] = {
            "mse_mean": float(values.mean()),
            "mse_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "mae_mean": float(mae_values.mean()),
            "mae_std": (
                float(mae_values.std(ddof=1)) if len(mae_values) > 1 else 0.0
            ),
        }
    branching_mse = np.asarray(
        [float(item["branching_ratio"]["mse"]) for item in metrics]
    )
    by_branching_ratio: dict[str, dict[str, float]] = {}
    for regime in ("low", "intermediate", "near_critical"):
        regime_metrics = [
            item["by_branching_ratio"][regime]
            for item in metrics
            if regime in item.get("by_branching_ratio", {})
        ]
        if regime_metrics:
            regime_mse = np.asarray(
                [float(item["macro_mse"]) for item in regime_metrics]
            )
            by_branching_ratio[regime] = {
                "macro_mse_mean": float(regime_mse.mean()),
                "macro_mse_std": (
                    float(regime_mse.std(ddof=1)) if len(regime_mse) > 1 else 0.0
                ),
                "macro_mae_mean": float(
                    np.mean([float(item["macro_mae"]) for item in regime_metrics])
                ),
            }
    latencies = [
        float(item["latency_batch_1"]["median_ms"])
        for item in metrics
        if "latency_batch_1" in item
    ]
    return {
        "seeds": len(metrics),
        "macro_mse_mean": float(macro.mean()),
        "macro_mse_std": float(macro.std(ddof=1)) if len(macro) > 1 else 0.0,
        "macro_mse_95ci": macro_mse_95ci,
        "macro_mse_approx_95ci": macro_mse_95ci,
        "macro_mae_mean": float(macro_mae.mean()),
        "macro_mae_std": (
            float(macro_mae.std(ddof=1)) if len(macro_mae) > 1 else 0.0
        ),
        "validity_rate_mean": float(validity.mean()),
        "branching_ratio_mse_mean": float(branching_mse.mean()),
        "per_parameter": per_parameter,
        "by_branching_ratio": by_branching_ratio,
        "latency_batch_1_median_ms_mean": (
            float(np.mean(latencies)) if latencies else None
        ),
    }


def group_indices_by_branching_ratio(targets: ArrayLike) -> dict[str, NDArray[np.int64]]:
    expected = np.asarray(targets, dtype=np.float64)
    g = (expected[:, 1] + expected[:, 2]) / expected[:, 3]
    return {
        "low": np.flatnonzero(g < 0.35),
        "intermediate": np.flatnonzero((g >= 0.35) & (g < 0.70)),
        "near_critical": np.flatnonzero(g >= 0.70),
    }
