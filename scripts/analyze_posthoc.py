#!/usr/bin/env python
"""Análise post-hoc do experimento final (TG — estimação neural de Hawkes).

Não retreina modelos e não altera nenhum artefato do repositório. Lê o conjunto de
teste sintético congelado, os checkpoints ``best.pt`` das cinco seeds de cada rede e
os registros do MLE; grava previsões por trajetória, métricas robustas, testes
pareados, diagnósticos do MLE, latências comparáveis e figuras em ``--out``.

Executar a partir da raiz do repositório do TG:

    cd <raiz do repositório>
    PYTHONPATH=src .venv/bin/python scripts/analyze_posthoc.py \
        --out ~/Documents/JM/TG/04_analises_posthoc/resultados \
        --figures ~/Documents/JM/TG/04_analises_posthoc/figuras \
        --calibration ~/Documents/JM/TG/04_analises_posthoc/resultados/calibracao_mle_jan15-22.json

``--lang en`` grava as figuras como ``<nome>_en.{pdf,png}``. ``--figures-only`` apenas
redesenha as figuras a partir do que já está salvo em ``--out`` (``posthoc_sintetico.json``
e ``previsoes_por_trajetoria/``), dos registros do MLE, dos alvos do teste sintético e de
``--calibration``: não carrega checkpoints, não reajusta o MLE e não mede latências.
O subconjunto de 3.000 trajetórias da figura de beta é o mesmo da execução completa.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy import stats  # noqa: E402

from hawkes_crypto.config import load_experiment_config  # noqa: E402
from hawkes_crypto.estimate.mle import (  # noqa: E402
    _encode,
    _negative_log_likelihood_with_gradient,
    fit_mle,
)
from hawkes_crypto.simulate.hawkes_sim import HawkesParameters  # noqa: E402
from hawkes_crypto.synthetic import load_split, manifest_records  # noqa: E402
from hawkes_crypto.training import build_model  # noqa: E402

PARAMS = ("mu", "alpha_self", "alpha_cross", "beta")
LABELS = {"mu": r"$\mu$", "alpha_self": r"$\alpha_{self}$",
          "alpha_cross": r"$\alpha_{cross}$", "beta": r"$\beta$"}
REGIMES = ("low", "intermediate", "near_critical")
COLORS = {"MLE": "#4c72b0", "LSTM": "#dd8452", "Transformer": "#55a868",
          "Média a priori": "#8c8c8c"}
COMPARISONS = {
    "LSTM_ens vs Transformer_ens": ("LSTM (ensemble)", "Transformer (ensemble)"),
    "LSTM_ens vs MLE": ("LSTM (ensemble)", "MLE"),
    "Transformer_ens vs MLE": ("Transformer (ensemble)", "MLE"),
}
N_BOOT = 2000
RNG_SEED = 20261030
BETA_SUBSET = 3000
LANGUAGES = ("pt", "en")
FIGURE_TEXT = {
    "pt": {
        "suffix": "",
        "calibration_points": "MLE em janelas reais (15–22/jan/2024)",
        "calibration_prior": "suporte do prior sintético",
        "calibration_mu": r"$\mu$ (escala log)",
        "calibration_title": "Parâmetros MLE em dados reais versus prior usado no treino",
        "beta_true": r"$\beta$ verdadeiro",
        "beta_hat": r"$\hat\beta$ (escala log)",
        "g_true": "g verdadeiro",
        "r2_title": "Fração da variância a priori explicada, por parâmetro (teste sintético)",
        "regimes": {"low": "g baixo", "intermediate": "g intermediário",
                    "near_critical": "g quase crítico"},
        "beta_error": r"$|\hat\beta-\beta|$ (escala log)",
    },
    "en": {
        "suffix": "_en",
        "calibration_points": "MLE on real windows (15–22 Jan 2024)",
        "calibration_prior": "synthetic prior support",
        "calibration_mu": r"$\mu$ (log scale)",
        "calibration_title": "MLE parameters on real data versus the prior used in training",
        "beta_true": r"true $\beta$",
        "beta_hat": r"$\hat\beta$ (log scale)",
        "g_true": "true g",
        "r2_title": "Fraction of prior variance explained, by parameter (synthetic test set)",
        "regimes": {"low": "low g", "intermediate": "intermediate g",
                    "near_critical": "near-critical g"},
        "beta_error": r"$|\hat\beta-\beta|$ (log scale)",
    },
}


# --------------------------------------------------------------------------- métricas
def branching(values: np.ndarray) -> np.ndarray:
    return (values[:, 1] + values[:, 2]) / values[:, 3]


def regime_indices(targets: np.ndarray) -> dict[str, np.ndarray]:
    g = branching(targets)
    return {
        "low": np.flatnonzero(g < 0.35),
        "intermediate": np.flatnonzero((g >= 0.35) & (g < 0.70)),
        "near_critical": np.flatnonzero(g >= 0.70),
    }


def block_metrics(pred: np.ndarray, true: np.ndarray, prior_var: np.ndarray) -> dict:
    err = pred - true
    squared = err**2
    log_err = np.log(np.maximum(pred, 1e-12)) - np.log(true)
    per_parameter = {}
    for k, name in enumerate(PARAMS):
        e = err[:, k]
        per_parameter[name] = {
            "mse": float(np.mean(e**2)),
            "mae": float(np.mean(np.abs(e))),
            "median_ae": float(np.median(np.abs(e))),
            "p95_ae": float(np.quantile(np.abs(e), 0.95)),
            "bias": float(np.mean(e)),
            "r2_vs_prior": float(1.0 - np.mean(e**2) / prior_var[k]),
            "log_mse": float(np.mean(log_err[:, k] ** 2)),
        }
    return {
        "n": int(len(true)),
        "macro_mse": float(squared.mean()),
        "macro_mae": float(np.abs(err).mean()),
        "macro_median_ae": float(np.mean([np.median(np.abs(err[:, k])) for k in range(4)])),
        "normalized_macro_mse": float(np.mean(squared.mean(axis=0) / prior_var)),
        "macro_log_mse": float(np.mean(log_err**2)),
        "g_mse": float(np.mean((branching(pred) - branching(true)) ** 2)),
        "per_parameter": per_parameter,
    }


def all_metrics(pred, true, prior_var, groups) -> dict:
    result = {"overall": block_metrics(pred, true, prior_var)}
    for name, idx in groups.items():
        result[name] = block_metrics(pred[idx], true[idx], prior_var)
    return result


def trajectory_loss(pred, true, prior_var) -> np.ndarray:
    """Erro quadrático padronizado por trajetória (média nos 4 parâmetros)."""
    return np.mean((pred - true) ** 2 / prior_var, axis=1)


def bootstrap_batches(rng, n, n_boot=N_BOOT):
    """Índices bootstrap em lotes de 100 reamostras (ordem de consumo do gerador fixa)."""
    for start in range(0, n_boot, 100):
        stop = min(start + 100, n_boot)
        yield start, stop, rng.integers(0, n, size=(stop - start, n))


def paired_comparison(loss_a, loss_b, rng, n_boot=N_BOOT) -> dict:
    """Compara A e B nas mesmas trajetórias. Diferença < 0 favorece A."""
    diff = loss_a - loss_b
    n = len(diff)
    boot = np.empty(n_boot)
    for start, stop, idx in bootstrap_batches(rng, n, n_boot):
        boot[start:stop] = diff[idx].mean(axis=1)
    wilcoxon = stats.wilcoxon(loss_a, loss_b)
    return {
        "mean_loss_a": float(loss_a.mean()),
        "mean_loss_b": float(loss_b.mean()),
        "mean_difference_a_minus_b": float(diff.mean()),
        "bootstrap_95ci": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "median_loss_a": float(np.median(loss_a)),
        "median_loss_b": float(np.median(loss_b)),
        "win_rate_a": float(np.mean(loss_a < loss_b)),
        "wilcoxon_p": float(wilcoxon.pvalue),
    }


# ------------------------------------------------------------------- previsões neurais
def neural_predictions(config, model_name, seed, inputs, device, cache_dir) -> np.ndarray:
    cache = cache_dir / f"{model_name}_seed{seed}.npy"
    if cache.exists():
        return np.load(cache)
    checkpoint = (config.artifacts_root / "length_1000" / model_name / f"seed_{seed}"
                  / "best.pt")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model(model_name, config)
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    batch = 256 if model_name == "lstm_lee" else 32
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(inputs), batch):
            chunk = torch.from_numpy(inputs[start:start + batch]).to(device)
            outputs.append(model(chunk).float().cpu().numpy())
    predictions = np.concatenate(outputs).astype(np.float64)
    np.save(cache, predictions)
    return predictions


def cpu_latency(config, model_name, sample, threads, repetitions=30) -> dict:
    torch.set_num_threads(threads)
    model = build_model(model_name, config).eval()
    tensor = torch.from_numpy(sample)
    timings = []
    with torch.inference_mode():
        for _ in range(3):
            model(tensor)
        for _ in range(repetitions):
            started = time.perf_counter()
            model(tensor)
            timings.append((time.perf_counter() - started) * 1000.0)
    return {"median_ms": float(np.median(timings)), "p95_ms": float(np.quantile(timings, 0.95))}


# --------------------------------------------------------------------- diagnóstico MLE
def _log_likelihood(task):
    params, times, types = task
    value, _ = _negative_log_likelihood_with_gradient(_encode(params), times, types)
    return -float(value)


def _refit(task):
    times, types, seed = task
    fit = fit_mle(times, types, n_starts=20, seed=seed, maxiter=1000)
    p = fit.parameters
    return [p.mu, p.alpha_self, p.alpha_cross, p.beta], fit.log_likelihood


def path_of(inputs, index):
    times = np.cumsum(inputs[index, :, 0], dtype=np.float64)
    types = inputs[index, :, 1].astype(np.int8)
    return times, types


def to_params(row) -> HawkesParameters:
    return HawkesParameters(mu=float(row[0]), alpha_self=float(row[1]),
                            alpha_cross=float(row[2]), beta=float(row[3]))


# ---------------------------------------------------------------------------- figuras
def worst_beta_paths(mle: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Trajetórias ordenadas pelo erro quadrático de beta do MLE (pior primeiro)."""
    sq = (mle[:, 3] - targets[:, 3]) ** 2
    return np.argsort(sq)[::-1]


def replay_rng_until_figures(rng, n_paths, groups, worst, n_mle_timing=20) -> None:
    """Consome ``rng`` exatamente como ``main`` até o sorteio do subconjunto da figura.

    Deve acompanhar a ordem de ``main``: bootstrap das comparações pareadas (agregado e
    regimes, depois sem beta), sorteio do grupo controle do diagnóstico do MLE e sorteio
    das trajetórias cronometradas do MLE (1 e 5 inícios).
    """
    sizes = [n_paths, *(len(idx) for idx in groups.values())]
    for _ in COMPARISONS:
        for n in sizes:
            for _batch in bootstrap_batches(rng, n):
                pass
    for _ in COMPARISONS:
        for _batch in bootstrap_batches(rng, n_paths):
            pass
    rng.choice(np.setdiff1d(np.arange(n_paths), worst), len(worst), replace=False)
    for _starts in (1, 5):
        rng.choice(n_paths, n_mle_timing, replace=False)


def calibration_arrays(calibration: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    empirical = np.asarray([[rec["parameters"][p] for p in PARAMS]
                            for rec in calibration["records"]], dtype=np.float64)
    g_emp = branching(empirical)
    s_emp = empirical[:, 1] / (empirical[:, 1] + empirical[:, 2])
    return empirical, g_emp, s_emp


def plot_calibration(empirical, g_emp, s_emp, sampling, figures: Path, lang: str) -> None:
    text = FIGURE_TEXT[lang]
    s = sampling
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(empirical[:, 3], s_emp, s=6, alpha=0.4, color="#c44e52",
                    label=text["calibration_points"])
    axes[0].add_patch(plt.Rectangle((s.beta[0], s.self_fraction[0]),
                                    s.beta[1] - s.beta[0],
                                    s.self_fraction[1] - s.self_fraction[0],
                                    fill=False, lw=2, color="black",
                                    label=text["calibration_prior"]))
    axes[0].set_xlabel(r"$\beta$")
    axes[0].set_ylabel(r"$\alpha_{self}/(\alpha_{self}+\alpha_{cross})$")
    axes[0].legend(fontsize=8, loc="lower right")
    axes[1].scatter(empirical[:, 0], g_emp, s=6, alpha=0.4, color="#c44e52")
    axes[1].add_patch(plt.Rectangle((s.mu[0], s.g_strata[0][0]), s.mu[1] - s.mu[0],
                                    s.g_strata[-1][1] - s.g_strata[0][0], fill=False,
                                    lw=2, color="black"))
    axes[1].set_xscale("log")
    axes[1].set_xlabel(text["calibration_mu"])
    axes[1].set_ylabel(r"$g$")
    fig.suptitle(text["calibration_title"], fontsize=11)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(figures / f"posthoc_calibracao_vs_prior{text['suffix']}.{ext}", dpi=160)
    plt.close(fig)


def plot_synthetic_figures(targets, groups, main_estimators, metrics, subset,
                           figures: Path, lang: str) -> None:
    text = FIGURE_TEXT[lang]
    suffix = text["suffix"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for axis, (name, pred) in zip(axes, main_estimators.items(), strict=True):
        g_true = branching(targets[subset])
        scatter = axis.scatter(targets[subset, 3], pred[subset, 3], c=g_true, s=4,
                               cmap="viridis", alpha=0.6, vmin=0.1, vmax=0.95)
        axis.plot([0.8, 2.0], [0.8, 2.0], color="black", lw=1)
        axis.set_yscale("log")
        axis.set_title(name)
        axis.set_xlabel(text["beta_true"])
    axes[0].set_ylabel(text["beta_hat"])
    fig.colorbar(scatter, ax=axes, label=text["g_true"], shrink=0.8)
    for ext in ("pdf", "png"):
        fig.savefig(figures / f"posthoc_beta_verdadeiro_vs_estimado{suffix}.{ext}", dpi=160,
                    bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 4))
    width = 0.2
    bar_estimators = {"MLE": metrics["MLE"]["overall"],
                      "LSTM": metrics["LSTM (ensemble)"]["overall"],
                      "Transformer": metrics["Transformer (ensemble)"]["overall"]}
    for offset, (name, block) in enumerate(bar_estimators.items()):
        values = [block["per_parameter"][p]["r2_vs_prior"] for p in PARAMS]
        axis.bar(np.arange(4) + (offset - 1) * width, np.clip(values, -1, 1), width,
                 label=name, color=COLORS[name])
        for k, value in enumerate(values):
            if value < -1:
                axis.annotate(f"{value:.1f}", (k + (offset - 1) * width, -1),
                              ha="center", va="top", fontsize=7)
    axis.axhline(0, color="black", lw=0.8)
    axis.set_xticks(np.arange(4), [LABELS[p] for p in PARAMS])
    axis.set_ylabel(r"$R^2 = 1 - \mathrm{MSE}/\mathrm{Var}_{prior}$")
    axis.set_ylim(-1.05, 1.0)
    axis.legend()
    axis.set_title(text["r2_title"])
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(figures / f"posthoc_r2_por_parametro{suffix}.{ext}", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for axis, regime in zip(axes, REGIMES, strict=True):
        idx = groups[regime]
        data = [np.abs(pred[idx, 3] - targets[idx, 3]) for pred in main_estimators.values()]
        box = axis.boxplot(data, showfliers=True, flierprops={"markersize": 2, "alpha": 0.3},
                           patch_artist=True)
        for patch, name in zip(box["boxes"], main_estimators, strict=True):
            patch.set_facecolor(COLORS[name])
        axis.set_xticks([1, 2, 3], list(main_estimators))
        axis.set_yscale("log")
        axis.set_title(text["regimes"][regime])
    axes[0].set_ylabel(text["beta_error"])
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(figures / f"posthoc_erro_beta_por_regime{suffix}.{ext}", dpi=160)
    plt.close(fig)


def load_mle_records(config) -> tuple[np.ndarray, np.ndarray]:
    records_path = config.artifacts_root / "length_1000" / "mle" / "records.jsonl"
    records = [json.loads(line) for line in records_path.read_text().splitlines() if line]
    records.sort(key=lambda item: int(item["index"]))
    mle = np.asarray([item["parameters"] for item in records], dtype=np.float64)
    mle_ll = np.asarray([item["log_likelihood"] for item in records], dtype=np.float64)
    return mle, mle_ll


def figures_only(args, config) -> None:
    """Redesenha as figuras a partir dos artefatos salvos, sem recomputação pesada."""
    report_path = args.out / "posthoc_sintetico.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    cache_dir = args.out / "previsoes_por_trajetoria"
    _, targets = load_split(config.synthetic_root, 1000, "test")
    targets = targets.astype(np.float64)
    groups = regime_indices(targets)
    mle, _ = load_mle_records(config)
    ensembles = {}
    for model_name in ("lstm_lee", "transformer"):
        per_seed = []
        for seed in config.training.seeds:
            cache = cache_dir / f"{model_name}_seed{seed}.npy"
            if not cache.exists():
                raise FileNotFoundError(f"Previsões salvas ausentes: {cache}")
            per_seed.append(np.load(cache))
        ensembles[model_name] = np.mean(per_seed, axis=0)

    if args.calibration and args.calibration.exists():
        calibration = json.loads(args.calibration.read_text())
        plot_calibration(*calibration_arrays(calibration), config.sampling, args.figures,
                         args.lang)

    n_diagnose = int(report["mle_diagnostics"]["likelihood_check"]["worst_n"])
    worst = worst_beta_paths(mle, targets)[:n_diagnose]
    rng = np.random.default_rng(RNG_SEED)
    replay_rng_until_figures(rng, len(targets), groups, worst)
    subset = rng.choice(len(targets), BETA_SUBSET, replace=False)
    main_estimators = {"MLE": mle, "LSTM": ensembles["lstm_lee"],
                       "Transformer": ensembles["transformer"]}
    plot_synthetic_figures(targets, groups, main_estimators, report["metrics"], subset,
                           args.figures, args.lang)
    print(json.dumps({"done": True, "figures": str(args.figures), "lang": args.lang}),
          flush=True)


# ------------------------------------------------------------------------------ main
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/lee_local_optimized.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--diagnose", type=int, default=300)
    parser.add_argument("--refit", type=int, default=100)
    parser.add_argument("--lang", choices=LANGUAGES, default="pt",
                        help="língua das figuras (en grava <nome>_en.pdf)")
    parser.add_argument("--figures-only", action="store_true",
                        help="só redesenha as figuras a partir dos artefatos salvos em --out")
    args = parser.parse_args()
    args.figures.mkdir(parents=True, exist_ok=True)
    if args.figures_only:
        figures_only(args, load_experiment_config(args.config))
        return
    args.out.mkdir(parents=True, exist_ok=True)
    cache_dir = args.out / "previsoes_por_trajetoria"
    cache_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    config = load_experiment_config(args.config)
    report: dict = {"config": str(config.source_path)}

    # ---- dados de teste e prior de treino
    inputs, targets = load_split(config.synthetic_root, 1000, "test")
    targets = targets.astype(np.float64)
    train_targets = np.concatenate([
        np.load(Path(str(record["path"])))["targets"]
        for record in manifest_records(config.synthetic_root, 1000, "train")
    ]).astype(np.float64)
    prior_mean = train_targets.mean(axis=0)
    prior_var = train_targets.var(axis=0)
    groups = regime_indices(targets)
    report["prior"] = {
        "train_mean": dict(zip(PARAMS, prior_mean.tolist(), strict=True)),
        "train_variance": dict(zip(PARAMS, prior_var.tolist(), strict=True)),
        "regime_sizes": {name: int(len(idx)) for name, idx in groups.items()},
    }

    # ---- MLE (registros já existentes)
    mle, mle_ll = load_mle_records(config)

    # ---- redes: previsões por trajetória a partir dos checkpoints congelados
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    predictions: dict[str, dict[int, np.ndarray]] = {"lstm_lee": {}, "transformer": {}}
    sanity = {}
    for model_name in predictions:
        for seed in config.training.seeds:
            started = time.perf_counter()
            pred = neural_predictions(config, model_name, seed, inputs, device, cache_dir)
            predictions[model_name][seed] = pred
            result_path = (config.artifacts_root / "length_1000" / model_name
                           / f"seed_{seed}" / "result.json")
            reported = json.loads(result_path.read_text())["evaluation"]["macro_mse"]
            recomputed = float(np.mean((pred - targets) ** 2))
            sanity[f"{model_name}/{seed}"] = {
                "reported_macro_mse": reported,
                "recomputed_macro_mse": recomputed,
                "relative_difference": abs(recomputed - reported) / reported,
                "seconds": time.perf_counter() - started,
            }
            progress = {"predicted": model_name, "seed": seed, **sanity[f"{model_name}/{seed}"]}
            print(json.dumps(progress), flush=True)
    report["sanity_check_vs_result_json"] = sanity

    estimators = {
        "Média a priori": np.tile(prior_mean, (len(targets), 1)),
        "MLE": mle,
        "LSTM (média das seeds)": None,
        "Transformer (média das seeds)": None,
        "LSTM (ensemble)": np.mean(list(predictions["lstm_lee"].values()), axis=0),
        "Transformer (ensemble)": np.mean(list(predictions["transformer"].values()), axis=0),
    }
    metrics = {}
    for name, pred in estimators.items():
        if pred is not None:
            metrics[name] = all_metrics(pred, targets, prior_var, groups)
    for label, model_name in (("LSTM (média das seeds)", "lstm_lee"),
                              ("Transformer (média das seeds)", "transformer")):
        per_seed = [all_metrics(p, targets, prior_var, groups)
                    for p in predictions[model_name].values()]
        metrics[label] = {"per_seed": per_seed}
    report["metrics"] = metrics

    # ---- comparações estatísticas
    seed_mse = {
        name: np.asarray([np.mean((p - targets) ** 2) for p in predictions[name].values()])
        for name in predictions
    }
    welch = stats.ttest_ind(seed_mse["lstm_lee"], seed_mse["transformer"], equal_var=False)
    mann = stats.mannwhitneyu(seed_mse["lstm_lee"], seed_mse["transformer"],
                              alternative="two-sided", method="exact")
    report["seed_level_lstm_vs_transformer"] = {
        "lstm_macro_mse_per_seed": seed_mse["lstm_lee"].tolist(),
        "transformer_macro_mse_per_seed": seed_mse["transformer"].tolist(),
        "welch_t": float(welch.statistic), "welch_p": float(welch.pvalue),
        "mann_whitney_p": float(mann.pvalue),
    }
    loss = {name: trajectory_loss(pred, targets, prior_var)
            for name, pred in estimators.items() if pred is not None}
    paired = {}
    for key, (a, b) in COMPARISONS.items():
        paired[key] = {"overall": paired_comparison(loss[a], loss[b], rng)}
        for regime, idx in groups.items():
            paired[key][regime] = paired_comparison(loss[a][idx], loss[b][idx], rng)
    # a mesma comparação excluindo beta (somente mu e alphas)
    mask = np.array([True, True, True, False])
    for key, (a, b) in COMPARISONS.items():
        la = np.mean((estimators[a] - targets)[:, mask] ** 2 / prior_var[mask], axis=1)
        lb = np.mean((estimators[b] - targets)[:, mask] ** 2 / prior_var[mask], axis=1)
        paired[key]["overall_without_beta"] = paired_comparison(la, lb, rng)
    report["paired_trajectory_level"] = paired

    # ---- diagnóstico do MLE
    beta_hat = mle[:, 3]
    beta_true = targets[:, 3]
    diag = {"beta_hat_quantiles": dict(zip(
        ["q01", "q05", "q25", "q50", "q75", "q95", "q99", "max"],
        np.quantile(beta_hat, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]).tolist(),
        strict=True))}
    for regime, idx in groups.items():
        diag[regime] = {
            "frac_beta_hat_gt_2": float(np.mean(beta_hat[idx] > 2.0)),
            "frac_beta_hat_gt_4": float(np.mean(beta_hat[idx] > 4.0)),
            "frac_beta_hat_gt_10": float(np.mean(beta_hat[idx] > 10.0)),
            "frac_abs_beta_error_gt_1": float(np.mean(np.abs(beta_hat[idx] - beta_true[idx]) > 1)),
        }
    # contribuição das piores 1% trajetórias ao MSE de beta do MLE
    sq = (beta_hat - beta_true) ** 2
    order = worst_beta_paths(mle, targets)
    top1 = order[: len(order) // 100]
    diag["share_of_beta_sse_from_worst_1pct"] = float(sq[top1].sum() / sq.sum())
    diag["beta_mse_without_worst_1pct"] = float(np.delete(sq, top1).mean())

    worst = order[: args.diagnose]
    control = rng.choice(np.setdiff1d(np.arange(len(targets)), worst), args.diagnose,
                         replace=False)
    tasks_true, tasks_hat = [], []
    for index in np.concatenate([worst, control]):
        times, types = path_of(inputs, index)
        tasks_true.append((to_params(targets[index]), times, types))
        tasks_hat.append((to_params(mle[index]), times, types))
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        ll_true = np.asarray(list(executor.map(_log_likelihood, tasks_true, chunksize=8)))
        ll_hat = np.asarray(list(executor.map(_log_likelihood, tasks_hat, chunksize=8)))
        refit_index = worst[: args.refit]
        refits = list(executor.map(
            _refit,
            [(*path_of(inputs, i), int(config.seed + i + 777)) for i in refit_index],
            chunksize=4,
        ))
    gap = ll_hat - ll_true
    n_worst = len(worst)
    diag["likelihood_check"] = {
        "worst_n": n_worst,
        "frac_worst_where_mle_ll_ge_true_ll": float(np.mean(gap[:n_worst] >= -1e-6)),
        "frac_control_where_mle_ll_ge_true_ll": float(np.mean(gap[n_worst:] >= -1e-6)),
        "median_ll_gain_worst": float(np.median(gap[:n_worst])),
        "regime_of_worst": {r: int(np.isin(worst, groups[r]).sum()) for r in REGIMES},
    }
    refit_params = np.asarray([item[0] for item in refits])
    refit_ll = np.asarray([item[1] for item in refits])
    diag["refit_20_starts_on_worst"] = {
        "n": int(len(refit_index)),
        "frac_ll_improved_by_more_than_1e-3": float(np.mean(refit_ll > mle_ll[refit_index] + 1e-3)),
        "beta_mse_before": float(np.mean((mle[refit_index, 3] - beta_true[refit_index]) ** 2)),
        "beta_mse_after": float(np.mean((refit_params[:, 3] - beta_true[refit_index]) ** 2)),
    }
    report["mle_diagnostics"] = diag

    # ---- latências comparáveis (mesma máquina, CPU)
    sample = inputs[:1]
    latency = {}
    for model_name in ("lstm_lee", "transformer"):
        latency[f"{model_name}_cpu_1thread"] = cpu_latency(config, model_name, sample, 1)
        latency[f"{model_name}_cpu_default_threads"] = cpu_latency(
            config, model_name, sample, os.cpu_count() or 4)
    torch.set_num_threads(1)
    mle_times = {}
    for starts in (1, 5):
        seconds = []
        for index in rng.choice(len(targets), 20, replace=False):
            times, types = path_of(inputs, index)
            started = time.perf_counter()
            fit_mle(times, types, n_starts=starts, seed=int(index), maxiter=1000)
            seconds.append(time.perf_counter() - started)
        mle_times[f"n_starts_{starts}"] = {"median_s": float(np.median(seconds)),
                                           "mean_s": float(np.mean(seconds))}
    latency["mle_single_process"] = mle_times
    report["latency"] = latency

    # ---- calibração empírica versus prior sintético
    if args.calibration and args.calibration.exists():
        calibration = json.loads(args.calibration.read_text())
        empirical, g_emp, s_emp = calibration_arrays(calibration)
        s = config.sampling
        inside = {
            "mu": (empirical[:, 0] >= s.mu[0]) & (empirical[:, 0] <= s.mu[1]),
            "beta": (empirical[:, 3] >= s.beta[0]) & (empirical[:, 3] <= s.beta[1]),
            "self_fraction": (s_emp >= s.self_fraction[0]) & (s_emp <= s.self_fraction[1]),
            "g": (g_emp >= s.g_strata[0][0]) & (g_emp <= s.g_strata[-1][1]),
        }
        inside["all"] = np.logical_and.reduce(list(inside.values()))
        quant = lambda v: dict(zip(["q05", "q25", "q50", "q75", "q95"],  # noqa: E731
                                   np.quantile(v, [0.05, 0.25, 0.5, 0.75, 0.95]).tolist(),
                                   strict=True))
        report["calibration_vs_prior"] = {
            "windows": int(len(empirical)),
            "fraction_inside_prior": {k: float(v.mean()) for k, v in inside.items()},
            "quantiles": {**{p: quant(empirical[:, k]) for k, p in enumerate(PARAMS)},
                          "g": quant(g_emp), "self_fraction": quant(s_emp)},
            "frac_alpha_cross_below_1e-3": float(np.mean(empirical[:, 2] < 1e-3)),
        }
        plot_calibration(empirical, g_emp, s_emp, config.sampling, args.figures, args.lang)

    # ---- figuras
    main_estimators = {"MLE": mle, "LSTM": estimators["LSTM (ensemble)"],
                       "Transformer": estimators["Transformer (ensemble)"]}
    subset = rng.choice(len(targets), BETA_SUBSET, replace=False)
    plot_synthetic_figures(targets, groups, main_estimators, metrics, subset, args.figures,
                           args.lang)

    (args.out / "posthoc_sintetico.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"done": True, "out": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
