#!/usr/bin/env python
"""Diagnóstico de ajuste do Hawkes 2D nas janelas empíricas (etapa 3 do roadmap).

Usa ``calibration_train`` (15--22/jan/2024); validação e teste não são lidos. Na etapa 8,
``--split test_closed --open-test`` repete o mesmo diagnóstico nas janelas do teste. Para cada
janela de 1.000 eventos, com os parâmetros do MLE já ajustados por ``make calibration-full``:

1. resíduos por *time-rescaling* (Ogata, 1988; Brown et al., 2002) em cada dimensão, que devem
   ser i.i.d. Exp(1) se o modelo estiver correto; teste KS contra Exp(1) e Ljung--Box;
2. o mesmo para o Poisson homogêneo 2D, como referência;
3. razão de verossimilhança e AIC, Hawkes contra Poisson;
4. sensibilidade à grade de 100 ms: os tempos recebem um deslocamento uniforme dentro da
   célula, o MLE é reajustado e os resíduos recalculados. Sem isso, o KS rejeita também por
   causa da discretização, não só por erro de especificação;
5. controle simulado: para cada janela, uma trajetória do próprio Hawkes ajustado, em tempo
   contínuo e depois levada à grade (um evento por célula) e suavizada como em 4. A taxa de
   rejeição do controle é o que se esperaria se o modelo estivesse certo.

    PYTHONPATH=src .venv/bin/python scripts/run_real_data_diagnostics.py \
        --config configs/lee_local_optimized.yaml \
        --calibration artifacts/empirical_mle/calibration_jan15-22.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from hawkes_crypto.config import load_experiment_config  # noqa: E402
from hawkes_crypto.empirical import PARAMETER_NAMES, window_events  # noqa: E402
from hawkes_crypto.estimate.mle import fit_mle_batch, log_likelihood  # noqa: E402
from hawkes_crypto.simulate.hawkes_sim import HawkesParameters, simulate_path  # noqa: E402

GRID = 0.1
LJUNG_BOX_LAGS = 10
ALPHA = 0.05


def rescaled_residuals(
    parameters: HawkesParameters, times: np.ndarray, event_types: np.ndarray
) -> list[np.ndarray]:
    """Incrementos do compensador entre eventos consecutivos de cada dimensão."""
    alpha = parameters.alpha_matrix
    beta = parameters.beta
    excitation = np.zeros(2)
    compensator = np.zeros(2)
    last = np.zeros(2)
    residuals: list[list[float]] = [[], []]
    previous = 0.0
    for event_time, event_type in zip(times, event_types, strict=True):
        elapsed = event_time - previous
        decay = np.exp(-beta * elapsed)
        compensator += parameters.mu * elapsed + excitation * (1.0 - decay) / beta
        excitation *= decay
        dimension = int(event_type) - 1
        residuals[dimension].append(compensator[dimension] - last[dimension])
        last[dimension] = compensator[dimension]
        excitation += alpha[:, dimension]
        previous = float(event_time)
    # O primeiro resíduo de cada dimensão começa no início arbitrário da janela.
    return [np.asarray(values[1:]) for values in residuals]


def poisson_residuals(times: np.ndarray, event_types: np.ndarray) -> list[np.ndarray]:
    horizon = float(times[-1])
    out = []
    for dimension in (1, 2):
        own = times[event_types == dimension]
        out.append(np.diff(own) * len(own) / horizon)
    return out


def poisson_log_likelihood(times: np.ndarray, event_types: np.ndarray) -> float:
    horizon = float(times[-1])
    counts = np.asarray([np.sum(event_types == 1), np.sum(event_types == 2)], float)
    return float(np.sum(counts * np.log(counts / horizon) - counts))


def ljung_box_pvalue(values: np.ndarray, lags: int = LJUNG_BOX_LAGS) -> float:
    centered = values - values.mean()
    denominator = float(np.sum(centered**2))
    n = len(values)
    statistic = 0.0
    for lag in range(1, lags + 1):
        rho = float(np.sum(centered[lag:] * centered[:-lag])) / denominator
        statistic += rho**2 / (n - lag)
    return float(stats.chi2.sf(n * (n + 2) * statistic, lags))


def residual_tests(residuals: list[np.ndarray]) -> dict[str, list[float]]:
    return {
        "ks_pvalue": [float(stats.kstest(r, "expon").pvalue) for r in residuals],
        "ljung_box_pvalue": [ljung_box_pvalue(r) for r in residuals],
        "mean": [float(r.mean()) for r in residuals],
    }


def to_grid(times: np.ndarray, event_types: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Leva a trajetória à grade como no pré-processamento: um evento por célula (o último)."""
    cells = np.ceil(times / GRID - 1e-9).astype(np.int64)
    keep = np.append(cells[1:] != cells[:-1], True)
    return cells[keep] * GRID, event_types[keep]


def jitter(times: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # Há no máximo um evento por célula de 100 ms; o deslocamento preserva a ordem.
    shifted = times + rng.uniform(-GRID / 2, GRID / 2, size=len(times))
    return shifted - min(0.0, shifted[0] - GRID / 2)


def summarize(rows: list[dict], key: str) -> dict[str, float]:
    """Fração de janelas rejeitadas a 5% (em qualquer das duas dimensões)."""
    ks = np.asarray([row[key]["ks_pvalue"] for row in rows])
    lb = np.asarray([row[key]["ljung_box_pvalue"] for row in rows])
    means = np.asarray([row[key]["mean"] for row in rows])
    return {
        "ks_reject_any_dimension": float(np.mean(np.any(ks < ALPHA, axis=1))),
        "ks_reject_per_dimension": float(np.mean(ks < ALPHA)),
        "ks_median_pvalue": float(np.median(ks)),
        "ljung_box_reject_per_dimension": float(np.mean(lb < ALPHA)),
        "residual_mean_median": float(np.median(means)),
    }


def to_parameters(values: dict[str, float]) -> HawkesParameters:
    return HawkesParameters(**{name: float(values[name]) for name in PARAMETER_NAMES})


def pooled_quantiles(samples: list[np.ndarray], probabilities: np.ndarray) -> np.ndarray:
    return np.quantile(np.concatenate(samples), probabilities)


def plot(pooled: dict[str, list[np.ndarray]], rates: dict[str, dict], figures: Path) -> list[str]:
    """Figura em inglês, como o texto do TG."""
    figures.mkdir(parents=True, exist_ok=True)
    probabilities = np.linspace(0.005, 0.995, 199)
    theoretical = stats.expon.ppf(probabilities)
    styles = {
        "Poisson": ("#8c8c8c", "--"),
        "Hawkes": ("#4c72b0", "-"),
        "Hawkes, jittered grid": ("#dd8452", "-"),
    }
    figure, (left, right) = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for label, samples in pooled.items():
        color, line = styles[label]
        left.plot(theoretical, pooled_quantiles(samples, probabilities), line,
                  color=color, label=label, linewidth=1.6)
    limit = float(theoretical[-1])
    left.plot([0, limit], [0, limit], color="black", linewidth=0.8)
    left.set_xlabel("Exp(1) quantile")
    left.set_ylabel("Residual quantile")
    left.set_title("Pooled time-rescaled residuals")
    left.legend(frameon=False, fontsize=8)

    groups = (
        ("Real data", "hawkes_jitter", "#dd8452"),
        ("Simulated, grid", "control_simulated_grid_jitter", "#55a868"),
        ("Simulated, continuous", "control_simulated_continuous", "#8c8c8c"),
    )
    tests = (("KS", "ks_reject_per_dimension"), ("Ljung-Box", "ljung_box_reject_per_dimension"))
    width = 0.26
    for offset, (label, key, color) in enumerate(groups):
        values = [100 * rates[key][metric] for _, metric in tests]
        positions = np.arange(len(tests)) + (offset - 1) * width
        right.bar(positions, values, width, color=color, label=label)
    right.axhline(100 * ALPHA, color="black", linewidth=0.8, linestyle=":")
    right.set_xticks(np.arange(len(tests)), [name for name, _ in tests])
    right.set_ylabel("Rejected at 5% (%)")
    right.set_title("Rejection rate per window and dimension")
    right.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    written = []
    for suffix in ("pdf", "png"):
        path = figures / f"real_data_residual_diagnostics.{suffix}"
        figure.savefig(path, dpi=200)
        written.append(str(path))
    plt.close(figure)
    return written


PERIODS = {
    "calibration_train": "2024-01-15 a 2024-01-22",
    "test_closed": "2024-01-26 a 2024-01-28",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/lee_local_optimized.yaml"))
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("artifacts/empirical_mle/calibration_jan15-22.json"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/um_futures"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/empirical_diagnostics/diagnostico_jan15-22.json"),
    )
    parser.add_argument("--figures", type=Path, default=Path("thesis/build/figures"))
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--split", choices=tuple(PERIODS), default="calibration_train")
    parser.add_argument("--open-test", action="store_true")
    args = parser.parse_args()
    if args.split == "test_closed" and not args.open_test:
        raise SystemExit("Recusado: o teste só é lido com --open-test (etapa 8)")
    config = load_experiment_config(args.config)
    payload = json.loads(args.calibration.read_text())
    records = [
        record
        for record in payload["records"]
        if record["split"] == args.split and record.get("converged")
    ]
    if not args.open_test and any(
        record["split"] == "test_closed" for record in payload["records"]
    ):
        raise SystemExit("Recusado: o arquivo de rótulos contém o teste; use --open-test")

    started = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    paths, jittered = [], []
    for record in records:
        events = window_events(record, args.data_root, columns=["inter_arrival", "event_type"])
        times = events["inter_arrival"].cumsum().to_numpy(np.float64)
        types = events["event_type"].to_numpy(np.int8)
        paths.append((times, types))
        jittered.append((jitter(times, rng), types))

    # Controle: trajetórias do modelo ajustado, com a mesma quantidade de eventos na grade.
    continuous, gridded = [], []
    for index, record in enumerate(records):
        simulated = simulate_path(
            to_parameters(record["parameters"]), 1300, seed=args.seed + index
        )
        continuous.append((simulated.times[:1000], simulated.event_types[:1000]))
        grid_times, grid_types = to_grid(simulated.times, simulated.event_types)
        gridded.append((jitter(grid_times[:1000], rng), grid_types[:1000]))

    def fit(batch: list[tuple[np.ndarray, np.ndarray]]) -> list:
        return fit_mle_batch(
            batch,
            n_starts=config.mle.n_starts,
            seed=config.seed,
            maxiter=config.mle.maxiter,
            workers=config.mle.workers,
        )

    jitter_fits = fit(jittered)
    control = {}
    for name, batch in (("continuous", continuous), ("grid_jitter", gridded)):
        fits = fit(batch)
        control[name] = summarize(
            [
                {"r": residual_tests(rescaled_residuals(f.parameters, times, types))}
                for f, (times, types) in zip(fits, batch, strict=True)
            ],
            "r",
        )

    rows = []
    pooled: dict[str, list[np.ndarray]] = {
        "Poisson": [], "Hawkes": [], "Hawkes, jittered grid": []
    }
    for record, (times, types), (jit_times, _), jit_fit in zip(
        records, paths, jittered, jitter_fits, strict=True
    ):
        parameters = to_parameters(record["parameters"])
        hawkes = rescaled_residuals(parameters, times, types)
        poisson = poisson_residuals(times, types)
        smoothed = rescaled_residuals(jit_fit.parameters, jit_times, types)
        pooled["Hawkes"].extend(hawkes)
        pooled["Poisson"].extend(poisson)
        pooled["Hawkes, jittered grid"].extend(smoothed)
        hawkes_ll = log_likelihood(parameters, times, types)
        poisson_ll = poisson_log_likelihood(times, types)
        rows.append(
            {
                "symbol": record["symbol"],
                "date": record["date"],
                "start_event": record["start_event"],
                "branching_ratio": (parameters.alpha_self + parameters.alpha_cross)
                / parameters.beta,
                "hawkes": residual_tests(hawkes),
                "poisson": residual_tests(poisson),
                "hawkes_jitter": residual_tests(smoothed),
                "jitter_parameters": {
                    name: float(getattr(jit_fit.parameters, name)) for name in PARAMETER_NAMES
                },
                "likelihood_ratio": 2.0 * (hawkes_ll - poisson_ll),
                "delta_aic": 2.0 * (4 - 2) - 2.0 * (hawkes_ll - poisson_ll),
            }
        )

    ratios = np.asarray([row["likelihood_ratio"] for row in rows])
    # Três parâmetros a mais, um deles na fronteira (α_cross ≈ 0): χ²₃ é conservador.
    lr_pvalues = stats.chi2.sf(ratios, 3)
    jitter_values = {
        name: np.asarray([row["jitter_parameters"][name] for row in rows])
        for name in PARAMETER_NAMES
    }
    original_values = {
        name: np.asarray([record["parameters"][name] for record in records])
        for name in PARAMETER_NAMES
    }
    summary = {
        "windows": len(rows),
        "split": args.split,
        "period": PERIODS[args.split],
        "seconds": time.perf_counter() - started,
        "residuals": {
            "poisson": summarize(rows, "poisson"),
            "hawkes": summarize(rows, "hawkes"),
            "hawkes_jitter": summarize(rows, "hawkes_jitter"),
            "control_simulated_continuous": control["continuous"],
            "control_simulated_grid_jitter": control["grid_jitter"],
        },
        "hawkes_vs_poisson": {
            "lr_reject_poisson_at_1pct": float(np.mean(lr_pvalues < 0.01)),
            "delta_aic_median": float(np.median([row["delta_aic"] for row in rows])),
            "aic_prefers_hawkes": float(np.mean([row["delta_aic"] < 0 for row in rows])),
        },
        "jitter_effect_on_mle": {
            name: {
                "median_original": float(np.median(original_values[name])),
                "median_jitter": float(np.median(jitter_values[name])),
            }
            for name in PARAMETER_NAMES
        },
        "branching_ratio_median": float(np.median([row["branching_ratio"] for row in rows])),
    }
    summary["figures"] = plot(pooled, summary["residuals"], args.figures)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"summary": summary, "windows": rows}, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
