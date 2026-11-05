#!/usr/bin/env python
"""Critério C4: volatilidade de Hawkes contra *realized volatility*, janela a janela.

Para o Hawkes 2D simétrico com núcleos exponenciais, normas a_s = α_self/β e
a_c = α_cross/β e intensidade estacionária Λ = μ/(1 − a_s − a_c) por dimensão, o preço
X = Σ s_k J_k (s = +1 em alta, −1 em baixa; J = |retorno log| do evento, marcas i.i.d.
independentes da intensidade) tem variância de longo prazo por unidade de tempo
(Bacry et al., 2013, com marcas):

    σ²_H = 2Λ [Var(J) + E[J]² / (1 − a_s + a_c)²].

O benchmark é a variância realizada no mesmo intervalo, amostrando o log do mid-price a cada
Δ segundos (Andersen et al., 2003). Com Δ = 0,1 s soma-se o quadrado de todos os eventos; Δ
maior incorpora a persistência de direção (gráfico de assinatura). Mede-se a razão
σ_H/σ_RV(Δ) e a correlação de Spearman entre janelas.

Os parâmetros vêm do MLE de cada janela e, com ``--variant``, também do ensemble das redes
treinadas no escopo A. ``test_closed`` só é lido com ``--open-test`` (etapa 8).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy import stats  # noqa: E402

from hawkes_crypto.config import load_experiment_config  # noqa: E402
from hawkes_crypto.empirical import PARAMETER_NAMES, window_events  # noqa: E402
from hawkes_crypto.training import build_model, selected_device  # noqa: E402

SAMPLING = (0.1, 1.0, 5.0, 10.0, 30.0)


def hawkes_variance_rate(params: np.ndarray, mark_mean: float, mark_var: float) -> float:
    mu, alpha_self, alpha_cross, beta = params
    a_s, a_c = alpha_self / beta, alpha_cross / beta
    intensity = mu / (1.0 - a_s - a_c)
    return 2.0 * intensity * (mark_var + mark_mean**2 / (1.0 - a_s + a_c) ** 2)


def realized_variance(seconds: np.ndarray, log_price: np.ndarray, step: float) -> float:
    """Soma dos quadrados dos retornos do último preço observado em cada múltiplo de ``step``."""
    grid = np.arange(seconds[0], seconds[-1] + 1e-9, step)
    index = np.searchsorted(seconds, grid + 1e-9, side="right") - 1
    return float(np.sum(np.diff(log_price[index]) ** 2))


def network_predictions(
    variant: str, models_dir: Path, config_path: Path, inputs: np.ndarray, seeds: list[int]
) -> np.ndarray:
    name, init = variant.split("/")
    config = load_experiment_config(config_path)
    device = selected_device()
    outputs = []
    for seed in seeds:
        model = build_model(name, config).to(device)
        path = models_dir / name / init / f"seed_{seed}" / "best.pt"
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval()
        with torch.inference_mode():
            batches = [
                model(torch.from_numpy(inputs[i : i + 64]).to(device)).cpu().numpy()
                for i in range(0, len(inputs), 64)
            ]
        outputs.append(np.concatenate(batches))
    return np.mean(outputs, axis=0)


def compare(vol_hawkes: np.ndarray, rv: dict[float, np.ndarray]) -> dict[str, dict]:
    out = {}
    for step, values in rv.items():
        realized = np.sqrt(values)
        ratio = vol_hawkes / realized
        out[f"{step:g}s"] = {
            "median_ratio": float(np.median(ratio)),
            "ratio_iqr": [float(np.quantile(ratio, 0.25)), float(np.quantile(ratio, 0.75))],
            "spearman": float(stats.spearmanr(vol_hawkes, realized).statistic),
            "pearson_log": float(np.corrcoef(np.log(vol_hawkes), np.log(realized))[0, 1]),
        }
    return out


def plot(rows: pd.DataFrame, estimators: list[str], path: Path) -> None:
    """Figura em inglês, como o texto do TG."""
    figure, (left, right) = plt.subplots(1, 2, figsize=(9.2, 3.6))
    steps = np.asarray(SAMPLING)
    root_time = np.sqrt(rows["duration"].to_numpy())  # volatilidade por √s
    median_rv = [np.median(np.sqrt(rows[f"rv_{s:g}"]) / root_time) for s in steps]
    left.plot(steps, median_rv, "o-", color="#4c72b0", label="Realized, sampled every Δ")
    colors = {"MLE": "#dd8452", "Network": "#55a868"}
    for name in estimators:
        level = np.median(rows[f"vol_{name}"] / root_time)
        left.axhline(level, color=colors.get(name, "#8c8c8c"), linestyle="--",
                     label=f"Hawkes, {name}")
    left.set_xscale("log")
    left.set_xlabel("Sampling interval Δ (s)")
    left.set_ylabel(r"Median volatility per $\sqrt{s}$")
    left.set_title("Signature plot")
    left.legend(frameon=False, fontsize=8)
    for name in estimators:
        right.scatter(np.sqrt(rows["rv_10"]), rows[f"vol_{name}"], s=5, alpha=0.4,
                      color=colors.get(name, "#8c8c8c"), label=f"Hawkes, {name}")
    lims = [min(right.get_xlim()[0], right.get_ylim()[0]),
            max(right.get_xlim()[1], right.get_ylim()[1])]
    right.plot(lims, lims, color="black", linewidth=0.8)
    right.set_xscale("log")
    right.set_yscale("log")
    for axis in (right.xaxis, right.yaxis):
        axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    right.set_xlabel("Realized volatility, Δ = 10 s")
    right.set_ylabel("Hawkes volatility")
    right.set_title("Per window")
    right.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(path.with_suffix(f".{suffix}"), dpi=200)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels", type=Path,
        default=Path("artifacts/empirical_mle/calibration_validation.json"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/um_futures"))
    parser.add_argument("--splits", nargs="+", default=["calibration_train", "validation"])
    parser.add_argument("--variant", default=None, help="ex.: lstm_lee/finetune")
    parser.add_argument("--models-dir", type=Path,
                        default=Path("artifacts/empirical_models/scope_a"))
    parser.add_argument("--config", type=Path, default=Path("configs/lee_local_optimized.yaml"))
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/empirical_volatility/volatility.json"))
    parser.add_argument("--figure", type=Path,
                        default=Path("thesis/build/figures/hawkes_volatility.pdf"))
    parser.add_argument("--open-test", action="store_true")
    args = parser.parse_args()
    if "test_closed" in args.splits and not args.open_test:
        raise SystemExit("Recusado: use --open-test para ler test_closed")
    payload = json.loads(args.labels.read_text())
    records = [
        r for r in payload["records"]
        if r["split"] in args.splits and r.get("converged") and int(r["length"]) == 1000
    ]

    rows, inputs = [], []
    for record in records:
        events = window_events(
            record, args.data_root,
            columns=["inter_arrival", "event_type", "mid_price", "timestamp"],
        )
        seconds = events["inter_arrival"].cumsum().to_numpy(np.float64)
        log_price = np.log(events["mid_price"].to_numpy(np.float64))
        marks = np.abs(np.diff(log_price))
        row = {
            "split": record["split"], "symbol": record["symbol"], "date": record["date"],
            "start_event": record["start_event"],
            "hour_utc": int(pd.Timestamp(events["timestamp"].iloc[0]).hour),
            "duration": float(seconds[-1] - seconds[0]),
            "mark_mean": float(marks.mean()), "mark_var": float(marks.var()),
            "params_MLE": [float(record["parameters"][n]) for n in PARAMETER_NAMES],
        }
        for step in SAMPLING:
            row[f"rv_{step:g}"] = realized_variance(seconds, log_price, step)
        rows.append(row)
        inputs.append(events[["inter_arrival", "event_type"]].to_numpy(np.float32))

    estimators = ["MLE"]
    if args.variant:
        seeds = list(load_experiment_config(args.config).training.seeds)
        predicted = network_predictions(args.variant, args.models_dir, args.config,
                                        np.stack(inputs), seeds)
        for row, params in zip(rows, predicted, strict=True):
            row["params_Network"] = [float(v) for v in params]
        estimators.append("Network")
    for row in rows:
        for name in estimators:
            rate = hawkes_variance_rate(np.asarray(row[f"params_{name}"]), row["mark_mean"],
                                        row["mark_var"])
            row[f"vol_{name}"] = float(np.sqrt(rate * row["duration"]))

    frame = pd.DataFrame(rows)
    summary: dict[str, object] = {
        "splits": args.splits,
        "windows": len(frame),
        "variant": args.variant,
        "formula": "sigma2 = 2*Lambda*(Var(J) + E[J]^2/(1 - a_s + a_c)^2)",
        "by_estimator": {},
    }
    for name in estimators:
        vol = frame[f"vol_{name}"].to_numpy()
        per_split = {
            split: compare(vol[(frame["split"] == split).to_numpy()],
                           {s: frame.loc[frame["split"] == split, f"rv_{s:g}"].to_numpy()
                            for s in SAMPLING})
            for split in args.splits
        }
        # Estabilidade: variação relativa entre janelas consecutivas do mesmo dia e símbolo.
        ordered = frame.sort_values(["symbol", "date", "start_event"])
        jumps = ordered.groupby(["symbol", "date"])[f"vol_{name}"].apply(
            lambda s: np.abs(np.diff(np.log(s.to_numpy())))
        )
        rv_jumps = ordered.groupby(["symbol", "date"])["rv_10"].apply(
            lambda s: np.abs(np.diff(0.5 * np.log(s.to_numpy())))
        )
        summary["by_estimator"][name] = {
            "per_split": per_split,
            "finite_and_subcritical": float(np.mean(np.isfinite(vol) & (vol > 0))),
            "median_abs_log_change_consecutive": float(np.median(np.concatenate(jumps.values))),
            "median_abs_log_change_consecutive_rv10": float(
                np.median(np.concatenate(rv_jumps.values))
            ),
        }
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    plot(frame, estimators, args.figure)
    summary["figure"] = str(args.figure)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"summary": summary, "windows": frame.to_dict(orient="records")}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
