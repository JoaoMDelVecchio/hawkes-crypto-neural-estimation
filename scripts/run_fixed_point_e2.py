#!/usr/bin/env python
"""Experimento E2: distribuição amostral no ponto fixo publicado por Lee (2023).

Simula trajetórias independentes com mu=0,3, alpha_self=0,4, alpha_cross=0,7 e
beta=1,5, aplica os checkpoints congelados (sem retreino) e o MLE em duas versões:
heurística (protocolo do projeto, sem acesso ao valor verdadeiro) e oráculo
(inicializada no valor verdadeiro, como no artigo). Reporta média e desvio-padrão
de cada estimador no formato da tabela de Lee (2023).

``--render-only`` apenas reescreve a tabela LaTeX a partir do ``summary.json`` já salvo,
sem simular, ajustar o MLE nem carregar checkpoints. ``--lang en`` escreve a versão em
inglês em ``thesis/latex/en/e2_fixed_point.tex`` (o ``\\label`` é o mesmo).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402

from hawkes_crypto.config import load_experiment_config  # noqa: E402
from hawkes_crypto.estimate.mle import fit_mle  # noqa: E402
from hawkes_crypto.simulate.hawkes_sim import HawkesParameters, simulate_path  # noqa: E402
from hawkes_crypto.training import build_model, selected_device  # noqa: E402

PARAMETER_NAMES = ("mu", "alpha_self", "alpha_cross", "beta")
FIXED_POINT = HawkesParameters(mu=0.3, alpha_self=0.4, alpha_cross=0.7, beta=1.5)
LEE_TABLE = {
    "Lee (2023) rede": {"mean": [0.3151, 0.4429, 0.5733, 1.5736],
                        "sd": [0.0358, 0.0661, 0.0697, 0.1364]},
    "Lee (2023) MLE": {"mean": [0.3036, 0.3988, 0.7024, 1.5078],
                       "sd": [0.0314, 0.0500, 0.0608, 0.1145]},
}
LANGUAGES = ("pt", "en")
DEFAULT_LATEX = {
    "pt": Path("thesis/latex/e2_fixed_point.tex"),
    "en": Path("thesis/latex/en/e2_fixed_point.tex"),
}
ROW_LABELS_EN = {
    "LSTM (ensemble de 5 seeds)": "LSTM (5-seed ensemble)",
    "Transformer (ensemble de 5 seeds)": "Transformer (5-seed ensemble)",
    "MLE heurístico (5 inícios)": "Heuristic MLE (5 starts)",
    "MLE oráculo (início no verdadeiro)": "Oracle MLE (started at the true value)",
    "Lee (2023) rede": "Lee (2023) network",
    "Lee (2023) MLE": "Lee (2023) MLE",
}


def simulate(task: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    length, seed = task
    path = simulate_path(FIXED_POINT, length, seed=seed)
    return path.inter_arrivals, path.event_types


def fit(task: tuple[np.ndarray, np.ndarray, bool, int, int]) -> list[float]:
    times, types, oracle, n_starts, seed = task
    result = fit_mle(
        times,
        types,
        initial=FIXED_POINT if oracle else None,
        n_starts=n_starts,
        seed=seed,
        maxiter=1000,
    )
    p = result.parameters
    return [p.mu, p.alpha_self, p.alpha_cross, p.beta]


def describe(estimates: np.ndarray) -> dict[str, object]:
    truth = np.asarray([FIXED_POINT.mu, FIXED_POINT.alpha_self, FIXED_POINT.alpha_cross,
                        FIXED_POINT.beta])
    return {
        "mean": estimates.mean(axis=0).tolist(),
        "sd": estimates.std(axis=0, ddof=1).tolist(),
        "rmse": np.sqrt(np.mean((estimates - truth) ** 2, axis=0)).tolist(),
        "macro_mse": float(np.mean((estimates - truth) ** 2)),
    }


def latex_table(
    rows: dict[str, dict[str, object]], paths: int, length: int, lang: str = "pt"
) -> str:
    en = lang == "en"

    def cell(value: float) -> str:
        text = f"{value:.4f}"
        return text if en else text.replace(".", ",")

    body = []
    for label, stats in rows.items():
        values = " & ".join(
            f"{cell(m)} & {cell(s)}" for m, s in zip(stats["mean"], stats["sd"], strict=True)
        )
        shown = ROW_LABELS_EN.get(label, label) if en else label
        body.append(f"{shown} & {values}\\\\")
    if en:
        caption = (
            r"\caption{Sampling distribution at the fixed point of \citet{lee2023} "
            r"($\mu=0.3$, $\alpha_{\mathrm{self}}=0.4$, "
            r"$\alpha_{\mathrm{cross}}=0.7$, $\beta=1.5$): mean and standard deviation "
            rf"over {paths:,} paths of {length:,} events. Lee's rows use "
            r"100,000 paths of 2,000 steps.}"
        )
        header = r"Estimator & Mean & SD & Mean & SD & Mean & SD & Mean & SD\\"
    else:
        paths_text = f"{paths:,}".replace(",", ".")
        length_text = f"{length:,}".replace(",", ".")
        caption = (
            r"\caption{Distribuição amostral no ponto fixo de \citet{lee2023} "
            r"($\mu=0{,}3$, $\alpha_{\mathrm{self}}=0{,}4$, "
            r"$\alpha_{\mathrm{cross}}=0{,}7$, $\beta=1{,}5$): média e desvio-padrão "
            rf"em {paths_text} trajetórias de {length_text} eventos. As linhas de Lee usam "
            r"100.000 trajetórias de 2.000 passos.}"
        )
        header = r"Estimador & Média & DP & Média & DP & Média & DP & Média & DP\\"
    return "\n".join(
        [
            r"\begin{table}[htbp]",
            r"\centering",
            caption,
            r"\label{tab:e2-ponto-fixo}",
            r"\scriptsize",
            r"\begin{tabular}{lrrrrrrrr}",
            r"\toprule",
            r" & \multicolumn{2}{c}{$\mu$} & \multicolumn{2}{c}{$\alpha_{\mathrm{self}}$}"
            r" & \multicolumn{2}{c}{$\alpha_{\mathrm{cross}}$} & \multicolumn{2}{c}{$\beta$}\\",
            header,
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def write_latex(
    rows: dict[str, dict[str, object]],
    lee_rows: dict[str, dict[str, object]],
    paths: int,
    length: int,
    output: Path,
    lang: str = "pt",
) -> None:
    table_rows = {**{k: v for k, v in rows.items() if "individual" not in k}, **lee_rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(latex_table(table_rows, paths, length, lang), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/lee_local_optimized.yaml"))
    parser.add_argument("--paths", type=int, default=2000)
    parser.add_argument("--length", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/fixed_point_e2/summary.json")
    )
    parser.add_argument(
        "--latex",
        type=Path,
        default=None,
        help="padrão: thesis/latex/e2_fixed_point.tex (pt) ou thesis/latex/en/... (en)",
    )
    parser.add_argument("--lang", choices=LANGUAGES, default="pt")
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="só reescreve a tabela LaTeX a partir de --output (summary.json já salvo)",
    )
    args = parser.parse_args()
    latex_path = args.latex or DEFAULT_LATEX[args.lang]
    if args.render_only:
        saved = json.loads(args.output.read_text(encoding="utf-8"))
        write_latex(saved["estimators"], saved["lee_2023_reported"], saved["paths"],
                    saved["length"], latex_path, args.lang)
        print(json.dumps({"latex": str(latex_path)}, indent=2))
        return
    config = load_experiment_config(args.config)
    base_seed = config.seed + 3_000_000
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        simulated = list(executor.map(
            simulate, [(args.length, base_seed + i) for i in range(args.paths)], chunksize=16
        ))
        inputs = np.stack([
            np.stack([inter, types.astype(np.float64)], axis=1) for inter, types in simulated
        ]).astype(np.float32)
        times = [np.cumsum(inputs[i, :, 0], dtype=np.float64) for i in range(args.paths)]
        types = [inputs[i, :, 1].astype(np.int8) for i in range(args.paths)]
        mle = {}
        for label, oracle, starts in (("MLE heurístico (5 inícios)", False, 5),
                                      ("MLE oráculo (início no verdadeiro)", True, 1)):
            tasks = [(times[i], types[i], oracle, starts, base_seed + i) for i in range(args.paths)]
            mle[label] = np.asarray(list(executor.map(fit, tasks, chunksize=8)))
    simulation_and_mle_seconds = time.perf_counter() - started

    device = selected_device("auto")
    neural: dict[str, np.ndarray] = {}
    for name, label in (("lstm_lee", "LSTM"), ("transformer", "Transformer")):
        per_seed = []
        for seed in config.training.seeds:
            checkpoint = (config.artifacts_root / f"length_{args.length}" / name
                          / f"seed_{seed}" / "best.pt")
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model = build_model(name, config)
            model.load_state_dict(payload["state_dict"])
            model.to(device).eval()
            batch = 256 if name == "lstm_lee" else 32
            outputs = []
            with torch.inference_mode():
                for start in range(0, args.paths, batch):
                    chunk = torch.from_numpy(inputs[start:start + batch]).to(device)
                    outputs.append(model(chunk).float().cpu().numpy())
            per_seed.append(np.concatenate(outputs).astype(np.float64))
        neural[f"{label} (ensemble de 5 seeds)"] = np.mean(per_seed, axis=0)
        neural[f"{label} (seed individual, média)"] = np.concatenate(per_seed)

    rows = {label: describe(values) for label, values in {**neural, **mle}.items()}
    payload = {
        "fixed_point": dict(zip(PARAMETER_NAMES, [0.3, 0.4, 0.7, 1.5], strict=True)),
        "paths": args.paths,
        "length": args.length,
        "seed": base_seed,
        "simulation_and_mle_seconds": simulation_and_mle_seconds,
        "estimators": rows,
        "lee_2023_reported": LEE_TABLE,
        "note": "Seed individual agrega as 5 seeds (5 x paths estimativas).",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_latex(rows, LEE_TABLE, args.paths, args.length, latex_path, args.lang)
    print(json.dumps({"output": str(args.output), "latex": str(latex_path)}, indent=2))


if __name__ == "__main__":
    main()
