#!/usr/bin/env python
"""Gera o fragmento LaTeX (em inglês) da etapa empírica: diagnóstico e escopo A.

Lê ``diagnostico_jan15-22.json`` (``make diagnostics``) e, se existirem, o ``summary.json``
do escopo A (``make empirical-models``) e o ``summary_test.json`` (etapa 8). Escreve
``thesis/latex/empirical_results.tex``; nenhum número do texto é digitado à mão.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

VARIANTS = (
    ("label_mean", "Training-label mean"),
    ("lstm_lee/zero_shot", "LSTM, synthetic weights"),
    ("transformer/zero_shot", "Transformer, synthetic weights"),
    ("lstm_lee/scratch", "LSTM, trained from scratch"),
    ("lstm_lee/finetune", "LSTM, fine-tuned"),
    ("transformer/scratch", "Transformer, trained from scratch"),
    ("transformer/finetune", "Transformer, fine-tuned"),
)
DIAGNOSTIC_ROWS = (
    ("poisson", "Poisson, real data"),
    ("hawkes", "Hawkes, real data"),
    ("hawkes_jitter", "Hawkes, real data, jittered grid"),
    ("control_simulated_grid_jitter", "Hawkes, simulated on the grid, jittered"),
    ("control_simulated_continuous", "Hawkes, simulated in continuous time"),
)


def pct(value: float) -> str:
    return f"{100 * value:.1f}\\%"


def num(value: float, digits: int = 3) -> str:
    text = f"{abs(value):.{digits}f}"
    return f"$-${text}" if value < 0 and text.strip("0.") else text


def pvalue(value: float) -> str:
    if value < 1e-3:
        exponent = int(f"{value:.0e}".split("e")[1])
        return f"$<10^{{{exponent + 1}}}$"
    return f"{value:.3f}"


def diagnostics_section(report: dict, test: dict | None = None) -> str:
    s = report["summary"]
    res = s["residuals"]
    rows = "\n".join(
        f"{label} & {pct(res[key]['ks_reject_per_dimension'])} & "
        f"{pvalue(res[key]['ks_median_pvalue'])} & "
        f"{pct(res[key]['ljung_box_reject_per_dimension'])}\\\\"
        for key, label in DIAGNOSTIC_ROWS
    )
    versus = s["hawkes_vs_poisson"]
    jitter = s["jitter_effect_on_mle"]
    return f"""\\section{{Goodness of fit on real data}}
\\label{{sec:real-diagnostics}}

The diagnostic uses only the calibration block ({s['windows']:,} windows of 1,000 events,
15--22 January 2024), with the MLE already fitted to each window. Under a correctly specified
model, the time-rescaled residuals of each dimension are i.i.d.\\ $\\mathrm{{Exp}}(1)$
\\citep{{ogata1988,brown2002}}; they are tested with Kolmogorov--Smirnov (KS) against
$\\mathrm{{Exp}}(1)$ and with Ljung--Box (10 lags) for serial dependence. Because the data are
observed on a 100~ms grid, two controls separate grid effects from misspecification: real
times jittered uniformly within their cell (with the MLE refitted), and paths simulated from
each window's fitted Hawkes, either kept in continuous time or mapped to the grid (one event per
cell) and jittered in the same way.

\\begin{{table}}[htbp]
\\centering
\\caption{{Residual diagnostics per window and dimension. Under a correct model, about 5\\%
of the tests reject at the 5\\% level.}}
\\label{{tab:real-diagnostics}}
\\small
\\begin{{tabular}}{{lrrr}}
\\toprule
Model and data & KS rejects & Median KS $p$ & Ljung--Box rejects\\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=\\linewidth]{{real_data_residual_diagnostics.pdf}}
\\caption{{Left: Q--Q plot of the pooled time-rescaled residuals against $\\mathrm{{Exp}}(1)$.
Right: rejection rates on real data and in the two simulated controls; the dotted line is the
nominal 5\\%.}}
\\label{{fig:real-diagnostics}}
\\end{{figure}}

The Hawkes model is far better than a homogeneous Poisson process: the likelihood-ratio test
rejects Poisson at 1\\% in {pct(versus['lr_reject_poisson_at_1pct'])} of the windows, and the
median AIC difference is {num(versus['delta_aic_median'], 0)} in favour of Hawkes. It is not,
however, the data-generating process. The grid alone makes KS reject in
{pct(res['control_simulated_grid_jitter']['ks_reject_per_dimension'])} of the simulated
controls, so the {pct(res['hawkes_jitter']['ks_reject_per_dimension'])} rejection on real data
is only partly misspecification. Ljung--Box, on the other hand, keeps its nominal size on the
grid ({pct(res['control_simulated_grid_jitter']['ljung_box_reject_per_dimension'])}) and
rejects in {pct(res['hawkes_jitter']['ljung_box_reject_per_dimension'])} of the real windows:
part of the clustering is left in the residuals, which a single exponential kernel cannot
absorb. Jittering barely moves the MLE (median $\\beta$ from
{jitter['beta']['median_original']:.2f} to {jitter['beta']['median_jitter']:.2f}; median
$\\alpha_{{\\mathrm{{self}}}}$ from {jitter['alpha_self']['median_original']:.2f} to
{jitter['alpha_self']['median_jitter']:.2f}). On real data, therefore, the target of a neural
estimator is the MLE of this approximating model, not a true parameter.
""" + (test_diagnostics(test["summary"]) if test else "")


def test_diagnostics(s: dict) -> str:
    res = s["residuals"]
    return f"""
The same diagnostic was repeated once on the {s['windows']:,} test windows (26--28 January
2024), after the test block was opened. The pattern is the same: the likelihood ratio rejects
Poisson in {pct(s['hawkes_vs_poisson']['lr_reject_poisson_at_1pct'])} of the windows (median
AIC difference {num(s['hawkes_vs_poisson']['delta_aic_median'], 0)}), and Ljung--Box rejects in
{pct(res['hawkes_jitter']['ljung_box_reject_per_dimension'])} of the jittered real windows,
against {pct(res['control_simulated_grid_jitter']['ljung_box_reject_per_dimension'])} in the
simulated controls on the grid.
"""


def r2(variant: dict, baseline: dict, name: str) -> float:
    """Fração da variância dos rótulos reproduzida, relativa à média dos rótulos de treino."""
    return 1.0 - (
        variant["ensemble"]["per_parameter"][name]["mse"]
        / baseline["ensemble"]["per_parameter"][name]["mse"]
    )


def models_section(summary: dict, test: dict | None) -> str:
    windows = summary["windows"]
    variants = summary["variants"]
    selected = summary["selected_variant"]
    has_test = test is not None
    lines = []
    for key, label in VARIANTS:
        if key not in variants:
            continue
        val = variants[key]["validation"]
        base = variants["label_mean"]["validation"]
        seeds = val.get("seeds")
        spread = (
            f"{num(seeds['macro_mse_mean'], 4)} $\\pm$ {num(seeds['macro_mse_std'], 4)}"
            if seeds else "--"
        )
        cells = [
            label + (" $^{\\ast}$" if key == selected else ""),
            num(val["ensemble"]["macro_mse"], 4),
            spread,
            *(num(r2(val, base, p), 2) for p in ("mu", "alpha_self", "beta")),
        ]
        if has_test:
            test_variant = test["variants"].get(key)
            cells.append(num(test_variant["test"]["ensemble"]["macro_mse"], 4)
                         if test_variant else "--")
        lines.append(" & ".join(cells) + "\\\\")
    test_header = " & Test" if has_test else ""
    header = (
        "Variant & Ensemble & Seeds & $R^2_{\\mu}$ & $R^2_{\\alpha_{\\mathrm{self}}}$ "
        "& $R^2_{\\beta}$"
    )
    columns = "lrrrrr" + ("r" if has_test else "")
    test_note = (
        "The test block was opened once, after the selection was frozen."
        if has_test else
        "The test block is still closed."
    )
    return f"""\\section{{Neural estimators on real windows}}
\\label{{sec:real-networks}}

Each real window is labelled with its MLE (five heuristic starts), and the networks learn to
reproduce that label. Training uses the calibration block ({windows['calibration_train']:,}
windows), selection uses the validation block ({windows['validation']:,} windows, 23--25
January 2024). Each variant is trained with the {len(summary['seeds'])} seeds of the synthetic
protocol, either from scratch or starting from the synthetic weights of the same seed
(fine-tuning, learning rate divided by 10); the synthetic networks without retraining and the
mean of the training labels are reference points. The selected variant is the one with the
lowest validation macro MSE of the {len(summary['seeds'])}-seed ensemble. {test_note}

\\begin{{table}}[htbp]
\\centering
\\caption{{Agreement with the MLE label on real windows: macro MSE of the seed ensemble and
mean $\\pm$ standard deviation across seeds on the validation block, and $R^2$ per parameter
relative to the training-label mean (share of the label variance reproduced).
$^{{\\ast}}$Selected variant.}}
\\label{{tab:real-networks}}
\\small
\\begin{{tabular}}{{{columns}}}
\\toprule
{header}{test_header}\\\\
\\midrule
{chr(10).join(lines)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def volatility_section(report: dict, test: dict | None) -> str:
    summary = report["summary"]
    blocks = [(split, summary) for split in summary["splits"]]
    if test is not None:
        blocks += [("test_closed", test["summary"])]
    names = {"calibration_train": "Calibration", "validation": "Validation",
             "test_closed": "Test"}
    estimators = list(summary["by_estimator"])
    lines = []
    for split, source in blocks:
        for estimator in estimators:
            per = source["by_estimator"].get(estimator, {}).get("per_split", {}).get(split)
            if per is None:
                continue
            cells = [names[split], "network" if estimator == "Network" else estimator]
            for step in ("1s", "10s", "30s"):
                cells += [f"{per[step]['median_ratio']:.2f}", f"{per[step]['spearman']:.2f}"]
            lines.append(" & ".join(cells) + "\\\\")
    mle = summary["by_estimator"]["MLE"]
    # Modelo sem f-string: as chaves do LaTeX ficam legíveis; os valores entram por replace.
    template = r"""\section{Hawkes volatility}
\label{sec:real-volatility}

For the symmetric model with exponential kernels, kernel norms
$a_s=\alpha_{\mathrm{self}}/\beta$ and $a_c=\alpha_{\mathrm{cross}}/\beta$, stationary
intensity $\Lambda=\mu/(1-a_s-a_c)$ per direction and i.i.d.\ jump sizes $J$ (absolute log
returns), the long-run variance of the price per unit time is
\begin{equation}
\sigma_H^2 = 2\Lambda\left[\operatorname{Var}(J)
  + \frac{\mathbb{E}[J]^2}{(1-a_s+a_c)^2}\right]
\end{equation}
\citep{bacry2013}. Each window's Hawkes volatility, $\sigma_H$ times the square root of
its duration, is compared with the realized volatility of the log mid-price sampled every
$\Delta$ seconds \citep{andersen2003}. The metrics were fixed before the test was opened:
the median ratio $\sigma_H/\sigma_{RV}$ and the Spearman correlation across windows, with
$\Delta=10$~s as the main horizon. The estimate is finite in @FINITE@ of the windows, and its
median absolute log change between consecutive windows (@STAB@) is close to that of the
realized volatility (@STABRV@).

\begin{table}[htbp]
\centering
\caption{Hawkes volatility against realized volatility: median ratio
$\sigma_H/\sigma_{RV}$ and Spearman correlation across windows, by sampling interval
$\Delta$.}
\label{tab:real-volatility}
\small
\begin{tabular}{llrrrrrr}
\toprule
 & & \multicolumn{2}{c}{$\Delta=1$ s} & \multicolumn{2}{c}{$\Delta=10$ s}
 & \multicolumn{2}{c}{$\Delta=30$ s}\\
Block & Parameters & Ratio & $\rho_S$ & Ratio & $\rho_S$ & Ratio & $\rho_S$\\
\midrule
@ROWS@
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{hawkes_volatility.pdf}
\caption{Left: median realized volatility per $\sqrt{\mathrm{s}}$ as a function of the
sampling interval (signature plot) and the median Hawkes level. Right: Hawkes against
realized volatility ($\Delta=10$~s) per window.}
\label{fig:real-volatility}
\end{figure}
"""
    return (
        template.replace("@FINITE@", pct(mle["finite_and_subcritical"]))
        .replace("@STAB@", f"{mle['median_abs_log_change_consecutive']:.3f}")
        .replace("@STABRV@", f"{mle['median_abs_log_change_consecutive_rv10']:.3f}")
        .replace("@ROWS@", "\n".join(lines))
    )


def robustness_section(report: dict) -> str:
    rows = []
    for key, label in VARIANTS:
        block = report["variants"].get(key)
        if block is None:
            continue
        overall = block["all"]
        thirds = block["by_volatility_tercile"]
        cells = [
            label + (r" $^{\ast}$" if key == report["selected_variant"] else ""),
            num(overall["macro_mse"], 4),
            *(num(thirds[t]["macro_mse"], 4) for t in ("low", "medium", "high")),
        ]
        rows.append(" & ".join(cells) + "\\\\")
    template = r"""\section{Robustness window}
\label{sec:real-robustness}

The week of 23--29 October 2023 was never used for training or selection. It is cut into
@WINDOWS@ non-overlapping windows of 1,000 events, labeled with the MLE (convergence in @CONV@ of
the windows), and every variant is evaluated with the weights stored before the test was
opened.

\begin{table}[htbp]
\centering
\caption{Agreement with the MLE label in the October 2023 robustness window: macro MSE of the
seed ensemble, overall and by tercile of realized volatility. $^{\ast}$Variant selected on
the January validation block.}
\label{tab:real-robustness}
\small
\begin{tabular}{lrrrr}
\toprule
Variant & All & Low vol. & Medium vol. & High vol.\\
\midrule
@ROWS@
\bottomrule
\end{tabular}
\end{table}
"""
    return (
        template.replace("@WINDOWS@", f"{report['windows']:,}")
        .replace("@CONV@", pct(report["mle_convergence_rate"]))
        .replace("@ROWS@", "\n".join(rows))
    )


def paired_section(report: dict) -> str:
    """Parágrafo da comparação pareada pós-teste (``compare_scope_a_finetune.py``)."""
    test, october = report["blocks"]["test_closed"], report["blocks"]["october_2023"]

    def ci(block: dict) -> str:
        low, high = block["bootstrap_95ci"]
        return f"[{num(low, 4)}, {num(high, 4)}]"

    def p_text(block: dict) -> str:
        value = pvalue(block["wilcoxon_p"])
        return f"$p$ {value}" if value.startswith("$") else f"$p={value}$"

    return f"""
The two fine-tuned networks were also compared window by window, as a post hoc analysis that
was not part of the pre-registered protocol: the selection had already been made on
validation. The loss of each window is the squared error averaged over the four parameters,
and the difference is LSTM minus Transformer. On the test block the difference is
{num(test['mean_difference_a_minus_b'], 4)}, with a paired bootstrap 95\\% interval of
{ci(test)} and Wilcoxon {p_text(test)}: the two variants cannot be told
apart. In the October window the difference is {num(october['mean_difference_a_minus_b'], 4)}
({ci(october)}, {p_text(october)}), and the Transformer has the lower loss in
{pct(1 - october['win_rate_a'])} of the windows: under a change of regime, the fine-tuned
Transformer degrades less than the selected LSTM.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--diagnostics",
        type=Path,
        default=Path("artifacts/empirical_diagnostics/diagnostico_jan15-22.json"),
    )
    parser.add_argument(
        "--diagnostics-test", type=Path,
        default=Path("artifacts/empirical_diagnostics/diagnostico_teste.json"),
    )
    parser.add_argument(
        "--models", type=Path, default=Path("artifacts/empirical_models/scope_a/summary.json")
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=Path("artifacts/empirical_models/scope_a/summary_test.json"),
    )
    parser.add_argument(
        "--volatility", type=Path,
        default=Path("artifacts/empirical_volatility/volatility.json"),
    )
    parser.add_argument(
        "--volatility-test", type=Path,
        default=Path("artifacts/empirical_volatility/volatility_test.json"),
    )
    parser.add_argument(
        "--robustness", type=Path,
        default=Path("artifacts/robustness/scope_a_october.json"),
    )
    parser.add_argument(
        "--paired", type=Path,
        default=Path("artifacts/empirical_models/scope_a/paired_finetune.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("thesis/latex/empirical_results.tex"))
    args = parser.parse_args()
    parts = [
        "% Gerado por scripts/generate_empirical_fragment.py; não editar à mão.",
        diagnostics_section(
            json.loads(args.diagnostics.read_text()),
            json.loads(args.diagnostics_test.read_text())
            if args.diagnostics_test.exists() else None,
        ),
    ]
    if args.models.exists():
        test = json.loads(args.test.read_text()) if args.test.exists() else None
        parts.append(models_section(json.loads(args.models.read_text()), test))
    if args.volatility.exists():
        test = (json.loads(args.volatility_test.read_text())
                if args.volatility_test.exists() else None)
        parts.append(volatility_section(json.loads(args.volatility.read_text()), test))
    if args.robustness.exists():
        parts.append(robustness_section(json.loads(args.robustness.read_text())))
    if args.paired.exists():
        parts.append(paired_section(json.loads(args.paired.read_text())))
    args.output.write_text("\n".join(parts), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
