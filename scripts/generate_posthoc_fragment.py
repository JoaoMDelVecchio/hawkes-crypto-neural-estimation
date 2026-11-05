#!/usr/bin/env python
"""Gera o fragmento LaTeX da análise post-hoc a partir de ``posthoc_sintetico.json``.

``--lang pt`` (padrão) escreve ``thesis/latex/posthoc_results.tex``; ``--lang en`` escreve a
versão em inglês em ``thesis/latex/en/posthoc_results.tex`` (ponto decimal, texto em
inglês, figuras ``*_en.pdf``). Os ``\\label`` são os mesmos nas duas línguas.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PARAMETERS = ("mu", "alpha_self", "alpha_cross", "beta")
LANGUAGES = ("pt", "en")
REGIMES = {
    "pt": (("low", "$g$ baixo"), ("intermediate", "$g$ intermediário"),
           ("near_critical", "$g$ quase crítico")),
    "en": (("low", "low $g$"), ("intermediate", "intermediate $g$"),
           ("near_critical", "near-critical $g$")),
}
ESTIMATORS = {
    "pt": (
        ("Média a priori", "Média a priori"),
        ("MLE", "MLE"),
        ("LSTM (ensemble)", "LSTM (ensemble)"),
        ("Transformer (ensemble)", "Transformer (ensemble)"),
    ),
    "en": (
        ("Média a priori", "Prior mean"),
        ("MLE", "MLE"),
        ("LSTM (ensemble)", "LSTM (ensemble)"),
        ("Transformer (ensemble)", "Transformer (ensemble)"),
    ),
}
DEFAULT_OUTPUTS = {
    "pt": Path("thesis/latex/posthoc_results.tex"),
    "en": Path("thesis/latex/en/posthoc_results.tex"),
}
FIGURE_SUFFIX = {"pt": "", "en": "_en"}


def number(value: float, digits: int = 3, lang: str = "pt") -> str:
    text = f"{value:.{digits}f}"
    return text.replace(".", ",") if lang == "pt" else text


def cell(value: float, digits: int = 3, lang: str = "pt") -> str:
    """Número para célula de tabela, com sinal de menos tipográfico."""
    text = number(abs(value), digits, lang)
    return f"$-${text}" if value < 0 and text.strip("0,.") else text


def render(report: dict, lang: str = "pt") -> str:
    if lang not in LANGUAGES:
        raise ValueError(f"Língua não suportada: {lang}")
    en = lang == "en"

    def num(value: float, digits: int = 3) -> str:
        return number(value, digits, lang)

    def cel(value: float, digits: int = 3) -> str:
        return cell(value, digits, lang)

    metrics = report["metrics"]
    rows_r2 = []
    for key, label in ESTIMATORS[lang]:
        block = metrics[key]["overall"]
        values = [block["per_parameter"][name]["r2_vs_prior"] for name in PARAMETERS]
        rows_r2.append(
            f"{label} & " + " & ".join(cel(v, 2) for v in values)
            + f" & {num(block['macro_median_ae'], 4)} & {num(block['g_mse'], 4)}\\\\"
        )

    paired = report["paired_trajectory_level"]
    rows_paired = []
    names = {
        "LSTM_ens vs Transformer_ens": "LSTM $\\times$ Transformer",
        "LSTM_ens vs MLE": "LSTM $\\times$ MLE",
        "Transformer_ens vs MLE": "Transformer $\\times$ MLE",
    }
    overall_label = "all" if en else "todos"
    separator = ", " if en else "; "
    for key, label in names.items():
        for regime, regime_label in (("overall", overall_label), *REGIMES[lang]):
            item = paired[key][regime]
            low, high = item["bootstrap_95ci"]
            rows_paired.append(
                f"{label} & {regime_label} & {num(item['median_loss_a'])} & "
                f"{num(item['median_loss_b'])} & {num(100 * item['win_rate_a'], 1)}\\% & "
                f"[{cel(low)}{separator}{cel(high)}]\\\\"
            )
        rows_paired.append(r"\midrule")
    rows_paired.pop()

    diagnostics = report["mle_diagnostics"]
    likelihood = diagnostics["likelihood_check"]
    refit = diagnostics["refit_20_starts_on_worst"]
    seed_level = report["seed_level_lstm_vs_transformer"]
    calibration = report.get("calibration_vs_prior")
    inside = calibration["fraction_inside_prior"] if calibration else None
    quant = calibration["quantiles"] if calibration else None
    figures = "../build/figures/"
    suffix = FIGURE_SUFFIX[lang]
    worst_share = num(100 * diagnostics["share_of_beta_sse_from_worst_1pct"], 1)
    ll_gain = num(likelihood["median_ll_gain_worst"], 2)
    welch_p = num(seed_level["welch_p"], 2)
    mann_whitney_p = num(seed_level["mann_whitney_p"], 3)

    if en:
        intro = [
            r"\section{Complementary analysis of the synthetic results}",
            r"\label{sec:posthoc}",
            "",
            r"The per-path predictions were recomputed from the frozen checkpoints",
            r"(maximum relative difference of $10^{-8}$ with respect to the reported MSE),",
            r"without retraining and without changing the protocol. Because the parameters",
            r"have different scales, each squared error is divided by the variance of the",
            r"parameter in the training set; the complementary result is the $R^2$ relative",
            r"to the trivial estimator that always predicts the prior mean.",
        ]
        r2_caption = [
            r"\caption{Per-parameter $R^2$ relative to the prior mean, macro median",
            r"absolute error and MSE of the branching ratio on the synthetic test set.}",
        ]
        r2_header = (
            r"Estimator & $\mu$ & $\alpha_{\mathrm{self}}$ & $\alpha_{\mathrm{cross}}$ & $\beta$"
            r" & Median AE & MSE of $g$\\"
        )
        paired_caption = [
            r"\caption{Paired comparison by path (mean standardized squared error over",
            r"the four parameters). Columns: median loss of A and of B, fraction of paths",
            r"in which A has the smaller error, and \emph{bootstrap} 95\% CI of the mean",
            r"difference A$-$B (negative favors A).}",
        ]
        paired_header = (
            r"A $\times$ B & Regime & Med.\ A & Med.\ B & A better & 95\% CI (A$-$B)\\"
        )
        discussion = [
            r"First, the networks' advantage",
            r"over the MLE in aggregate MSE comes from a few paths: "
            rf"{worst_share}\% of the sum",
            r"of squared errors of the MLE's $\beta$ is concentrated in 1\% of the paths,",
            r"almost all of them in the low-$g$ regime. In all of the "
            rf"{likelihood['worst_n']} worst paths, the",
            r"log-likelihood at the estimate exceeds that at the true parameter (median gain",
            rf"of {ll_gain}), and refitting {refit['n']} of them with 20 starts barely changes",
            r"the result. This is not a numerical failure: with weak excitation, $\beta$ is",
            r"weakly identified in 1,000 events. Second, on the typical path the MLE is as",
            r"good as the networks or better: its median loss is practically equal to the",
            r"LSTM's overall and lower in the intermediate and near-critical regimes. Third,",
            r"in the paired comparison the LSTM ensemble beats the Transformer ensemble in",
            r"every regime; across individual seeds, however, the difference is not",
            rf"significant (Welch, $p={welch_p}$; Mann--Whitney, $p={mann_whitney_p}$),",
            r"because the LSTM showed two levels of convergence across seeds.",
        ]
        beta_caption = [
            r"\caption{True versus estimated $\beta$ (3,000 test paths, logarithmic",
            r"scale). The networks shrink the estimates toward the center of the prior;",
            r"the MLE is unbiased over most of the domain but produces extreme values when",
            r"$g$ is low.}",
        ]
    else:
        intro = [
            r"\section{Análise complementar dos resultados sintéticos}",
            r"\label{sec:posthoc}",
            "",
            r"As previsões por trajetória foram recalculadas a partir dos checkpoints",
            r"congelados (diferença relativa máxima de $10^{-8}$ em relação ao MSE reportado),",
            r"sem retreino e sem alterar o protocolo. Como os parâmetros têm escalas",
            r"diferentes, cada erro quadrático é dividido pela variância do parâmetro no",
            r"conjunto de treino; o resultado complementar é o $R^2$ em relação ao",
            r"estimador trivial que sempre prevê a média a priori.",
        ]
        r2_caption = [
            r"\caption{$R^2$ por parâmetro em relação à média a priori, erro absoluto",
            r"mediano macro e MSE da razão de ramificação no teste sintético.}",
        ]
        r2_header = (
            r"Estimador & $\mu$ & $\alpha_{\mathrm{self}}$ & $\alpha_{\mathrm{cross}}$ & $\beta$"
            r" & EA mediano & MSE de $g$\\"
        )
        paired_caption = [
            r"\caption{Comparação pareada por trajetória (erro quadrático padronizado médio",
            r"nos quatro parâmetros). Colunas: perda mediana de A e de B, fração de",
            r"trajetórias em que A erra menos e IC 95\% \emph{bootstrap} da diferença média",
            r"A$-$B (negativo favorece A).}",
        ]
        paired_header = (
            r"A $\times$ B & Regime & Med.\ A & Med.\ B & A melhor & IC 95\% (A$-$B)\\"
        )
        discussion = [
            r"Primeiro, a vantagem das redes sobre",
            r"o MLE no MSE agregado vem de poucas trajetórias: "
            rf"{worst_share}\% da soma",
            r"dos erros quadráticos de $\beta$ do MLE se concentra em 1\% das trajetórias,",
            r"quase todas no regime de $g$ baixo. Em todas as "
            rf"{likelihood['worst_n']} piores, a log-verossimilhança",
            r"no estimador supera a do parâmetro verdadeiro (ganho mediano de "
            rf"{ll_gain}), e reajustar",
            rf"{refit['n']} delas com 20 inícios praticamente não altera o resultado. Não é",
            r"falha numérica: com excitação fraca, $\beta$ é pouco identificado em 1.000",
            r"eventos. Segundo, na trajetória típica o MLE é tão bom quanto as redes ou",
            r"melhor: a perda mediana é praticamente igual à da LSTM no agregado e menor nos",
            r"regimes intermediário e quase crítico. Terceiro, na comparação pareada o",
            r"ensemble LSTM supera o ensemble Transformer em todos os regimes; entre seeds",
            r"isoladas, contudo, a diferença não é significativa (Welch, "
            rf"$p={welch_p}$;",
            rf"Mann--Whitney, $p={mann_whitney_p}$), porque a LSTM",
            r"apresentou dois níveis de convergência entre seeds.",
        ]
        beta_caption = [
            r"\caption{$\beta$ verdadeiro versus estimado (3.000 trajetórias de teste,",
            r"escala logarítmica). As redes encolhem as estimativas para o centro do",
            r"prior; o MLE é não viesado na maior parte do domínio, mas produz valores",
            r"extremos quando $g$ é baixo.}",
        ]

    lines = [
        *intro,
        "",
        r"\begin{table}[htbp]",
        r"\centering",
        *r2_caption,
        r"\label{tab:posthoc-r2}",
        r"\small",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r2_header,
        r"\midrule",
        *rows_r2,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
        r"\begin{table}[htbp]",
        r"\centering",
        *paired_caption,
        r"\label{tab:posthoc-pareado}",
        r"\footnotesize",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        paired_header,
        r"\midrule",
        *rows_paired,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
        *discussion,
        "",
        r"\begin{figure}[htbp]",
        r"\centering",
        rf"\includegraphics[width=0.95\linewidth]{{{figures}"
        rf"posthoc_beta_verdadeiro_vs_estimado{suffix}.pdf}}",
        *beta_caption,
        r"\label{fig:posthoc-beta}",
        r"\end{figure}",
        "",
    ]
    if calibration:
        if en:
            calibration_text = [
                r"\section{Empirical calibration and support of the synthetic prior}",
                r"\label{sec:calibracao}",
                "",
                rf"The MLE was fitted on the {calibration['windows']:,} windows of 1,000 events",
                r"in the \texttt{calibration\_train} block (15 to 22 January 2024), the only",
                r"block authorized for calibration. The median $\beta$ was "
                rf"{num(quant['beta']['q50'], 2)} (interquartile",
                rf"range {num(quant['beta']['q25'], 2)}--{num(quant['beta']['q75'], 2)}),"
                r" above the upper limit of 2.0 used in training, and",
                r"$\alpha_{\mathrm{cross}}$ was below $10^{-3}$ in "
                rf"{num(100 * calibration['frac_alpha_cross_below_1e-3'], 1)}\% of the windows.",
                rf"Only {num(100 * inside['beta'], 1)}\% of the windows have $\beta$ inside the",
                rf"prior, {num(100 * inside['self_fraction'], 1)}\% have the self-excited"
                r" fraction",
                rf"inside $[0.2,\,0.8]$, and {num(100 * inside['all'], 1)}\% satisfy all the"
                r" ranges at once.",
                r"Because an estimator trained with quadratic loss approximates the posterior",
                r"mean under the training prior, the current networks cannot produce estimates",
                r"in this region; the empirical evaluation requires recalibrating the prior or",
                r"training on MLE labels from real windows.",
            ]
            calibration_caption = [
                r"\caption{MLE estimates on the real calibration windows (points) and support",
                r"of the prior used to generate the training data (rectangles).}",
            ]
        else:
            calibration_text = [
                r"\section{Calibração empírica e suporte do prior sintético}",
                r"\label{sec:calibracao}",
                "",
                rf"O MLE foi ajustado nas {calibration['windows']} janelas de 1.000 eventos do",
                r"bloco \texttt{calibration\_train} (15 a 22 de janeiro de 2024), único bloco",
                r"autorizado para calibração. A mediana de $\beta$ foi "
                rf"{num(quant['beta']['q50'], 2)} (intervalo",
                rf"interquartil {num(quant['beta']['q25'], 2)}--{num(quant['beta']['q75'], 2)}),"
                r" acima do limite de 2,0 usado no treino, e",
                rf"$\alpha_{{\mathrm{{cross}}}}$ ficou abaixo de $10^{{-3}}$ em "
                rf"{num(100 * calibration['frac_alpha_cross_below_1e-3'], 1)}\% das janelas.",
                rf"Apenas {num(100 * inside['beta'], 1)}\% das janelas têm $\beta$ dentro do",
                rf"prior, {num(100 * inside['self_fraction'], 1)}\% têm a fração autoexcitada",
                rf"dentro de $[0{{,}}2;\,0{{,}}8]$ e {num(100 * inside['all'], 1)}\% satisfazem",
                r"todas as faixas ao mesmo tempo. Como um estimador treinado com perda",
                r"quadrática aproxima a média a posteriori sob o prior de treino, as redes",
                r"atuais não podem produzir estimativas nessa região; a avaliação empírica",
                r"exige recalibrar o prior ou treinar com rótulos MLE em janelas reais.",
            ]
            calibration_caption = [
                r"\caption{Estimativas MLE nas janelas reais de calibração (pontos) e suporte",
                r"do prior usado para gerar os dados de treino (retângulos).}",
            ]
        lines += [
            *calibration_text,
            "",
            r"\begin{figure}[htbp]",
            r"\centering",
            rf"\includegraphics[width=0.95\linewidth]{{{figures}"
            rf"posthoc_calibracao_vs_prior{suffix}.pdf}}",
            *calibration_caption,
            r"\label{fig:calibracao-prior}",
            r"\end{figure}",
            "",
        ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--lang", choices=LANGUAGES, default="pt")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="padrão: thesis/latex/posthoc_results.tex (pt) ou thesis/latex/en/... (en)",
    )
    args = parser.parse_args()
    output = args.output or DEFAULT_OUTPUTS[args.lang]
    report = json.loads(args.report.read_text(encoding="utf-8"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(report, args.lang), encoding="utf-8")


if __name__ == "__main__":
    main()
