#!/usr/bin/env python
"""Gera fragmento LaTeX final a partir do resumo consolidado.

``--lang pt`` (padrão) escreve ``thesis/latex/final_results.tex``; ``--lang en`` escreve a
versão em inglês em ``thesis/latex/en/final_results.tex`` (ponto decimal, legendas e texto
em inglês, figuras ``*_en.pdf``). Os ``\\label`` são os mesmos nas duas línguas.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scipy import stats

PARAMETERS = ("mu", "alpha_self", "alpha_cross", "beta")
PARAMETER_LABELS = {
    "mu": r"$\mu$",
    "alpha_self": r"$\alpha_{\mathrm{self}}$",
    "alpha_cross": r"$\alpha_{\mathrm{cross}}$",
    "beta": r"$\beta$",
}
REGIMES = ("low", "intermediate", "near_critical")
LANGUAGES = ("pt", "en")
DEFAULT_OUTPUTS = {
    "pt": Path("thesis/latex/final_results.tex"),
    "en": Path("thesis/latex/en/final_results.tex"),
}
FIGURE_SUFFIX = {"pt": "", "en": "_en"}


def _localize(text: str, lang: str) -> str:
    return text.replace(".", ",") if lang == "pt" else text


def decimal(value: float, lang: str = "pt") -> str:
    return _localize(f"{value:.6f}", lang)


def short_decimal(value: float, lang: str = "pt") -> str:
    return _localize(f"{value:.4f}", lang)


def compact_decimal(value: float, lang: str = "pt") -> str:
    return _localize(f"{value:.2f}", lang)


def percent(value: float, lang: str = "pt") -> str:
    return _localize(f"{100.0 * value:.2f}\\%", lang)


def percentage_points(value: float, lang: str = "pt") -> str:
    return _localize(f"{value:.1f}\\%", lang)


def seed_macro_mse(model: dict[str, object]) -> list[float]:
    return [float(run["evaluation"]["macro_mse"]) for run in model.get("runs", [])]


def network_sentence(lstm: float, transformer: float, lang: str = "pt") -> str:
    if lstm < transformer:
        gain = percentage_points(100.0 * (transformer - lstm) / transformer, lang)
        if lang == "en":
            return (
                f"Between the networks, the LSTM obtained a mean macro MSE {gain} lower "
                "than the Transformer."
            )
        return f"Entre as redes, a LSTM obteve MSE macro médio {gain} menor que o Transformer."
    if transformer < lstm:
        gain = percentage_points(100.0 * (lstm - transformer) / lstm, lang)
        if lang == "en":
            return (
                f"Between the networks, the Transformer obtained a mean macro MSE {gain} "
                "lower than the LSTM."
            )
        return f"Entre as redes, o Transformer obteve MSE macro médio {gain} menor que a LSTM."
    if lang == "en":
        return "The two networks obtained equivalent mean macro MSE."
    return "As duas redes obtiveram MSE macro médio equivalente."


def significance_sentence(lstm: list[float], transformer: list[float], lang: str = "pt") -> str:
    en = lang == "en"
    if len(lstm) < 2 or len(transformer) < 2:
        if en:
            return "Without per-seed results, the difference between the networks was not tested."
        return "Sem resultados por seed, a diferença entre as redes não foi testada."
    test = stats.ttest_ind(lstm, transformer, equal_var=False)
    p_value = compact_decimal(float(test.pvalue), lang)
    if test.pvalue < 0.05:
        if en:
            return (
                "The difference between the networks is statistically significant across "
                f"seeds (Welch test, $p={p_value}$)."
            )
        return (
            "A diferença entre as redes é estatisticamente significativa entre seeds "
            f"(teste de Welch, $p={p_value}$)."
        )
    if en:
        return (
            f"With {len(lstm)} and {len(transformer)} seeds, however, the difference is not "
            f"statistically significant (Welch test, $p={p_value}$), and the confidence "
            "intervals overlap; the data do not allow the networks to be ranked."
        )
    return (
        f"Com {len(lstm)} e {len(transformer)} seeds, porém, a diferença não é "
        f"estatisticamente significativa (teste de Welch, $p={p_value}$), e os "
        "intervalos de confiança se sobrepõem; os dados não permitem ordenar as redes."
    )


def render_fragment(summary: dict[str, object], lang: str = "pt") -> str:
    if lang not in LANGUAGES:
        raise ValueError(f"Língua não suportada: {lang}")
    en = lang == "en"
    if summary.get("missing"):
        raise RuntimeError("Resumo ainda possui jobs neurais ausentes")
    lengths = summary["lengths"]
    if set(lengths) != {"1000"}:
        raise ValueError("O relatório final exige somente o protocolo de 1.000 eventos")
    result = lengths["1000"]
    models = result["models"]
    mle = result["mle"]
    lstm = models["lstm_lee"]
    transformer = models["transformer"]
    estimators = (
        ("LSTM", lstm, True),
        ("Transformer", transformer, True),
        ("MLE", mle, False),
    )
    mle_beta_error_share = (
        100.0
        * mle["per_parameter"]["beta"]["mse"]
        / (4.0 * mle["macro_mse"])
    )
    mle_low_regime_ratio = (
        mle["by_branching_ratio"]["low"]["macro_mse"]
        / mle["by_branching_ratio"]["intermediate"]["macro_mse"]
    )

    def regime_mse(metrics: dict[str, object], neural: bool, regime: str) -> float:
        key = "macro_mse_mean" if neural else "macro_mse"
        return float(metrics["by_branching_ratio"][regime][key])

    def parameter_mse(metrics: dict[str, object], neural: bool, name: str) -> float:
        key = "mse_mean" if neural else "mse"
        return float(metrics["per_parameter"][name][key])

    def dec(value: float) -> str:
        return decimal(value, lang)

    def short(value: float) -> str:
        return short_decimal(value, lang)

    def compact(value: float) -> str:
        return compact_decimal(value, lang)

    mle_best_parameters = [
        PARAMETER_LABELS[name]
        for name in PARAMETERS
        if parameter_mse(mle, False, name)
        < min(parameter_mse(lstm, True, name), parameter_mse(transformer, True, name))
    ]
    if mle_best_parameters:
        conjunction = " and " if en else " e "
        listed = (
            mle_best_parameters[0]
            if len(mle_best_parameters) == 1
            else ", ".join(mle_best_parameters[:-1]) + conjunction + mle_best_parameters[-1]
        )
        mle_best_sentence = (
            f"The MLE obtained the lowest MSE among the three estimators for {listed}."
            if en
            else f"O MLE obteve o menor MSE entre os três estimadores em {listed}."
        )
    else:
        mle_best_sentence = (
            "The MLE did not obtain the lowest MSE for any individual parameter."
            if en
            else "O MLE não obteve o menor MSE em nenhum parâmetro isolado."
        )

    summary_rows = []
    for label, metrics, neural in estimators:
        if neural:
            lower, upper = metrics.get(
                "macro_mse_95ci", metrics["macro_mse_approx_95ci"]
            )
            separator = ", " if en else "; "
            interval = f"[{dec(lower)}{separator}{dec(upper)}]"
            estimator_mse = metrics["macro_mse_mean"]
            macro_mae = metrics["macro_mae_mean"]
            branching_mse = metrics["branching_ratio_mse_mean"]
            validity = metrics["validity_rate_mean"]
        else:
            interval = "--"
            estimator_mse = metrics["macro_mse"]
            macro_mae = metrics["macro_mae"]
            branching_mse = metrics["branching_ratio"]["mse"]
            validity = metrics["validity_rate"]
        summary_rows.append(
            f"{label} & {dec(estimator_mse)} & {interval} & "
            f"{dec(macro_mae)} & {dec(branching_mse)} & "
            f"{percent(validity, lang)}\\\\"
        )

    parameter_rows = []
    for label, metrics, neural in estimators:
        values = [parameter_mse(metrics, neural, name) for name in PARAMETERS]
        parameter_rows.append(
            f"{label} & " + " & ".join(dec(value) for value in values) + r"\\"
        )

    regime_rows = []
    for label, metrics, neural in estimators:
        values = [regime_mse(metrics, neural, regime) for regime in REGIMES]
        regime_rows.append(
            f"{label} & " + " & ".join(dec(value) for value in values) + r"\\"
        )

    figures = "../build/figures/"
    suffix = FIGURE_SUFFIX[lang]
    regime_values = {
        regime: (
            short(regime_mse(mle, False, regime)),
            short(regime_mse(lstm, True, regime)),
            short(regime_mse(transformer, True, regime)),
        )
        for regime in ("intermediate", "near_critical")
    }
    beta_share = percentage_points(mle_beta_error_share, lang)
    low_ratio = compact(mle_low_regime_ratio)
    intermediate = regime_values["intermediate"]
    near_critical = regime_values["near_critical"]

    if en:
        intro = [
            r"\section*{Results of the final optimized run}",
            "",
            r"The final protocol uses only paths of 1,000 events, with 75,000 training",
            r"samples, 10,000 validation samples and 15,000 test samples.",
            r"The networks are summarized over five independent seeds; the MLE is fitted",
            r"separately on each path of the same test set.",
            r"The network intervals use Student's $t$ distribution across seeds.",
        ]
        captions = {
            "summary": r"\caption{Aggregate metrics on the frozen synthetic test set "
            r"(15,000 paths of 1,000 events).}",
            "summary_header": r"Estimator & Macro MSE & 95\% CI & Macro MAE & MSE of $g$"
            r" & Validity\\",
            "parameters": r"\caption{Test MSE by parameter (networks: mean over five seeds).}",
            "parameters_header": r"Estimator & $\mu$ & $\alpha_{\mathrm{self}}$ & "
            r"$\alpha_{\mathrm{cross}}$ & $\beta$\\",
            "regimes": r"\caption{Macro MSE by regime of the branching ratio $g$.}",
            "regimes_header": r"Estimator & Low $g$ & Intermediate $g$ & Near-critical $g$\\",
        }
        discussion = [
            rf"For the MLE, the parameter $\beta$ accounted for {beta_share} of the sum of",
            r"squared errors, and the macro MSE in the low-$g$ regime was "
            rf"{low_ratio} times",
            r"that of the intermediate regime: with weak excitation, $\beta$ is weakly",
            r"identified and some paths produce extreme estimates. Outside this regime the",
            r"MLE is competitive: in the intermediate regime its macro MSE was "
            rf"{intermediate[0]} (LSTM:",
            rf"{intermediate[1]}; Transformer: {intermediate[2]}) and, in the near-critical"
            r" regime,",
            rf"{near_critical[0]} (LSTM: {near_critical[1]}; Transformer: "
            rf"{near_critical[2]}).",
            mle_best_sentence,
            r"Therefore, the aggregate macro MSE does not rank the estimators robustly,",
            r"and numerical convergence does not imply good parameter recovery.",
        ]
        figure_captions = [
            [
                r"\caption{Test MSE by parameter under the 1,000-event protocol.",
                r"The network error bars represent one standard deviation across seeds.}",
            ],
            [
                r"\caption{Macro MSE by regime of the branching ratio. The network error",
                r"bars represent one standard deviation across seeds.}",
            ],
            [
                r"\caption{Cost per path recorded during the final run, on a logarithmic",
                r"scale. For the MLE, wall-clock time with four processes competing with",
                r"training on the GPU; for the networks, median latency at batch size one on",
                r"MPS. A later controlled benchmark measured about 10~ms per path for the MLE",
                r"on a single core and should be preferred to this figure.}",
            ],
        ]
        closing = [
            rf"The MLE converged on {percent(mle['convergence_rate'], lang)} of the paths",
            rf"and used a median of {compact(mle['median_iterations'])} iterations.",
            r"All comparisons use the same frozen test set.",
        ]
    else:
        intro = [
            r"\section*{Resultados da execução final otimizada}",
            "",
            r"O protocolo final usa exclusivamente trajetórias de 1.000 eventos, com",
            r"75.000 amostras de treino, 10.000 de validação e 15.000 de teste.",
            r"As redes são resumidas sobre cinco seeds independentes; o MLE é ajustado",
            r"separadamente em cada trajetória do mesmo conjunto de teste.",
            r"Os intervalos das redes usam a distribuição $t$ de Student entre seeds.",
        ]
        captions = {
            "summary": r"\caption{Métricas agregadas no teste sintético congelado "
            r"(15.000 trajetórias de 1.000 eventos).}",
            "summary_header": r"Estimador & MSE macro & IC 95\% & MAE macro & MSE de $g$"
            r" & Validade\\",
            "parameters": r"\caption{MSE de teste por parâmetro (redes: média de cinco seeds).}",
            "parameters_header": r"Estimador & $\mu$ & $\alpha_{\mathrm{self}}$ & "
            r"$\alpha_{\mathrm{cross}}$ & $\beta$\\",
            "regimes": r"\caption{MSE macro por regime da razão de ramificação $g$.}",
            "regimes_header": r"Estimador & $g$ baixo & $g$ intermediário & $g$ quase crítico\\",
        }
        discussion = [
            rf"No MLE, o parâmetro $\beta$ respondeu por {beta_share} da soma dos erros"
            r" quadráticos,",
            rf"e o MSE macro no regime de $g$ baixo foi {low_ratio} vezes o do regime"
            r" intermediário:",
            r"com excitação fraca, $\beta$ é pouco identificado e algumas trajetórias",
            r"produzem estimativas extremas. Fora desse regime, o MLE é competitivo:",
            rf"no regime intermediário, seu MSE macro foi {intermediate[0]} (LSTM: "
            rf"{intermediate[1]}; Transformer: {intermediate[2]}) e, no",
            rf"quase crítico, {near_critical[0]} (LSTM: {near_critical[1]}; Transformer: "
            rf"{near_critical[2]}).",
            mle_best_sentence,
            r"Portanto, o MSE macro agregado não ordena os estimadores de forma robusta,",
            r"e convergência numérica não implica boa recuperação paramétrica.",
        ]
        figure_captions = [
            [
                r"\caption{MSE de teste por parâmetro no protocolo de 1.000 eventos.",
                r"As barras de erro das redes representam um desvio-padrão entre seeds.}",
            ],
            [
                r"\caption{MSE macro por regime da razão de ramificação. As barras de erro",
                r"das redes representam um desvio-padrão entre seeds.}",
            ],
            [
                r"\caption{Custo por trajetória registrado durante a execução final, em escala",
                r"logarítmica. Para o MLE, tempo de parede com quatro processos que concorriam",
                r"com o treino na GPU; para as redes, latência mediana em batch unitário no",
                r"MPS. Um benchmark controlado posterior mediu cerca de 10~ms por trajetória",
                r"para o MLE em um único núcleo e deve ser preferido a esta figura.}",
            ],
        ]
        closing = [
            rf"O MLE convergiu em {percent(mle['convergence_rate'], lang)} das trajetórias",
            rf"e usou mediana de {compact(mle['median_iterations'])} iterações.",
            r"Todas as comparações usam o mesmo conjunto de teste congelado.",
        ]

    figure_blocks = []
    for (name, width), caption in zip(
        (("parameter_mse", "0.88"), ("branching_regime_mse", "0.88"),
         ("inference_cost", "0.80")),
        figure_captions,
        strict=True,
    ):
        figure_blocks += [
            r"\begin{figure}[htbp]",
            r"\centering",
            rf"\includegraphics[width={width}\linewidth]{{{figures}{name}{suffix}.pdf}}",
            *caption,
            r"\end{figure}",
            "",
        ]

    return "\n".join(
        [
            *intro,
            "",
            r"\begin{table}[htbp]",
            r"\centering",
            captions["summary"],
            r"\label{tab:final-agregado}",
            r"\small",
            r"\begin{tabular}{lrrrrr}",
            r"\toprule",
            captions["summary_header"],
            r"\midrule",
            *summary_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
            r"\begin{table}[htbp]",
            r"\centering",
            captions["parameters"],
            r"\label{tab:final-parametros}",
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            captions["parameters_header"],
            r"\midrule",
            *parameter_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
            r"\begin{table}[htbp]",
            r"\centering",
            captions["regimes"],
            r"\label{tab:final-regimes}",
            r"\begin{tabular}{lrrr}",
            r"\toprule",
            captions["regimes_header"],
            r"\midrule",
            *regime_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
            network_sentence(lstm["macro_mse_mean"], transformer["macro_mse_mean"], lang),
            significance_sentence(seed_macro_mse(lstm), seed_macro_mse(transformer), lang),
            *discussion,
            "",
            *figure_blocks,
            *closing,
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("artifacts/final_hawkes/lee_local_optimized/summary.json"),
    )
    parser.add_argument("--lang", choices=LANGUAGES, default="pt")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="padrão: thesis/latex/final_results.tex (pt) ou thesis/latex/en/... (en)",
    )
    args = parser.parse_args()
    output = args.output or DEFAULT_OUTPUTS[args.lang]
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_fragment(summary, args.lang), encoding="utf-8")


if __name__ == "__main__":
    main()
