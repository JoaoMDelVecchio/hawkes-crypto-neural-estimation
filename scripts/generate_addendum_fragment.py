#!/usr/bin/env python
"""Gera ``thesis/latex/addendum_results.tex`` (ADRs 009–011) a partir dos artefatos do adendo.

Cada parte entra só se o artefato existir: teste de β (``beta_sweep*.json``), viés da grade,
seleção dos braços (``summary.json``), semana de fevereiro (``holdout.json``), escopo B
reduzido (``scope_b_010/``), correção da grade (``grid_correction_011/``) e volatilidade.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_empirical_fragment import num, pct, pvalue  # noqa: E402

LABELS = {
    "lstm_lee/finetune": "LSTM, fine-tuned (frozen, selected in the TG)",
    "transformer/finetune": "Transformer, fine-tuned (frozen)",
    "lstm_lee/finetune_long": "LSTM, 1,000-epoch cap",
    "lstm_lee/finetune_sched": "LSTM, cap + schedule",
    "transformer/finetune_sched": "Transformer, cap + schedule",
    "lstm_lee/finetune_std": "LSTM, cap + schedule + standardized loss",
    "transformer/finetune_std": "Transformer, cap + schedule + standardized loss",
    "mix/lstm+transformer": "Average of the best LSTM and the best Transformer",
    "label_mean": "Mean of the training labels",
}
SCOPE_B = {
    "label_mean": "Mean of the training labels",
    "lstm_lee/finetune": "LSTM, fine-tuned (frozen, selected in the TG)",
    "lstm_lee/finetune_long": "LSTM, 1,000-epoch cap",
    "transformer/finetune": "Transformer, fine-tuned (frozen)",
    "mix/lstm+transformer": "Average of the two above",
    "B/zero": "LSTM, scope B prior, no fine-tuning",
    "B/finetune": "LSTM, scope B prior, fine-tuned",
    "mix/B+transformer": "Average of the latter and the Transformer",
}
VOLATILITY = {
    "MLE": "MLE of the window",
    "lstm_lee/finetune": "LSTM, fine-tuned (frozen, selected in the TG)",
    "lstm_lee/finetune_long": "LSTM, 1,000-epoch cap",
    "transformer/finetune": "Transformer, fine-tuned (frozen)",
    "mix/lstm+transformer": "Average of the two above",
}
GRID = {
    "network": "LSTM trained on the continuous-time truth",
    "mle_grid": "MLE on the grid",
    "mle_corrected": "MLE on the grid, corrected by the map",
}
PHRASE = {
    "lstm_lee/finetune_long": "the LSTM with the 1,000-epoch cap",
    "lstm_lee/finetune_sched": "the LSTM with the cap and the schedule",
    "transformer/finetune_sched": "the Transformer with the cap and the schedule",
    "lstm_lee/finetune_std": "the LSTM with the standardized loss",
    "transformer/finetune_std": "the Transformer with the standardized loss",
    "mix/lstm+transformer": "the average of the best LSTM and the best Transformer",
}
SHORT = {
    "lstm_lee/finetune": "LSTM (frozen)",
    "lstm_lee/finetune_sched": "LSTM, schedule",
    "transformer/finetune_sched": "Transformer, schedule",
    "lstm_lee/finetune_std": "LSTM, standardized",
    "transformer/finetune_std": "Transformer, standardized",
    "mix/lstm+transformer": "LSTM + Transformer average",
}


def ci(block: dict, digits: int = 4) -> str:
    low, high = block["bootstrap_95ci"]
    return f"[{num(low, digits)}, {num(high, digits)}]"


def p_text(value: float) -> str:
    text = pvalue(value)
    return f"$p$ {text}" if text.startswith("$") else f"$p={text}$"


def signed_pct(value: float) -> str:
    return f"$-${pct(-value)}" if value < 0 else f"+{pct(value)}"


def small_p(value: float) -> str:
    """Wilcoxon p que estoura para zero em ponto flutuante vira um limite superior."""
    return "$p<10^{-10}$" if value < 1e-10 else p_text(value)


def beta_section(sweep: dict, extension: dict | None) -> str:
    betas = {**sweep["betas"], **(extension["betas"] if extension else {})}
    order = sorted(betas, key=float)
    star = set(extension["betas"]) if extension else set()
    head = " & ".join(f"{float(b):.1f}" + ("$^{\\ast}$" if b in star else "")
                      for b in order)
    rows = [
        ("Median MLE $\\beta$ on the grid", [betas[b]["mle_median"]["beta"] for b in order], 2),
        ("LSTM, fine-tuned", [betas[b]["lstm_lee"]["mse_vs_mle"] for b in order], 3),
        ("Transformer, fine-tuned", [betas[b]["transformer"]["mse_vs_mle"] for b in order], 3),
    ]
    body = "\n".join(f"{name} & " + " & ".join(num(v, d) for v in values) + r" \\"
                     for name, values, d in rows)
    h = sweep["h_beta"]
    return f"""
\\subsection{{Does the LSTM degrade faster when \\texorpdfstring{{$\\beta$}}{{beta}} moves?}}
\\label{{sec:addendum-beta}}

The October result suggested that the LSTM stores the decay in its weights, tuned to the
January $\\beta$, and degrades faster than the Transformer when $\\beta$ changes. To test this
without new market data, {sweep['betas'][order[0]]['windows']} paths per value of $\\beta$
were simulated with $\\mu={sweep['fixed']['mu']}$, $g={sweep['fixed']['branching_ratio']}$
and $\\alpha_c={sweep['fixed']['alpha_cross']}$ at the centre of the January labels, taken to
the 100~ms grid exactly as the real data and cut to 1,000 events. The target is the MLE of
each path, which is what the fine-tuned networks learned to imitate; the frozen weights are
used without retraining (Table~\\ref{{tab:addendum-beta}}).

\\begin{{table}}[htbp]
\\centering
\\caption{{Mean loss against the MLE on simulated paths, ensemble of five seeds. The first
row is the true $\\beta$ of the simulation; $^{{\\ast}}$ marks values added after the
registered grid.}}
\\label{{tab:addendum-beta}}
\\small
\\begin{{tabular}}{{l{'r' * len(order)}}}
\\toprule
True $\\beta$ & {head} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

By the registered rule the hypothesis is supported: the LSTM-minus-Transformer difference
grows by {num(h['delta'])} from $\\beta=2.7$ to $\\beta=4.5$, with a 95\\% bootstrap interval
of {ci(h, 3)}. The premise of the rule, however, failed. The grid merges events that fall in
the same 100~ms cell and pulls the MLE $\\beta$ down: a true $\\beta$ of 4.5 yields a median
label of {num(betas['4.5']['mle_median']['beta'], 2)}, the centre of January, not a shift
away from it. In the space of the labels, the LSTM has the
lower loss inside and below the January range, where it is the Transformer that degrades;
the two tie near the October median ($\\beta \\approx 3.2$); and above the training range
(95\\% of the January labels lie below 3.74) the loss of the LSTM grows about fourfold and
crosses that of the Transformer. The hypothesis therefore holds only for $\\beta$ above the
training range, and $\\beta$ alone does not explain the October gap, since October also has
larger $\\mu$ and $\\alpha_s$. A side result: on the grid, the MLE recovers between 50\\% and
80\\% of the continuous-time $\\beta$ in this range.
"""


def grid_section(bias: dict) -> str:
    forward = bias["forward_map_median"]
    mus = list(forward)
    betas = list(forward[mus[0]])
    head = " & ".join(b.split("=")[1] for b in betas)
    body = "\n".join(
        f"${mu.split('=')[1]}$ & " + " & ".join(num(forward[mu][b]["beta_hat"], 2) for b in betas)
        + r" \\" for mu in mus)
    inv = bias["inversion"]
    train, val, october = inv["calibration_train"], inv["validation"], inv["october_2023_median"]
    design = bias["design"]
    return f"""
\\subsection{{How much of \\texorpdfstring{{$\\beta$}}{{beta}} does the 100~ms grid hide?}}
\\label{{sec:addendum-grid}}

The $\\beta$ sweep showed that the grid pulls the MLE $\\beta$ down. This post hoc analysis
measures the effect over the region of the real labels. For each pair $(\\mu,\\beta)$ of a
mesh, {design['paths_per_cell']} paths with $g={design['branching_ratio']}$ and
$\\alpha_c={design['alpha_cross']}$ were simulated in continuous time, taken to the grid
(one event per cell), cut to 1,000 events and fitted by the MLE
(Table~\\ref{{tab:addendum-grid}}). The median of the MLE defines a map from continuous-time
parameters to grid labels, interpolated in logarithms and inverted for each real window,
which is a form of indirect inference.

\\begin{{table}}[htbp]
\\centering
\\caption{{Median MLE $\\beta$ on the 100~ms grid for each continuous-time $(\\mu,\\beta)$,
$g={design['branching_ratio']}$, {design['paths_per_cell']} paths per cell.}}
\\label{{tab:addendum-grid}}
\\small
\\begin{{tabular}}{{l{'r' * len(betas)}}}
\\toprule
$\\mu$ $\\backslash$ true $\\beta$ & {head} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

The bias grows with both $\\beta$ and $\\mu$, that is, with the share of events that fall in
an occupied cell: the larger the intensity relative to the 100~ms step, the more events are
merged and the slower the decay looks. The branching ratio is also biased down, from 0.5 to
between 0.22 and 0.48. Inverting the map for the January labels (inside the map in
{pct(train['inside_map_rate'])} of the training windows), the median $\\beta$ of
{num(train['beta_grid_mle_quantiles'][1], 2)} on the grid corresponds to
{num(train['beta_continuous_quantiles'][1], 1)} in continuous time (5\\%--95\\%:
{num(train['beta_continuous_quantiles'][0], 1)}--{num(train['beta_continuous_quantiles'][2], 1)}),
and the grid label is a median {pct(train['median_ratio_grid_to_continuous'])} of the
continuous value; on validation the figures are {num(val['beta_continuous_quantiles'][1], 1)}
and {pct(val['median_ratio_grid_to_continuous'])}. The October median of
{num(october['beta_grid_mle'], 2)} corresponds to {num(october['beta_continuous'], 1)}. The
estimate of $\\mu$ is nearly unaffected. First, the labels of this
work describe the process as seen through the grid, not the process in continuous time; the
real decay is roughly twice as fast, about
{num(1000 / train['beta_continuous_quantiles'][1], 0)}~ms of mean memory instead of
{num(1000 / train['beta_grid_mle_quantiles'][1], 0)}~ms. Second,
jittering the times inside their cell, used in the diagnostics, does not undo this bias,
because it cannot restore the merged events; it only tests the effect of ties. The inversion
assumes $g=0.5$ and $\\alpha_c\\approx0$ and treats the median as the target; it gives the
order of magnitude of the bias, not a corrected estimate per window.
"""


def selection_section(summary: dict) -> str:
    mse = summary["validation_macro_mse"]
    rows = "\n".join(f"{LABELS[k]} & {num(v, 4)} \\\\" for k, v in mse.items())
    runs = summary["variants"]
    epochs = {k: [r["best_epoch"] for r in v.get("runs", [])] for k, v in runs.items()}
    long_epochs = epochs.get("lstm_lee/finetune_long", [])
    return f"""
\\subsection{{A better fine-tuning recipe}}
\\label{{sec:addendum-recipe}}

The fine-tuned LSTM reached its best epoch between 291 and 299 of 300, so the original run
stopped before convergence, and both fine-tuned networks predicted $\\alpha_c$ worse than the
mean of the labels because $\\beta$ dominated the loss. Five new arms start from the same
synthetic weights, with the same data, seeds, learning rate and batch size: a 1,000-epoch
cap; the cap plus a five-epoch warm-up, halving of the learning rate after ten epochs without
improvement and gradient clipping; and the latter plus a training loss standardized by the
variance of the training labels (floored at 0.01). Checkpoints and early stopping use the
validation MSE on the original scale, as in the main protocol, and the selection rule is the
same (Table~\\ref{{tab:addendum-validation}}). With the longer cap the LSTM stopped at epochs
{', '.join(str(e) for e in long_epochs)}.

\\begin{{table}}[htbp]
\\centering
\\caption{{Validation macro MSE of the five-seed ensembles. The selection among the new arms
was recorded before the new week was read.}}
\\label{{tab:addendum-validation}}
\\small
\\begin{{tabular}}{{lr}}
\\toprule
Variant & Validation MSE \\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

Selected on validation: {PHRASE[summary['selected_variant']]}.
"""


def holdout_section(report: dict, summary: dict) -> str:
    block, october = report["holdout_2024_02"], report["october_2023"]
    variants = block["variants"]
    order = [k for k in LABELS if k in variants]
    rows = "\n".join(
        f"{LABELS[k]} & {num(variants[k]['macro_mse'], 4)} & "
        + " & ".join(num(variants[k]["r2"][p], 2) for p in ("mu", "alpha_self", "beta"))
        + f" & {num(october['variants'][k]['macro_mse'], 4)} \\\\"
        for k in order)
    primary = block["primary"]
    secondary = block["secondary"]
    sec_rows = "\n".join(
        f"{SHORT[k.split(' vs ')[0]]} vs {SHORT[k.split(' vs ')[1]]} & "
        f"{num(v['mean_difference_a_minus_b'], 4)} & {ci(v)} & {pvalue(v['holm_p'])} \\\\"
        for k, v in secondary.items())
    return f"""
\\subsection{{A new week as a clean test}}
\\label{{sec:addendum-holdout}}

The test block of the main protocol had already been opened, so the new arms were evaluated
on a week that nobody had seen: BTCUSDT and ETHUSDT, 5--11 February 2024, processed exactly
as the primary period\\footnote{{The Binance files of that week have their lines interleaved
out of time order; each file was sorted by transaction time and update identifier, which
agree after sorting, before the same processing.}} and cut into
{block['windows']} non-overlapping windows of 1,000 events labelled by the MLE. The week was
read once, after the selection (Table~\\ref{{tab:addendum-holdout}}).

\\begin{{table}}[htbp]
\\centering
\\caption{{Macro MSE and $R^2$ per parameter on the new week, and macro MSE on October 2023
(exploratory), ensembles of five seeds. $R^2$ of $\\alpha_c$ is left out of the table because
its label variance is close to zero, which makes it unstable; it is discussed in the text.}}
\\label{{tab:addendum-holdout}}
\\small
\\begin{{tabular}}{{lrrrrr}}
\\toprule
Variant & MSE & $R^2_\\mu$ & $R^2_{{\\alpha_s}}$ & $R^2_\\beta$ & October \\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

Primary comparison, registered in advance: {PHRASE[primary['a']]} minus the frozen LSTM,
window by window. The mean difference is {num(primary['mean_difference_a_minus_b'], 4)}
(paired bootstrap 95\\% interval {ci(primary)}, Wilcoxon {p_text(primary['wilcoxon_p'])}),
and the new arm has the lower loss in {pct(primary['win_rate_a'])} of the windows. The
secondary comparisons, with Holm-adjusted $p$-values, are in
Table~\\ref{{tab:addendum-secondary}}.

\\begin{{table}}[htbp]
\\centering
\\caption{{Secondary paired comparisons on the new week (difference A minus B; negative
favours A). The interval is for the mean difference and the $p$-value is Wilcoxon's, which
looks at the median, so the two can disagree when a few windows dominate the mean.}}
\\label{{tab:addendum-secondary}}
\\footnotesize
\\begin{{tabular}}{{lrrr}}
\\toprule
A vs B & Difference & 95\\% interval & Holm $p$ \\\\
\\midrule
{sec_rows}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

The registered primary hypothesis is supported: training the same LSTM for longer lowers the
loss on a week that played no role in any decision, by about a fifth, and the gain is
consistent window by window. The two best candidates on validation are the two best on the
new week, in swapped order. The schedule (warm-up, plateau decay and clipping) did not
help: with it the LSTM stopped earlier and tied with the frozen network. The standardized
loss did what it was meant to do for $\\alpha_c$ ($R^2$ from
{num(variants['lstm_lee/finetune']['r2']['alpha_cross'], 1)} to
{num(variants['lstm_lee/finetune_std']['r2']['alpha_cross'], 1)} for the LSTM and
{num(variants['transformer/finetune_std']['r2']['alpha_cross'], 2)} for the Transformer, the
only variants that are not clearly worse than the constant mean on this parameter), but
paid for it in $\\beta$ and lost overall. The average of the LSTM and the Transformer,
registered before the week was read and only second on validation, has the lowest loss on
the new week ({num(variants['mix/lstm+transformer']['macro_mse'], 4)}), which is consistent with the
$\\beta$ sweep: the two networks err in different regions, so their errors partly cancel. No
Transformer arm beat the frozen LSTM on the new week. In October, which had been seen before
and is exploratory, the picture of the main protocol remains: the Transformers have the
lowest losses ({num(october['variants']['transformer/finetune']['macro_mse'], 3)} frozen),
the longer LSTM improves only slightly on the frozen one
({num(october['variants']['lstm_lee/finetune_long']['macro_mse'], 3)} against
{num(october['variants']['lstm_lee/finetune']['macro_mse'], 3)}), and the average
({num(october['variants']['mix/lstm+transformer']['macro_mse'], 3)}) is close to the best.
Inside the regime of the training data, then, the LSTM remains the better single network
and was undertrained in the main protocol; under a change of regime the Transformer errs
less; and the average of the two has the lowest loss on the new week and is close to the
best in October.
"""


def scope_b_section(report: dict, summary: dict, generated: dict,
                    validation_009: dict) -> str:
    march, feb, october = (report[k] for k in ("holdout_2024_03", "holdout_2024_02",
                                               "october_2023"))
    validation = {**validation_009, **summary["validation_macro_mse"]}
    rows = "\n".join(
        f"{SCOPE_B[k]} & {num(validation[k], 4) if k in validation else '--'} & "
        + " & ".join(num(b["variants"][k]["macro_mse"], 4) for b in (march, feb, october))
        + r" \\"
        for k in SCOPE_B)
    primary, sec = march["primary"], march["secondary"]
    mix = sec["mix/B+transformer vs mix/lstm+transformer"]
    frozen = sec["B/finetune vs lstm_lee/finetune"]
    zero = sec["B/zero vs label_mean"]
    variants = march["variants"]
    old, new = variants["lstm_lee/finetune_long"], variants["B/finetune"]
    epochs = [r["best_epoch"] for r in summary["variants"]["B/finetune"]["runs"]]
    return f"""
\\subsection{{A synthetic prior built from the real labels (reduced scope B)}}
\\label{{sec:addendum-scope-b}}

The main protocol pretrained the networks on a synthetic prior fixed in advance (scope A)
and left for later work a prior built from the real data (scope B). A reduced version was
registered (ADR 010) before any path was generated. The prior is a mixture: in 75\\% of the
draws, the label of a random training window is inverted to continuous time by the map of
Section~\\ref{{sec:addendum-grid}}, with log-normal noise on $\\mu$ and $\\beta$ and the
branching ratio inflated by a factor between 1 and 1.25 to offset the bias of the grid; in
25\\%, the parameters are drawn from a wide box around the real range. Each path is simulated
in continuous time, taken to the 100~ms grid, cut to 1,000 events and labelled by the MLE on
the grid, so that the synthetic target means the same as the real one.
{generated['kept']:,} paths were kept (none dropped), and the real
labels fall inside the 1--99\\% synthetic range in
{pct(min(generated['real_inside_synthetic_1_99'].values()))} to
{pct(max(generated['real_inside_synthetic_1_99'].values()))} of the training windows,
depending on the parameter. The LSTM was pretrained on these paths (five seeds, 150-epoch
cap, which all seeds reached) and then fine-tuned on the real training windows with the
recipe of Section~\\ref{{sec:addendum-recipe}} (best epochs {', '.join(map(str, sorted(epochs)))}).
Since the February week had been read, a second unseen week was downloaded and read once:
BTCUSDT and ETHUSDT, 4--10 March 2024, {march['windows']:,} windows
(Table~\\ref{{tab:addendum-scope-b}}).

\\begin{{table}}[htbp]
\\centering
\\caption{{Macro MSE of the five-seed ensembles. Validation was recorded before the March week
was read; February and October had been read before and are exploratory.}}
\\label{{tab:addendum-scope-b}}
\\footnotesize
\\begin{{tabular}}{{lrrrr}}
\\toprule
Variant & Validation & March 2024 & February 2024 & October 2023 \\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

Primary comparison, registered in advance: the fine-tuned scope B LSTM minus the LSTM with the
1,000-epoch cap, window by window on the March week. The mean difference is
{num(primary['mean_difference_a_minus_b'], 3)} (95\\% interval {ci(primary, 3)}, Wilcoxon
{small_p(primary['wilcoxon_p'])}), the loss falls by
{pct(1 - primary['mean_loss_a'] / primary['mean_loss_b'])}, and the scope B network has the
lower loss in {pct(primary['win_rate_a'])} of the windows. The three secondary comparisons
also favour it after Holm's correction: against the frozen LSTM,
{num(frozen['mean_difference_a_minus_b'], 3)} ({ci(frozen, 3)}); the average with the
Transformer against the average of Section~\\ref{{sec:addendum-holdout}},
{num(mix['mean_difference_a_minus_b'], 3)} ({ci(mix, 3)}); and the pretrained network without
fine-tuning against the mean of the labels, {num(zero['mean_difference_a_minus_b'], 3)}
({ci(zero, 3)}); all with Holm {small_p(max(v['holm_p'] for v in sec.values()))}. The sign is
the same on February and October.

The gain is concentrated where the earlier networks failed. March and October are the weeks
farthest from the calibration period (the constant mean errs three times more there than on
February), and most of the error of the earlier LSTM comes from $\\mu$: on March its MSE for
$\\mu$ is {num(old['per_parameter']['mu']['mse'], 3)} ($R^2={num(old['r2']['mu'], 2)}$),
against {num(new['per_parameter']['mu']['mse'], 3)} ($R^2={num(new['r2']['mu'], 2)}$) for the
scope B network, which also lowers the error in $\\beta$ from
{num(old['per_parameter']['beta']['mse'], 3)} to {num(new['per_parameter']['beta']['mse'], 3)}.
The wide part of the prior covers activity levels that the January calibration does not, and
the fine-tuning keeps the network anchored to the real labels: without it, the scope B
network is no better than the earlier ones on March and much worse on February, so the
level of the estimates comes from the fine-tuning. Unlike in
Section~\\ref{{sec:addendum-holdout}}, averaging with the Transformer does not help on March
or October: once the LSTM extrapolates well, the Transformer mostly adds its own error. Scope B,
reduced as it is, gives the largest out-of-sample gain of this work, and the selected model of
the main protocol is unchanged by rule.
"""


def grid_correction_section(report: dict) -> str:
    est = report["estimators"]
    rows = "\n".join(
        f"{GRID[k]} & {num(est[k]['macro_mse'], 3)} & {num(est[k]['mse']['mu'], 4)} & "
        f"{num(est[k]['mse']['beta'], 3)} & {signed_pct(est[k]['median_relative_error_beta'])} & "
        f"{pct(est[k]['median_abs_relative_error_beta'])} \\\\"
        for k in GRID)
    primary, sec = report["primary"], report["secondary"]
    vs_grid = sec["network vs mle_grid"]
    corr_vs_grid = sec["mle_corrected vs mle_grid"]
    real = report["real_windows"]
    val, feb = real["validation"], real["holdout_2024_02"]
    return f"""
\\subsection{{Can a network undo the bias of the grid?}}
\\label{{sec:addendum-grid-correction}}

The MLE on the grid cannot recover the continuous-time $\\beta$
(Section~\\ref{{sec:addendum-grid}}), but a network trained on gridded paths with the true
parameters as target can learn the correction, which is a structural advantage of amortized
estimation under discretized observation. The protocol was registered (ADR 011) before
training. The LSTM was trained from scratch on the paths of scope B
(Section~\\ref{{sec:addendum-scope-b}}) with their continuous-time parameters as target (five
seeds, 150-epoch cap) and compared on {report['test_paths']:,} new paths of the same prior
with two alternatives: the MLE on the grid and the MLE corrected by inverting the map of
Section~\\ref{{sec:addendum-grid}} (Table~\\ref{{tab:addendum-grid-correction}}).

\\begin{{table}}[htbp]
\\centering
\\caption{{Error against the continuous-time parameters on {report['test_paths']:,} new
simulated paths. Relative error of $\\beta$: signed and absolute medians.}}
\\label{{tab:addendum-grid-correction}}
\\footnotesize
\\begin{{tabular}}{{lrrrrr}}
\\toprule
Estimator & Macro MSE & MSE $\\mu$ & MSE $\\beta$ & Bias $\\beta$ & $|$Error$|$ $\\beta$ \\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

Primary comparison, registered in advance: network minus corrected MLE, path by path, mean
difference {num(primary['mean_difference_a_minus_b'], 3)} (95\\% interval
{ci(primary, 3)}, Wilcoxon {small_p(primary['wilcoxon_p'])}); the network has the lower loss
in {pct(primary['win_rate_a'])} of the paths. Secondary, with Holm: network minus MLE on the
grid, {num(vs_grid['mean_difference_a_minus_b'], 3)} ({ci(vs_grid, 3)},
{small_p(vs_grid['holm_p'])});
corrected minus uncorrected MLE, {num(corr_vs_grid['mean_difference_a_minus_b'], 3)}
({ci(corr_vs_grid, 3)}, {small_p(corr_vs_grid['holm_p'])}).
The correction by the map alone removes almost all of the bias in $\\beta$ (median relative
error from {signed_pct(est['mle_grid']['median_relative_error_beta'])} to
{signed_pct(est['mle_corrected']['median_relative_error_beta'])}) and has the lowest median error in
$\\beta$ and in $\\mu$. The network wins on the mean through the tails of $\\alpha_s$ and
$\\beta$, where the map, measured at a single branching ratio, extrapolates poorly.

On real windows the trained network implies a median continuous-time $\\beta$ of
{num(val['beta_network_quantiles'][1], 1)} on validation and
{num(feb['beta_network_quantiles'][1], 1)} on February, against
{num(val['beta_inverted_quantiles'][1], 1)} and {num(feb['beta_inverted_quantiles'][1], 1)}
from the inversion of the map and {num(val['beta_grid_mle_quantiles'][1], 2)} and
{num(feb['beta_grid_mle_quantiles'][1], 2)} on the grid (Spearman between network and
inversion {num(val['spearman_beta_network_vs_inverted'], 2)} and
{num(feb['spearman_beta_network_vs_inverted'], 2)}). On real data the two corrections
disagree: the network places $\\beta$ well above the map. The real windows are not Hawkes
paths (Section~\\ref{{sec:real-diagnostics}}), so a correction learned on simulated paths does
not transfer safely, and without the continuous-time truth neither can be checked. The network is
the better correction inside the prior, but on market data it has not been verified. Training
and test share the prior, which
favours the network; the map was measured only at $g=0.5$ and $\\alpha_c\\approx0$, which
penalizes the corrected MLE away from it. Both caveats were registered before the run.
"""


def volatility_section(report: dict) -> str:
    feb, val = report["holdout_2024_02"], report["validation"]
    rows = "\n".join(
        f"{VOLATILITY[k]} & {num(feb['estimators'][k]['spearman'], 3)} & "
        f"{num(feb['estimators'][k]['median_ratio'], 2)} & "
        f"{num(val['estimators'][k]['spearman'], 3)} & "
        f"{num(val['estimators'][k]['median_ratio'], 2)} \\\\"
        for k in VOLATILITY)
    primary = feb["primary"]
    return f"""
\\subsection{{Hawkes volatility with the addendum networks}}
\\label{{sec:addendum-volatility}}

Criterion C4 was measured with the selected network only, which ranks windows by volatility
much worse than the MLE (Section~\\ref{{sec:real-volatility}}). The same formula and the same
benchmark (realized volatility with $\\Delta={feb['step_seconds']:.0f}$~s) were applied, post
hoc but with the comparison registered before computing it (ADR 011), to the networks of the
addendum, on the {feb['windows']} windows of the February week and the {val['windows']}
validation windows (Table~\\ref{{tab:addendum-volatility}}).

\\begin{{table}}[htbp]
\\centering
\\caption{{Spearman correlation between the Hawkes volatility and the realized volatility
across windows, and median ratio of the two.}}
\\label{{tab:addendum-volatility}}
\\small
\\begin{{tabular}}{{lrrrr}}
\\toprule
& \\multicolumn{{2}}{{c}}{{February 2024}} & \\multicolumn{{2}}{{c}}{{Validation}} \\\\
Estimator & Spearman & Ratio & Spearman & Ratio \\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

The average of the two networks ranks the February windows better than the frozen LSTM by
{num(primary['spearman_difference'], 3)} in Spearman (bootstrap 95\\% interval
{ci(primary, 3)}), the registered comparison. The level is right for every estimator, with
ratios between 1.2 and 1.4; the estimators differ in the ranking. Training the LSTM longer
barely improves it, while the Transformer comes closest to the MLE. The LSTM has the lower
parameter error, but the Transformer preserves more of the differences between windows, on
which the ranking depends; the MLE remains the better input for this use.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts/addendum_009"))
    parser.add_argument("--scope-b", type=Path, default=Path("artifacts/scope_b_010"))
    parser.add_argument("--grid-correction", type=Path,
                        default=Path("artifacts/grid_correction_011"))
    parser.add_argument("--output", type=Path, default=Path("thesis/latex/addendum_results.tex"))
    args = parser.parse_args()

    def read(name: str, root: Path | None = None) -> dict | None:
        path = (root or args.root) / name
        return json.loads(path.read_text()) if path.exists() else None

    sweep, summary, holdout = read("beta_sweep.json"), read("summary.json"), read("holdout.json")
    parts = [
        "% Gerado por scripts/generate_addendum_fragment.py (ADRs 009–011). Não edite à mão.",
        r"\section{Addendum: analyses registered after the results were frozen}",
        r"\label{sec:addendum}",
        """
This addendum was registered (ADRs 009 to 011) after the main results had been frozen and
does not change them: the model selected in the main protocol remains the fine-tuned LSTM of
Section~\\ref{sec:empirical-protocol}. It tests the hypothesis raised by October, measures the
bias of the 100~ms grid, evaluates a better fine-tuning recipe and a reduced version of scope
B on weeks that played no role in any decision, and revisits the volatility criterion.
""",
    ]
    if sweep:
        parts.append(beta_section(sweep, read("beta_sweep_extension.json")))
    if bias := read("grid_bias.json"):
        parts.append(grid_section(bias))
    if summary:
        parts.append(selection_section(summary))
    if holdout and summary:
        parts.append(holdout_section(holdout, summary))
    scope_b = read("holdout.json", args.scope_b)
    if scope_b and summary:
        parts.append(scope_b_section(scope_b, read("summary.json", args.scope_b),
                                     read("generate.json", args.scope_b),
                                     summary["validation_macro_mse"]))
    if correction := read("evaluation.json", args.grid_correction):
        parts.append(grid_correction_section(correction))
    if volatility := read("volatility_addendum.json"):
        parts.append(volatility_section(volatility))
    args.output.write_text("\n".join(parts), encoding="utf-8")
    print(f"{args.output}: {len(parts) - 3} partes")

if __name__ == "__main__":
    main()
