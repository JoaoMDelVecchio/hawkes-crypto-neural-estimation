"""Custo do MLE por variante de implementação, para explicar o tempo reportado por Lee.

Lee (2023) reporta 1,763 s por estimativa do MLE, sem descrever a implementação. Este script
ajusta as mesmas trajetórias no ponto fixo de Lee com variantes que diferem em um fator por vez:
recursão vetorizada ou laço em Python, soma dupla O(n²), gradiente analítico ou por diferenças
finitas e otimizador (L-BFGS-B ou Nelder–Mead). Todas começam no θ verdadeiro, como em Lee, com
um único início e um núcleo de CPU. A soma dupla em Python puro é cara demais para ajustar; o
custo dela é extrapolado do tempo de uma avaliação vezes o número de avaliações do Nelder–Mead.

Análise de custo pós-congelamento: não muda nenhum número do TG. Rodar com um thread
(``OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1``).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from hawkes_crypto.estimate.mle import (
    _decode,
    _encode,
    _negative_log_likelihood_with_gradient,
    log_likelihood,
)
from hawkes_crypto.simulate.hawkes_sim import HawkesParameters, simulate_path

LEE_POINT = HawkesParameters(mu=0.3, alpha_self=0.4, alpha_cross=0.7, beta=1.5)


def nll_recursion(x, times, types):
    return _negative_log_likelihood_with_gradient(x, times, types)[0]


def nll_python_loop(x, times, types):
    value = log_likelihood(_decode(x), times, types)
    return -value if np.isfinite(value) else 1e300


def nll_pairwise(x, times, types):
    """Soma dupla O(n²) vetorizada: cada intensidade soma todos os eventos anteriores."""
    p = _decode(x)
    lags = times[:, None] - times[None, :]
    past = np.tril(np.ones_like(lags, dtype=bool), k=-1)
    same = types[:, None] == types[None, :]
    weights = np.where(same, p.alpha_self, p.alpha_cross)
    kernel = np.where(past, weights * np.exp(-p.beta * np.where(past, lags, 0.0)), 0.0)
    intensity = p.mu + kernel.sum(axis=1)
    horizon = float(times[-1])
    compensator = 2.0 * p.mu * horizon + (p.alpha_self + p.alpha_cross) / p.beta * float(
        np.sum(-np.expm1(-p.beta * (horizon - times)))
    )
    return -(float(np.sum(np.log(intensity))) - compensator)


def nll_pairwise_python(x, times, types):
    """Soma dupla O(n²) em laços de Python puro (só cronometrada, não ajustada)."""
    p = _decode(x)
    total = 0.0
    for k in range(len(times)):
        intensity = p.mu
        for j in range(k):
            alpha = p.alpha_self if types[j] == types[k] else p.alpha_cross
            intensity += alpha * np.exp(-p.beta * (times[k] - times[j]))
        total += np.log(intensity)
    horizon = float(times[-1])
    for t in times:
        total -= (p.alpha_self + p.alpha_cross) / p.beta * (1.0 - np.exp(-p.beta * (horizon - t)))
    return -(total - 2.0 * p.mu * horizon)


VARIANTS = {
    "recursion_analytic_lbfgs": (_negative_log_likelihood_with_gradient, "L-BFGS-B", True),
    "recursion_numeric_lbfgs": (nll_recursion, "L-BFGS-B", False),
    "python_loop_numeric_lbfgs": (nll_python_loop, "L-BFGS-B", False),
    "pairwise_numeric_lbfgs": (nll_pairwise, "L-BFGS-B", False),
    "pairwise_nelder_mead": (nll_pairwise, "Nelder-Mead", False),
}


def fit(variant, times, types, start):
    objective, method, jac = VARIANTS[variant]
    options = {"maxiter": 1000} if method == "L-BFGS-B" else {"maxiter": 4000, "maxfev": 4000}
    t0 = time.perf_counter()
    result = minimize(objective, start, args=(times, types), method=method, jac=jac,
                      options=options)
    elapsed = time.perf_counter() - t0
    p = _decode(result.x)
    return {
        "seconds": elapsed,
        "nfev": int(result.nfev),
        "nit": int(result.nit),
        "success": bool(result.success),
        "theta": [p.mu, p.alpha_self, p.alpha_cross, p.beta],
        "log_likelihood": float(-_negative_log_likelihood_with_gradient(result.x, times,
                                                                         types)[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=int, nargs="+", default=[1000, 2000])
    parser.add_argument("--paths", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/benchmarks/mle_variants.json"))
    args = parser.parse_args()

    start = _encode(LEE_POINT)
    report = {"point": [0.3, 0.4, 0.7, 1.5], "paths": args.paths, "lengths": {}}
    for n in args.lengths:
        paths = [simulate_path(LEE_POINT, n, seed=args.seed + i) for i in range(args.paths)]
        # aquece caches e importações antes de cronometrar
        fit("recursion_analytic_lbfgs", paths[0].times, paths[0].event_types, start)
        rows = {}
        for variant in VARIANTS:
            fits = [fit(variant, p.times, p.event_types, start) for p in paths]
            rows[variant] = fits
            med = float(np.median([f["seconds"] for f in fits]))
            print(f"n={n} {variant:28s} mediana {med * 1000:10.1f} ms  "
                  f"nfev {np.median([f['nfev'] for f in fits]):7.0f}", flush=True)
        # Python puro O(n²): uma avaliação cronometrada, vezes as avaliações do Nelder–Mead
        p0 = paths[0]
        t0 = time.perf_counter()
        nll_pairwise_python(start, p0.times, p0.event_types)
        per_eval = time.perf_counter() - t0
        nm_nfev = float(np.median([f["nfev"] for f in rows["pairwise_nelder_mead"]]))
        rows["pairwise_python_nelder_mead_extrapolated"] = {
            "seconds_per_evaluation": per_eval,
            "nfev_from_pairwise_nelder_mead": nm_nfev,
            "seconds": per_eval * nm_nfev,
        }
        print(f"n={n} pairwise_python_nelder_mead (extrapolado) "
              f"{per_eval * nm_nfev:10.1f} s  ({per_eval * 1000:.0f} ms por avaliação)",
              flush=True)
        reference = np.array([f["theta"] for f in rows["recursion_analytic_lbfgs"]])
        agreement = {
            v: float(np.max(np.abs(np.array([f["theta"] for f in rows[v]]) - reference)))
            for v in VARIANTS
        }
        report["lengths"][str(n)] = {"variants": rows, "max_abs_theta_diff_vs_ours": agreement}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"salvo em {args.output}")


if __name__ == "__main__":
    main()
