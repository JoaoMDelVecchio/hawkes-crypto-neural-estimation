"""Máxima verossimilhança para o Hawkes 2D simétrico exponencial."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_banded
from scipy.optimize import minimize
from scipy.special import expit

from hawkes_crypto.simulate.hawkes_sim import HawkesParameters


@dataclass(frozen=True)
class MLEFit:
    parameters: HawkesParameters
    log_likelihood: float
    converged: bool
    iterations: int
    starts: int
    message: str


class MLETask(NamedTuple):
    times: NDArray[np.float64]
    event_types: NDArray[np.int8]
    n_starts: int
    seed: int
    maxiter: int


def log_likelihood(
    parameters: HawkesParameters,
    times: NDArray[np.float64],
    event_types: NDArray[np.int8],
    *,
    horizon: float | None = None,
) -> float:
    """Log-verossimilhança condicional com histórico vazio em t=0."""
    times = np.asarray(times, dtype=np.float64)
    event_types = np.asarray(event_types, dtype=np.int8)
    if times.ndim != 1 or len(times) == 0:
        raise ValueError("times deve ser um vetor não vazio")
    if times.shape != event_types.shape:
        raise ValueError("times e event_types devem ter o mesmo tamanho")
    if np.any(np.diff(times) <= 0) or times[0] <= 0:
        raise ValueError("Tempos devem ser positivos e estritamente crescentes")
    if not np.all(np.isin(event_types, [1, 2])):
        raise ValueError("event_types deve conter apenas 1 ou 2")

    end_time = float(times[-1] if horizon is None else horizon)
    if end_time < times[-1]:
        raise ValueError("horizon não pode anteceder o último evento")

    alpha = parameters.alpha_matrix
    excitation = np.zeros(2, dtype=np.float64)
    event_term = 0.0
    previous = 0.0
    for event_time, event_type in zip(times, event_types, strict=True):
        excitation *= np.exp(-parameters.beta * (event_time - previous))
        intensity = parameters.mu + excitation[int(event_type) - 1]
        if intensity <= 0 or not np.isfinite(intensity):
            return -np.inf
        event_term += np.log(intensity)
        excitation += alpha[:, int(event_type) - 1]
        previous = float(event_time)

    kernel_mass = (
        (parameters.alpha_self + parameters.alpha_cross)
        / parameters.beta
        * np.sum(1.0 - np.exp(-parameters.beta * (end_time - times)))
    )
    compensator = 2.0 * parameters.mu * end_time + kernel_mass
    return float(event_term - compensator)


def _softplus(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.logaddexp(0.0, values)


def _inverse_softplus(value: float) -> float:
    return float(value + np.log(-np.expm1(-value)))


def _decode(unconstrained: NDArray[np.float64]) -> HawkesParameters:
    mu, alpha_self, alpha_cross, margin = _softplus(unconstrained) + 1e-8
    return HawkesParameters(
        mu=float(mu),
        alpha_self=float(alpha_self),
        alpha_cross=float(alpha_cross),
        beta=float(alpha_self + alpha_cross + margin),
    )


def _encode(parameters: HawkesParameters) -> NDArray[np.float64]:
    margin = parameters.beta - parameters.alpha_self - parameters.alpha_cross
    return np.array(
        [
            _inverse_softplus(parameters.mu),
            _inverse_softplus(parameters.alpha_self),
            _inverse_softplus(parameters.alpha_cross),
            _inverse_softplus(margin),
        ]
    )


def _negative_log_likelihood_with_gradient(
    unconstrained: NDArray[np.float64],
    times: NDArray[np.float64],
    event_types: NDArray[np.int8],
) -> tuple[float, NDArray[np.float64]]:
    """Calcula objetivo e gradiente analítico em uma única passagem pela trajetória."""
    mu, alpha_self, alpha_cross, margin = _softplus(unconstrained) + 1e-8
    beta = alpha_self + alpha_cross + margin

    deltas = np.diff(times, prepend=0.0)
    decays = np.exp(-beta * deltas)
    banded = np.ones((2, len(times)), dtype=np.float64)
    banded[1, :-1] = -decays[1:]
    sources = np.zeros((len(times), 2), dtype=np.float64)
    sources[1:, 0] = decays[1:] * (event_types[:-1] == 1)
    sources[1:, 1] = decays[1:] * (event_types[:-1] == 2)
    states = solve_banded(
        (1, 0), banded, sources, overwrite_b=True, check_finite=False
    )
    state_beta = solve_banded(
        (1, 0),
        banded,
        -deltas[:, None] * states,
        overwrite_b=True,
        check_finite=False,
    )
    is_type_one = event_types == 1
    same = np.where(is_type_one, states[:, 0], states[:, 1])
    other = np.where(is_type_one, states[:, 1], states[:, 0])
    same_beta = np.where(is_type_one, state_beta[:, 0], state_beta[:, 1])
    other_beta = np.where(is_type_one, state_beta[:, 1], state_beta[:, 0])
    intensity = mu + alpha_self * same + alpha_cross * other
    inverse_intensity = 1.0 / intensity
    log_likelihood_value = float(np.sum(np.log(intensity)))
    gradient = np.array(
        [
            np.sum(inverse_intensity),
            np.sum(same * inverse_intensity),
            np.sum(other * inverse_intensity),
            np.sum(
                (alpha_self * same_beta + alpha_cross * other_beta)
                * inverse_intensity
            ),
        ],
        dtype=np.float64,
    )

    horizon = float(times[-1])
    remaining = horizon - times
    kernel_mass_sum = float(np.sum(-np.expm1(-beta * remaining)))
    kernel_mass_gradient = float(np.sum(remaining * np.exp(-beta * remaining)))
    alpha_sum = alpha_self + alpha_cross
    log_likelihood_value -= (
        2.0 * mu * horizon + alpha_sum / beta * kernel_mass_sum
    )
    gradient[0] -= 2.0 * horizon
    gradient[1] -= kernel_mass_sum / beta
    gradient[2] -= kernel_mass_sum / beta
    gradient[3] += (
        alpha_sum * kernel_mass_sum / (beta * beta)
        - alpha_sum * kernel_mass_gradient / beta
    )

    raw_gradient = np.array(
        [
            gradient[0],
            gradient[1] + gradient[3],
            gradient[2] + gradient[3],
            gradient[3],
        ],
        dtype=np.float64,
    )
    raw_gradient *= expit(unconstrained)
    return -float(log_likelihood_value), -raw_gradient


def fit_mle(
    times: NDArray[np.float64],
    event_types: NDArray[np.int8],
    *,
    initial: HawkesParameters | None = None,
    n_starts: int = 5,
    seed: int = 0,
    maxiter: int = 1000,
) -> MLEFit:
    """Ajusta o modelo usando L-BFGS-B e múltiplas inicializações."""
    if n_starts <= 0:
        raise ValueError("n_starts deve ser positivo")
    times = np.asarray(times, dtype=np.float64)
    event_types = np.asarray(event_types, dtype=np.int8)
    if times.ndim != 1 or len(times) == 0:
        raise ValueError("times deve ser um vetor não vazio")
    if times.shape != event_types.shape:
        raise ValueError("times e event_types devem ter o mesmo tamanho")
    if np.any(np.diff(times) <= 0) or times[0] <= 0:
        raise ValueError("Tempos devem ser positivos e estritamente crescentes")
    if not np.all(np.isin(event_types, [1, 2])):
        raise ValueError("event_types deve conter apenas 1 ou 2")
    duration = float(times[-1])
    baseline = initial or HawkesParameters(
        mu=max(len(times) / (2.0 * duration) * 0.5, 1e-3),
        alpha_self=0.2,
        alpha_cross=0.2,
        beta=1.0,
    )
    center = _encode(baseline)
    rng = np.random.default_rng(seed)
    starts = [center]
    starts.extend(center + rng.normal(0.0, 0.75, size=4) for _ in range(n_starts - 1))

    best = None
    for start in starts:
        result = minimize(
            _negative_log_likelihood_with_gradient,
            start,
            args=(times, event_types),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": maxiter},
        )
        if best is None or result.fun < best.fun:
            best = result

    assert best is not None
    fitted = _decode(best.x)
    return MLEFit(
        parameters=fitted,
        log_likelihood=float(-best.fun),
        converged=bool(best.success),
        iterations=int(best.nit),
        starts=n_starts,
        message=str(best.message),
    )


def fit_mle_task(task: MLETask) -> MLEFit:
    return fit_mle(
        task.times,
        task.event_types,
        n_starts=task.n_starts,
        seed=task.seed,
        maxiter=task.maxiter,
    )


def fit_mle_batch(
    paths: list[tuple[NDArray[np.float64], NDArray[np.int8]]],
    *,
    n_starts: int = 5,
    seed: int = 0,
    maxiter: int = 1000,
    workers: int = 1,
) -> list[MLEFit]:
    """Ajusta trajetórias com inicialização heurística, sem parâmetros-oráculo."""
    tasks = [
        MLETask(times, event_types, n_starts, seed + index, maxiter)
        for index, (times, event_types) in enumerate(paths)
    ]
    if workers == 1:
        return [fit_mle_task(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(fit_mle_task, tasks))
