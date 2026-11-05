"""Simulação exata do Hawkes 2D exponencial por thinning de Ogata."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class HawkesParameters:
    mu: float
    alpha_self: float
    alpha_cross: float
    beta: float

    def __post_init__(self) -> None:
        values = (self.mu, self.alpha_self, self.alpha_cross, self.beta)
        if not all(np.isfinite(value) and value > 0 for value in values):
            raise ValueError("Todos os parâmetros devem ser positivos e finitos")
        if self.branching_ratio >= 1:
            raise ValueError("Processo instável: (alpha_self + alpha_cross) / beta >= 1")

    @property
    def branching_ratio(self) -> float:
        return (self.alpha_self + self.alpha_cross) / self.beta

    @property
    def alpha_matrix(self) -> NDArray[np.float64]:
        return np.array(
            [
                [self.alpha_self, self.alpha_cross],
                [self.alpha_cross, self.alpha_self],
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class HawkesPath:
    times: NDArray[np.float64]
    event_types: NDArray[np.int8]
    marks: NDArray[np.float64]

    @property
    def inter_arrivals(self) -> NDArray[np.float64]:
        return np.diff(self.times, prepend=0.0)


def simulate_path(
    parameters: HawkesParameters,
    n_events: int,
    *,
    seed: int,
) -> HawkesPath:
    """Simula uma trajetória iniciada com histórico vazio.

    Os tipos retornados seguem a codificação do paper: alta=1 e baixa=2.
    As marcas valem um; distribuições de salto serão incorporadas após o gate dos dados.
    """
    if n_events <= 0:
        raise ValueError("n_events deve ser positivo")

    rng = np.random.default_rng(seed)
    alpha = parameters.alpha_matrix
    excitation = np.zeros(2, dtype=np.float64)
    times = np.empty(n_events, dtype=np.float64)
    types = np.empty(n_events, dtype=np.int8)
    current_time = 0.0
    accepted = 0

    while accepted < n_events:
        upper = float((parameters.mu + excitation).sum())
        waiting_time = float(rng.exponential(1.0 / upper))
        current_time += waiting_time
        excitation *= np.exp(-parameters.beta * waiting_time)
        intensities = parameters.mu + excitation
        candidate_total = float(intensities.sum())

        if rng.random() * upper > candidate_total:
            continue

        event_index = int(rng.choice(2, p=intensities / candidate_total))
        times[accepted] = current_time
        types[accepted] = event_index + 1
        excitation += alpha[:, event_index]
        accepted += 1

    return HawkesPath(
        times=times,
        event_types=types,
        marks=np.ones(n_events, dtype=np.float64),
    )
