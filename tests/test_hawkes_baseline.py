import numpy as np
import pytest

from hawkes_crypto.estimate.mle import (
    _decode,
    _negative_log_likelihood_with_gradient,
    fit_mle,
    log_likelihood,
)
from hawkes_crypto.simulate.hawkes_sim import HawkesParameters, simulate_path


def test_parameters_reject_unstable_process() -> None:
    with pytest.raises(ValueError, match="instável"):
        HawkesParameters(mu=0.3, alpha_self=0.4, alpha_cross=0.7, beta=1.0)


def test_simulation_is_reproducible_and_ordered() -> None:
    parameters = HawkesParameters(0.3, 0.4, 0.7, 1.5)
    first = simulate_path(parameters, 200, seed=42)
    second = simulate_path(parameters, 200, seed=42)
    np.testing.assert_array_equal(first.times, second.times)
    np.testing.assert_array_equal(first.event_types, second.event_types)
    assert np.all(np.diff(first.times) > 0)
    assert set(first.event_types) == {1, 2}
    assert np.all(first.inter_arrivals > 0)


def test_log_likelihood_is_finite_for_generating_parameters() -> None:
    parameters = HawkesParameters(0.3, 0.4, 0.7, 1.5)
    path = simulate_path(parameters, 300, seed=7)
    value = log_likelihood(parameters, path.times, path.event_types)
    assert np.isfinite(value)


def test_mle_returns_stable_parameters() -> None:
    parameters = HawkesParameters(0.3, 0.4, 0.7, 1.5)
    path = simulate_path(parameters, 500, seed=17)
    fit = fit_mle(
        path.times,
        path.event_types,
        initial=parameters,
        n_starts=2,
        seed=17,
        maxiter=300,
    )
    assert np.isfinite(fit.log_likelihood)
    assert fit.parameters.branching_ratio < 1
    assert fit.parameters.mu > 0


def test_mle_analytic_gradient_matches_finite_difference() -> None:
    path = simulate_path(HawkesParameters(0.3, 0.4, 0.7, 1.5), 100, seed=23)
    raw = np.array([-0.7, -0.3, -0.8, -0.2], dtype=np.float64)
    objective, analytic = _negative_log_likelihood_with_gradient(
        raw, path.times, path.event_types
    )
    expected = -log_likelihood(_decode(raw), path.times, path.event_types)
    assert objective == pytest.approx(expected, rel=1e-12, abs=1e-12)
    epsilon = 1e-6
    numeric = np.empty(4, dtype=np.float64)
    for index in range(4):
        offset = np.zeros(4, dtype=np.float64)
        offset[index] = epsilon
        plus, _ = _negative_log_likelihood_with_gradient(
            raw + offset, path.times, path.event_types
        )
        minus, _ = _negative_log_likelihood_with_gradient(
            raw - offset, path.times, path.event_types
        )
        numeric[index] = (plus - minus) / (2.0 * epsilon)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-5)
