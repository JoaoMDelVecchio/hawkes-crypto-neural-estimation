import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_addendum_finetune import holm  # noqa: E402
from run_grid_bias import BETAS, MUS, invert  # noqa: E402


def test_holm_matches_step_down_and_stays_monotone() -> None:
    adjusted = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted["a"] == pytest.approx(0.03)
    assert adjusted["c"] == pytest.approx(0.06)
    assert adjusted["b"] == pytest.approx(0.06)
    assert holm({"a": 0.6, "b": 0.9}) == {"a": 1.0, "b": 1.0}


def test_grid_inversion_recovers_a_known_forward_map() -> None:
    grid = (np.log(MUS), np.log(BETAS))
    log_mu, log_beta = np.meshgrid(*grid, indexing="ij")
    forward_mu = RegularGridInterpolator(grid, log_mu)
    forward_beta = RegularGridInterpolator(grid, log_beta + np.log(0.6))
    mu, beta, inside = invert(forward_mu, forward_beta, 0.23, 0.6 * 4.5)
    assert inside
    assert mu == pytest.approx(0.23, rel=0.03)
    assert beta == pytest.approx(4.5, rel=0.03)
    *_, inside = invert(forward_mu, forward_beta, 0.23, 50.0)
    assert not inside
