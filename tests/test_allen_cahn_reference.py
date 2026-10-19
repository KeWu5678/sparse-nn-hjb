"""Check the nonlinear reference against its independent small-state LQR limit."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from src.OpenLoop.allen_cahn import AllenCahnProblem
from src.OpenLoop.collocation import CollocationConfig, CollocationSolver


@pytest.mark.parametrize("dimension", [6, 10])
def test_small_state_value_and_costate_match_finite_horizon_riccati_solution(dimension):
    h = 2.0 / dimension
    # Assemble the linearized problem independently, from conservative fluxes
    # and the exact actuator-cell overlaps of these two meshes.
    differences = np.diff(np.eye(dimension), axis=0) / h
    a = np.eye(dimension) - 0.1 * differences.T @ differences
    if dimension == 6:
        b = np.array([[.1, 0, 0], [.8, 0, 0], [0, .6, 0],
                      [0, .6, 0], [0, 0, .8], [0, 0, .1]])
    else:
        b = np.array([[0, 0, 0], [.5, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0],
                      [0, 1, 0], [0, 0, 0], [0, 0, 1], [0, 0, .5], [0, 0, 0]])
    q, control_gram = h * np.eye(dimension), b @ b.T / .01

    def riccati_rhs(_time, flattened):
        p = flattened.reshape(dimension, dimension)
        return (-a.T @ p - p @ a + p @ control_gram @ p - q).ravel()

    reference = solve_ivp(
        riccati_rhs, (3., 0.), np.zeros(dimension**2), rtol=1e-11, atol=1e-13,
    )
    assert reference.success
    p = reference.y[:, -1].reshape(dimension, dimension)
    initial = 1e-4 * np.linspace(-.6, 1., dimension)
    result = CollocationSolver(
        AllenCahnProblem(dimension=dimension),
        CollocationConfig(tol=1e-10),
    ).solve_sample(initial)
    # The cubic drift changes these small-state labels only at higher order.
    assert result.value == pytest.approx(initial @ p @ initial, rel=2e-7, abs=1e-16)
    np.testing.assert_allclose(result.gradient, 2 * p @ initial, rtol=2e-7, atol=1e-12)
