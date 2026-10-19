"""Problem-independent behavior of the shared Pontryagin collocation solver."""

import numpy as np
import pytest

from src.OpenLoop.allen_cahn import AllenCahnProblem
from src.OpenLoop.collocation import (
    BatchError,
    CollocationConfig,
    CollocationSolver,
    ConvergenceError,
    SampleFailure,
    SampleResult,
)

UNSOLVABLE = CollocationConfig(initial_mesh_points=2, collocation_max_nodes=2, collocation_tol=1e-10)


def test_failed_collocation_raises_with_initial_state_and_diagnostics():
    solver = CollocationSolver(AllenCahnProblem(dimension=6), UNSOLVABLE)
    initial = np.linspace(-.7, .9, 6)
    with pytest.raises(ConvergenceError) as caught:
        solver.solve_sample(initial)
    failure = caught.value.failure
    np.testing.assert_array_equal(failure.initial_state, initial)
    assert failure.diagnostics.attempts == 2
    assert failure.diagnostics.nodes == 2


@pytest.mark.parametrize("workers", [1, 2])
def test_failed_batch_attempts_every_state_and_reports_ordered_outcomes(workers):
    solver = CollocationSolver(AllenCahnProblem(dimension=6), UNSOLVABLE)
    states = np.vstack((np.zeros(6), np.linspace(-.7, .9, 6), np.linspace(.8, -.6, 6)))
    progress = []
    with pytest.raises(BatchError) as caught:
        solver.solve(states, lambda current, total: progress.append((current, total)), workers=workers)
    batch = caught.value
    assert len(batch.sample_results) == len(states)
    assert isinstance(batch.sample_results[0], SampleResult)
    assert all(isinstance(result, SampleFailure) for result in batch.sample_results[1:])
    np.testing.assert_array_equal(batch.failed_initial_states, states[1:])
    np.testing.assert_array_equal([result.initial_state for result in batch.sample_results], states)
    assert progress == [(0, 3), (1, 3), (2, 3), (3, 3)]


def test_retained_samples_and_progress_preserve_dimension():
    solver = CollocationSolver(AllenCahnProblem(dimension=10))
    progress = []
    states = np.vstack((np.zeros(10), np.linspace(-.1, .1, 10)))
    solution = solver.solve(states, lambda current, total: progress.append((current, total)))
    np.testing.assert_array_equal(solution.value_samples.x, states)
    assert solution.value_samples.dv.shape == (2, 10)
    assert progress == [(0, 2), (1, 2), (2, 2)]


@pytest.mark.parametrize("dimension", [6, 10])
def test_parallel_batch_matches_sequential_values_gradients_and_order(dimension):
    solver = CollocationSolver(AllenCahnProblem(dimension=dimension))
    states = np.vstack((np.zeros(dimension), np.linspace(-.3, .2, dimension)))
    sequential = solver.solve(states)
    parallel = solver.solve(states, workers=2)
    for name in ("x", "v", "dv"):
        np.testing.assert_array_equal(getattr(parallel.value_samples, name), getattr(sequential.value_samples, name))


@pytest.mark.parametrize("kwargs", [{"collocation_tol": np.nan}, {"initial_mesh_points": 1}, {"collocation_max_nodes": 4}, {"quadrature_order": 2.5}, {"initial_guess_decays": ()}, {"initial_guess_decays": (np.inf,)}])
def test_solver_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        CollocationConfig(**kwargs)


def test_solver_rejects_invalid_initial_states():
    solver = CollocationSolver(AllenCahnProblem())
    for state in (np.zeros(10), np.full(6, np.nan)):
        with pytest.raises(ValueError):
            solver.solve_sample(state)
