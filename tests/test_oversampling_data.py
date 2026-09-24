"""An added-budget dataset must preserve its original training samples."""

import runpy
from pathlib import Path

import numpy as np

from src.OpenLoop.pendulum.solver import PendulumPmpSolver
from src.OpenLoop.pendulum.trajectories import PmpTrajectory
from src.OpenLoop.value_samples import ValueSamples

additional_samples = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/investigation/make_twosided_oversampling_sets.py")
)["additional_samples"]


def test_additional_samples_exclude_existing_states_and_duplicate_pool_rows():
    x = np.column_stack((np.arange(10.0), np.zeros(10)))
    base = ValueSamples(x[:3], np.arange(3.0), np.ones((3, 2)))
    pool_x = np.vstack((x, x))
    pool = ValueSamples(pool_x, pool_x[:, 0], np.ones_like(pool_x))
    extra = additional_samples(pool, base, 5)
    augmented = ValueSamples.concatenate([base, extra])
    np.testing.assert_array_equal(augmented.x[:base.size], base.x)
    assert augmented.size == 8
    assert len(np.unique(augmented.x, axis=0)) == 8


def test_body_screen_rejects_reentrant_and_ambiguous_branches():
    x = np.column_stack(([0.0, 1.0, 2.0, 2 * np.pi + 1, 2 * np.pi + 2, 3.0, 3.01],
                         np.zeros(7)))
    v = np.array([1.0, 5.0, 3.0, 1.0, 3.0, 10.0, 1.0])
    raw = PmpTrajectory(
        boundary_angle=0.0, tau=np.arange(7.0), state=x, value=v,
        costate=np.zeros_like(x), control=np.zeros(7), hamiltonian=np.zeros(7),
    )
    candidates = np.array([0, 1, 2, 5])
    samples = ValueSamples(x[candidates], v[candidates], raw.costate[candidates])
    screened = PendulumPmpSolver().screen_body_samples(samples, (raw,))
    # Interior x=0 survives; x=1 loses to another target, x=2 is ambiguous,
    # and x=3 loses to a lower sheet of its own target branch.
    np.testing.assert_array_equal(screened.x, [[0.0, 0.0]])
