"""Focused tests for the candidate-search mechanics of Algorithms 1 and 2."""

from __future__ import annotations

import math

import pytest
import torch

import src.PDAP.insertion as insertion


def _inputs():
    X = torch.zeros((1, 1), dtype=torch.float64)
    residual_v = torch.ones((1, 1), dtype=torch.float64)
    residual_dv = torch.zeros((1, 1), dtype=torch.float64)
    return X, residual_v, residual_dv


def _starts(values):
    points = torch.as_tensor(values, dtype=torch.float64)

    def sample_sphere(n):
        assert n == points.shape[0]
        return points[:, :1].clone(), points[:, 1].clone()

    return sample_sphere


def _fake_lbfgs(monkeypatch, outputs):
    class FakeLBFGS:
        instances = []
        queued = [torch.as_tensor(out, dtype=torch.float64) for out in outputs]

        def __init__(self, params, **_kwargs):
            self.param = params[0]
            self.instances.append(self)

        def zero_grad(self):
            self.param.grad = None

        def step(self, _closure):
            if self.queued:
                with torch.no_grad():
                    self.param.copy_(self.queued.pop(0))

    monkeypatch.setattr(insertion.torch.optim, "LBFGS", FakeLBFGS)
    return FakeLBFGS


def _generate(
    monkeypatch,
    *,
    outputs,
    starts=None,
    radius=10.0,
    merge_tol=1e-2,
    existing_atoms=None,
    use_sphere=False,
    normalized=None,
):
    if starts is None:
        starts = [[1.0, 0.0] for _ in outputs]
    fake = _fake_lbfgs(monkeypatch, outputs)
    X, residual_v, residual_dv = _inputs()
    a, b, n = insertion._generate_candidates(
        X,
        residual_v,
        residual_dv,
        activation=torch.tanh,
        power=1.0,
        loss_weights=(1.0, 1.0),
        sample_sphere=_starts(starts),
        N=len(starts),
        merge_tol=merge_tol,
        two_sided=True,
        use_sphere=use_sphere,
        existing_atoms=existing_atoms,
        normalized=(not use_sphere) if normalized is None else normalized,
        radius=radius,
    )
    return a, b, n, fake


@pytest.mark.parametrize(
    ("radius", "outside"),
    [(1.0, 2.0), (None, math.exp(5.0) + 1.0)],
)
def test_algorithm1_discards_candidates_outside_sampling_radius(
    monkeypatch, radius, outside
):
    a, b, n, _ = _generate(
        monkeypatch, outputs=[[outside, 0.0]], radius=radius
    )
    assert n == 0
    assert a.shape == (0, 1)
    assert b.shape == (0,)


def test_algorithm1_ignores_existing_atoms_as_search_starts(monkeypatch):
    existing = (
        torch.tensor([[3.0]], dtype=torch.float64),
        torch.tensor([4.0], dtype=torch.float64),
    )
    _, _, n, fake = _generate(
        monkeypatch,
        outputs=[[1.0, 0.0], [0.0, 1.0]],
        starts=[[1.0, 0.0], [0.0, 1.0]],
        existing_atoms=existing,
    )
    assert n == 2
    assert len(fake.instances) == 2


def test_algorithm1_optimizes_each_random_start_once(monkeypatch):
    _, _, n, fake = _generate(
        monkeypatch,
        outputs=[[1.0, 0.0], [1.0, 0.0]],
        starts=[[1.0, 0.0], [1.0, 0.0]],
    )
    assert n == 1
    assert len(fake.instances) == 2


def test_all_nonhomogeneous_objectives_use_one_joint_solve_per_start(monkeypatch):
    _, _, n, fake = _generate(
        monkeypatch,
        outputs=[[1.0, 0.0], [0.0, 1.0]],
        starts=[[1.0, 0.0], [0.0, 1.0]],
        normalized=False,
    )
    assert n == 2
    assert len(fake.instances) == 2


def test_algorithm1_deduplicates_by_absolute_parameter_distance(monkeypatch):
    a, b, n, _ = _generate(
        monkeypatch,
        outputs=[[1.0, 0.0], [2.0, 0.0]],
    )
    assert n == 2
    assert torch.allclose(
        a.reshape(-1), torch.tensor([1.0, 2.0], dtype=torch.float64)
    )
    assert torch.allclose(b, torch.zeros(2, dtype=torch.float64))

    a, _, n, _ = _generate(
        monkeypatch,
        outputs=[[1.0, 0.0], [1.005, 0.0]],
    )
    assert n == 1
    assert a.item() == pytest.approx(1.0)


def test_algorithm1_filters_radius_before_first_kept_deduplication(monkeypatch):
    a, _, n, _ = _generate(
        monkeypatch,
        outputs=[[1.005, 0.0], [0.999, 0.0]],
        radius=1.0,
    )
    assert n == 1
    assert a.item() == pytest.approx(0.999)


def test_algorithm2_keeps_existing_atoms_as_sphere_starts(monkeypatch):
    existing = (
        torch.tensor([[-1.0]], dtype=torch.float64),
        torch.tensor([0.0], dtype=torch.float64),
    )
    a, b, n, fake = _generate(
        monkeypatch,
        outputs=[],
        starts=[[1.0, 0.0], [0.0, 1.0]],
        radius=None,
        existing_atoms=existing,
        use_sphere=True,
    )
    omega = torch.cat([a, b.reshape(-1, 1)], dim=1)
    assert n == 3
    assert len(fake.instances) == 3
    assert torch.allclose(
        torch.linalg.vector_norm(omega, dim=1),
        torch.ones(3, dtype=torch.float64),
    )
