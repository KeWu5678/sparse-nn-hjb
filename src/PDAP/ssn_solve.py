"""Run a fresh SSN coefficient solve and commit the result to the model."""

import logging
from dataclasses import dataclass

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader

from ..problems import SparseProblem
from ..SSN import SSN

logger = logging.getLogger(__name__)

# Selected by the 24-cell VDP/pendulum pilot in experiments/algorithm2_rho_pilot.md.
ALGORITHM2_PROX_RHO = 0.5


@dataclass(frozen=True)
class SolverConfig:
    """How the SSN outer solve is run (globalization + line-search tolerances)."""

    method: str = "levenberg_marquardt"
    max_ls_iter: int = 500
    tolerance_ls: float = 1.0 + 1e-8
    tolerance_grad: float = 0.0
    sigmamax: float = 10.0



def ssn_solve(model, problem: SparseProblem, solver: SolverConfig, train_loader: DataLoader, *,
              iterations: int, verbose: bool = False) -> float:
    """Refresh the support-dependent objective, solve, and update outer weights."""
    prepared = problem.prepare_coefficients(model, train_loader)
    params = [p for p in model.parameters() if p.requires_grad]
    coefficients = parameters_to_vector(params).detach().clone()
    theta = torch.nn.Parameter(problem.coefficient_coordinates(coefficients, prepared).clone())
    prox_scale = problem.warmstart_prox_scale(theta.detach(), ALGORITHM2_PROX_RHO)
    initial_active = problem.active_magnitudes(theta.detach())
    optimizer = SSN(
        [theta], problem, prepared, method=solver.method,
        max_ls_iter=solver.max_ls_iter, tolerance_ls=solver.tolerance_ls,
        tolerance_grad=solver.tolerance_grad, sigmamax=solver.sigmamax,
        prox_scale=prox_scale,
    )
    prev = float(problem.loss(theta, prepared).detach())
    failed_steps = 0
    for _ in range(iterations):
        prev = float(optimizer.step().detach())
        failed_steps += int(not optimizer.last_step_success)
    vector_to_parameters(problem.model_coefficients(theta.detach(), prepared), params)
    if verbose and prox_scale is not None:
        logger.info(
            "Algorithm 2 correction  q=%.6g  rho=%.3g  min|c|=%.6e  "
            "prox_scale=%.6e  inverse_step=%.6e  nonzero=%d->%d  "
            "failed_steps=%d/%d  train_loss=%.6e",
            problem.penalty_exponent, ALGORITHM2_PROX_RHO, float(initial_active.min()),
            prox_scale, problem.inverse_step(prox_scale), int(initial_active.numel()),
            int(problem.active_magnitudes(theta.detach()).numel()),
            failed_steps, iterations, prev,
        )
    if verbose:
        logger.debug("Output-weight solve complete  train_loss=%.6e", prev)
    return prev
