"""Allen--Cahn finite-volume value/gradient reference generation."""

from src.OpenLoop.allen_cahn.problem import AllenCahnProblem
from src.OpenLoop.allen_cahn.solver import (
    AllenCahnBatchError,
    AllenCahnConvergenceError,
    AllenCahnOpenLoopSolution,
    AllenCahnOpenLoopSolver,
    AllenCahnSampleFailure,
    AllenCahnSampleResult,
    AllenCahnSolveDiagnostics,
    AllenCahnSolverConfig,
    AllenCahnTrajectory,
)

__all__ = [
    "AllenCahnProblem", "AllenCahnOpenLoopSolution", "AllenCahnOpenLoopSolver",
    "AllenCahnSampleResult", "AllenCahnSolverConfig",
    "AllenCahnBatchError", "AllenCahnConvergenceError", "AllenCahnSampleFailure",
    "AllenCahnSolveDiagnostics", "AllenCahnTrajectory",
]
