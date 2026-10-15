from typing import Optional

import torch


class MPCG:
    """Modified projected CG for the normal-map Newton system.

    The operators are evaluated at the current normal-map variable q.

    Args:
        jac_normal: Selected derivative G′(q) of the optimality normal map.
            This matrix need not be symmetric.
        jac_prox: Symmetric positive semidefinite derivative P′(q) of the proximal
            map. jac_prox @ jac_normal must be symmetric. None uses the identity.
    """

    def __init__(
        self, jac_normal: torch.Tensor, jac_prox: Optional[torch.Tensor] = None
    ) -> None:
        self.jac_normal = jac_normal
        self.jac_prox = jac_prox

    def solve(
        self,
        r: torch.Tensor,
        rtol: float = 1e-3,
        maxit: int = 100,
        sigma: float = 0.0,
    ) -> tuple[torch.Tensor, str]:
        """Solve jac_normal @ step = r, truncating on curvature or the radius.

        Each call initializes the CG iteration from a zero step.

        Args:
            r: Initial residual −G(q), equal to the Newton right-hand side.
            rtol: Relative tolerance for predicted reduction and residual checks.
            maxit: Maximum CG iterations, excluding the final residual correction.
            sigma: Trust-region radius (<= 0 disables the radius constraint).

        Returns:
            step: Newton increment in q coordinates.
            stop_reason: 'convgd', 'maxitr', 'negdef' (nonpositive curvature),
                or 'radius'.
        """
        # All working values belong to this solve.
        step = torch.zeros_like(r)
        if not bool(torch.any(r)):
            return step, "convgd"

        jac_prox_d = r if self.jac_prox is None else (self.jac_prox @ r)
        rho = torch.dot(r, jac_prox_d)
        d = r
        pred_step = r.new_tensor(-0.0)
        pred = r.new_tensor(-0.0)
        res0 = torch.sqrt(torch.clamp(rho, min=0.0))

        iterations = 0
        stop_reason = "convgd"

        # Run CG with the MATLAB stopping rules and truncation conditions.
        while not self._cg_converged(pred_step, pred, rho, rtol, res0):
            iterations += 1
            if iterations > maxit:
                stop_reason = "maxitr"
                break

            jac_normal_d = self.jac_normal @ d
            gamma = torch.dot(jac_normal_d, jac_prox_d)

            if self._is_neg_def(gamma):
                if sigma > 0:
                    step = self._to_boundary(step, d, sigma)
                return step, "negdef"

            alpha = rho / (gamma + 1e-30)
            step_new = step + alpha * d
            if self._exceed_radius(step_new, sigma):
                return self._to_boundary(step, d, sigma), "radius"

            step = step_new
            pred_step = -0.5 * (alpha * rho)
            r, d, jac_prox_d, rho = self._cg_update(
                r, d, jac_prox_d, rho, alpha, jac_normal_d
            )
            pred = pred + pred_step

        # Correct the full residual by minimizing ||r - theta * jac_normal @ r||².
        jac_normal_r = self.jac_normal @ r
        numerator = torch.dot(r, jac_normal_r)
        denom = torch.dot(jac_normal_r, jac_normal_r) + 1e-30
        theta = numerator / denom

        step_new = step + theta * r
        if self._exceed_radius(step_new, sigma):
            stop_reason = "radius"
            step = self._to_boundary(step, r, sigma)
        else:
            step = step_new

        return step, stop_reason

    @staticmethod
    def _cg_converged(
        pred_step: torch.Tensor,
        pred: torch.Tensor,
        rho: torch.Tensor,
        rtol: float,
        res0: torch.Tensor,
    ) -> bool:
        """Check the inner CG model-decrease and weighted-residual tolerances."""
        left = torch.sqrt(torch.clamp(-pred_step, min=0.0))
        right = torch.sqrt(torch.clamp(-pred, min=0.0)) * rtol
        model_decrease_tol_met = not bool(left >= right)
        residual_tol_met = not bool(
            torch.sqrt(torch.clamp(rho, min=0.0)) >= res0 * (1e-2 * rtol)
        )
        return model_decrease_tol_met or residual_tol_met

    @staticmethod
    def _is_neg_def(gamma: torch.Tensor) -> bool:
        """Test nonpositive curvature along d, including zero curvature."""
        return bool(gamma <= 0)

    @staticmethod
    def _exceed_radius(step: torch.Tensor, sigma: float) -> bool:
        """Test whether a step lies outside an enabled trust region."""
        return sigma > 0 and bool(torch.norm(step) > sigma)

    def _cg_update(
        self,
        r: torch.Tensor,
        d: torch.Tensor,
        jac_prox_d: torch.Tensor,
        rho: torch.Tensor,
        alpha: torch.Tensor,
        jac_normal_d: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the next residual, direction, weighted direction, and rho."""
        r = r - alpha * jac_normal_d
        jac_prox_r = r if self.jac_prox is None else (self.jac_prox @ r)
        rho_prev = rho
        rho = torch.dot(r, jac_prox_r)

        beta = -(alpha * torch.dot(jac_normal_d, jac_prox_r)) / (rho_prev + 1e-30)
        d = r + beta * d
        jac_prox_d = jac_prox_r + beta * jac_prox_d
        return r, d, jac_prox_d, rho

    @staticmethod
    def _to_boundary(step: torch.Tensor, d: torch.Tensor, sigma: float) -> torch.Tensor:
        """Return the boundary step along d, starting inside the trust region."""
        dd = torch.dot(d, d)
        xd = torch.dot(step, d)
        xx = torch.dot(step, step)

        # Discriminant of the quadratic equation for tau.
        det = xd * xd + dd * (sigma * sigma - xx)
        det = torch.clamp(det, min=0.0)
        tau = (sigma * sigma - xx) / (xd + torch.sqrt(det + 0.0))
        return step + tau * d


__all__ = ["MPCG"]
