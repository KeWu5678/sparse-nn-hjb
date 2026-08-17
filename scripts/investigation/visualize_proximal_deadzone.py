"""Visualize the global fractional-power proximal map used by Algorithm 2.

The current solver supports q in {1/2, 2/3}.  This diagnostic uses q=1/2 and
shows the distinction between the stationary fold and the later global
objective switch.  The implementation selects zero at the exact switch.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.SSN.prox import power_prox
from src.paths import PLOTS_DIR


alpha = 1e-5
gamma = 0.0
activation_power = 3.0
q = 2.0 / (activation_power + 1.0)
rho_prox = 0.5
minimum_warm_weight = 0.1

baseline_mu = alpha / (1.0 + alpha * gamma)
warm_mu_bound = (
    rho_prox
    * minimum_warm_weight ** (2.0 - q)
    / (2.0 * (1.0 - q))
)
mu = min(baseline_mu, warm_mu_bound)

turn_output = (mu * q * (1.0 - q)) ** (1.0 / (2.0 - q))
turn_input = turn_output + mu * q * turn_output ** (q - 1.0)
switch_output = (2.0 * mu * (1.0 - q)) ** (1.0 / (2.0 - q))
switch_input = switch_output + mu * q * switch_output ** (q - 1.0)

print(f"activation power: {activation_power:g}")
print(f"penalty exponent: {q:g}")
print(f"baseline proximal scale: {baseline_mu:.6e}")
print(f"warm-start upper bound: {warm_mu_bound:.6e}")
print(f"proximal scale used: {mu:.6e}")
print(f"stationary fold input: {turn_input:.6e}")
print(f"global switch input: {switch_input:.6e}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

# Stationary equation.  The fold creates two positive stationary roots, but
# the larger root becomes globally preferable to zero only at switch_input.
ax = axes[0]
t = np.linspace(max(turn_output * 1e-4, 1e-14), 5.0 * switch_output, 2000)
stationary_input = t + mu * q * t ** (q - 1.0)
ax.plot(t, stationary_input, lw=2.0)
ax.scatter([turn_output], [turn_input], color="tab:orange", zorder=3)
ax.scatter([switch_output], [switch_input], color="tab:red", zorder=3)
ax.axhline(turn_input, color="tab:orange", ls="--", lw=1.2, label="stationary fold")
ax.axhline(switch_input, color="tab:red", ls="--", lw=1.2, label="global switch")
ax.set_xlabel("positive output t")
ax.set_ylabel("proximal input")
ax.set_title("Stationary equation")
ax.legend()
ax.grid(alpha=0.25)

# At the global switch, zero and the positive stationary point have equal
# objective value.  The deterministic implementation chooses zero at the tie.
ax = axes[1]
t_obj = np.linspace(0.0, 3.0 * switch_output, 2000)
for factor, color in ((0.9, "tab:blue"), (1.0, "tab:orange"), (1.1, "tab:green")):
    center = factor * switch_input
    objective = 0.5 * (t_obj - center) ** 2 + mu * t_obj ** q
    ax.plot(
        t_obj,
        objective - objective.min(),
        color=color,
        lw=2.0,
        label=f"input = {factor:.1f} × switch",
    )
ax.set_xlabel("candidate output t")
ax.set_ylabel("objective above its minimum")
ax.set_title("Scalar proximal objective")
ax.legend()
ax.grid(alpha=0.25)

# Evaluate the production closed form rather than duplicating its root solver.
ax = axes[2]
inputs = np.linspace(0.0, 4.0 * switch_input, 1200)
prox_values = power_prox(
    torch.as_tensor(inputs, dtype=torch.float64),
    mu,
    q=q,
).numpy()
ax.plot(inputs, prox_values, lw=2.2, color="tab:green")
ax.axvline(switch_input, color="tab:red", ls="--", lw=1.2)
ax.scatter([switch_input], [0.0], color="tab:red", zorder=3, label="tie selects zero")
ax.set_xlabel("proximal input")
ax.set_ylabel("global proximal output")
ax.set_title("Production global prox")
ax.legend()
ax.grid(alpha=0.25)

fig.suptitle(
    "Fractional-power global proximal map "
    f"(q={q:g}, scale={mu:.2e}, warm factor={rho_prox:g})"
)
fig.tight_layout()
output = PLOTS_DIR / "proximal_deadzone.png"
fig.savefig(output, dpi=180, bbox_inches="tight")
print(f"figure saved to {output}")
