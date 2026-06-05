"""Portable validation gates (migration matrix Wave 2.5):
param-binding + synthetic recovery.

The heavy lifted modules (jax/scipy/numba via the engines/solvers/se) are imported
lazily inside the gate functions, so importing this package stays light (no
jax/gamspy/numba at import). The certified provenance gate (jax_recovery_gate.py)
is NOT part of core.
"""

from dclaborsupply.gates.param_binding import (
    base_perturbation_theta,
    check_param_binding,
)
from dclaborsupply.gates.recovery import (
    build_synthetic_recovery_objective,
    draw_synthetic_choice,
    generate_theta_star,
    hessian_pd_verdict,
    recover,
    synthesize_actual_choices,
)

__all__ = [
    "base_perturbation_theta",
    "check_param_binding",
    "build_synthetic_recovery_objective",
    "draw_synthetic_choice",
    "generate_theta_star",
    "hessian_pd_verdict",
    "recover",
    "synthesize_actual_choices",
]
