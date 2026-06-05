"""Solvers: SciPy L-BFGS-B over the JAX joint negLL (Wave 2.3) + GAMSPy vectorized
estimation behind the optional [gamspy] extra (Wave 2.4).

jax / scipy / optimistix / gamspy are all imported lazily inside the functions, so
importing this package stays light (none load on import).
"""

from dclaborsupply.solvers.jax_optimize import (
    build_bounds_list,
    optimize_lbfgsb,
    polish_optimistix,
)
from dclaborsupply.solvers.gamspy_vectorized import (
    estimate_singles_vectorized_gamspy,
    estimate_couples_vectorized_gamspy,
    estimate_joint_vectorized_gamspy,
)

__all__ = [
    "build_bounds_list",
    "optimize_lbfgsb",
    "polish_optimistix",
    "estimate_singles_vectorized_gamspy",
    "estimate_couples_vectorized_gamspy",
    "estimate_joint_vectorized_gamspy",
]
