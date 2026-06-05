"""Alternative-set construction (migration matrix Wave 3.2).

Continuous RURO opportunity-draw generation lives in
:mod:`dclaborsupply.alternatives.continuous`: ``generate_draws_long`` (the draw
core) and ``build_continuous_alternatives`` (thin alias). Imports are numpy/pandas
only, so importing this package stays light (no jax/gamspy/java).
"""

from dclaborsupply.alternatives.continuous import (
    build_continuous_alternatives,
    generate_draws_long,
)

__all__ = [
    "generate_draws_long",
    "build_continuous_alternatives",
]
