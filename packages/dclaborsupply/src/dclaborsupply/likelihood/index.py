"""Choice-index placeholder for RUM and RURO."""

from __future__ import annotations

from typing import Any

from dclaborsupply.spec.parser import EstimationSpec


def compute_index(
    spec: EstimationSpec,
    data: Any,
    theta: Any,
    *,
    ruro: bool,
) -> Any:
    """Compute the choice index.

    Planned decomposition:
        v = u + log_h + log_w + log_occ + log_market - log_prior

    When ``ruro=False``, the opportunity-density and importance-correction terms
    are zeroed, giving the RUM fixed-choice-set case. The implementation is
    intentionally deferred.
    """
    raise NotImplementedError("v0.1 skeleton")

