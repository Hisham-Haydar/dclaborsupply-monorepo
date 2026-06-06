"""EUROMOD application layer: connector-injected pricing runner.

Importing this package pulls NO EUROMOD / pythonnet / Java — the real
:class:`EuromodConnector` imports ``euromod`` lazily inside ``run()``.
"""
from __future__ import annotations

from .connector import (
    EuromodConnector,
    PricingConnector,
    PricingConnectorResult,
)
from .runner import (
    EarningsMutationPolicy,
    EuromodPricingRunner,
    PricingColumns,
    PricingResult,
    WEEKS_PER_MONTH,
)

__all__ = [
    "PricingConnector",
    "PricingConnectorResult",
    "EuromodConnector",
    "EarningsMutationPolicy",
    "EuromodPricingRunner",
    "PricingColumns",
    "PricingResult",
    "WEEKS_PER_MONTH",
]
