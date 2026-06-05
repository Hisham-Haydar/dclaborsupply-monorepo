"""Core welfare protocol.

The core package defines the interface only. EUROMOD, France-specific welfare,
and decomposition logic belong in the application package.
"""

from __future__ import annotations

from typing import Any, Protocol


class WelfareProtocol(Protocol):
    """Protocol implemented by application-layer welfare calculators."""

    def compute_welfare(self, data: Any, result: Any, **kwargs: Any) -> Any:
        """Compute welfare outputs in an application package."""
        ...

