"""Application welfare implementation placeholder."""

from __future__ import annotations

from typing import Any

from dclaborsupply.welfare import WelfareProtocol


class AppWelfareCalculator(WelfareProtocol):
    """Placeholder implementation of the core welfare protocol."""

    def compute_welfare(self, data: Any, result: Any, **kwargs: Any) -> Any:
        """Compute welfare once application logic exists."""
        raise NotImplementedError("v0.1 skeleton")

