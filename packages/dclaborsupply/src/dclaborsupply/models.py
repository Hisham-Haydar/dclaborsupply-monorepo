"""Front-end model dataclasses for the v0.1 skeleton."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dclaborsupply.spec.parser import EstimationSpec


@dataclass(slots=True)
class Result:
    """Placeholder estimation result container."""

    params: dict[str, Any] = field(default_factory=dict)
    blocks: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> None:
        """Summaries are deferred until estimator logic exists."""
        raise NotImplementedError("v0.1 skeleton")

    def predict(self, data: Any) -> Any:
        """Prediction is deferred until estimator logic exists."""
        raise NotImplementedError("v0.1 skeleton")


@dataclass(slots=True)
class RUMModel:
    """RUM front-end for fixed choice-set workflows."""

    utility: str = "box_cox"
    choice_col: str = "chosen"
    unit_col: str = "idorighh"
    spec: EstimationSpec | None = None

    @classmethod
    def from_spec(cls, spec: EstimationSpec) -> "RUMModel":
        """Construct a RUM model from an existing spec stub."""
        return cls(spec=spec)

    def fit(
        self,
        data: Any,
        *,
        backend: str = "jax",
        warm_start: Any | None = None,
    ) -> Result:
        """Fit is intentionally not implemented in the skeleton."""
        raise NotImplementedError("v0.1 skeleton")


@dataclass(slots=True)
class RUROModel:
    """RURO front-end for latent-opportunity workflows."""

    utility: str = "box_cox"
    opportunity: Any | None = None
    correction: str = "importance_sampling"
    choice_col: str = "chosen"
    unit_col: str = "idorighh"
    spec: EstimationSpec | None = None

    @classmethod
    def from_spec(cls, spec: EstimationSpec) -> "RUROModel":
        """Construct a RURO model from an existing spec stub."""
        return cls(spec=spec)

    def fit(
        self,
        data: Any,
        *,
        backend: str = "jax",
        warm_start: Any | None = None,
    ) -> Result:
        """Fit is intentionally not implemented in the skeleton."""
        raise NotImplementedError("v0.1 skeleton")

