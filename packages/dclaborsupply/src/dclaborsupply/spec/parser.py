"""YAML-backed estimation spec stub."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class EstimationSpec:
    """Minimal container for raw YAML configuration.

    Full validation and schema parsing are deferred beyond the v0.1 skeleton.
    """

    raw: dict[str, Any] = field(default_factory=dict)
    source: Path | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EstimationSpec":
        """Load YAML into a raw dictionary without full scientific parsing."""
        source = Path(path)
        with source.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError("Estimation spec YAML must contain a mapping at top level.")
        return cls(raw=loaded, source=source)

