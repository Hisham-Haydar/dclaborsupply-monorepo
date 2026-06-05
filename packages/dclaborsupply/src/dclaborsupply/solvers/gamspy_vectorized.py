"""GAMSPy solver placeholder.

GAMSPy is an optional extra and is imported lazily only when requested.
"""

from __future__ import annotations

from typing import Any


def _load_gamspy() -> Any:
    try:
        import gamspy  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "GAMSPy solver requested but gamspy is not installed. "
            "Install with `pip install dclaborsupply[gamspy]`."
        ) from exc
    return gamspy


def estimate_with_gamspy(*args: Any, **kwargs: Any) -> Any:
    """Placeholder for future vectorized GAMSPy estimation."""
    _load_gamspy()
    raise NotImplementedError("v0.1 skeleton")

