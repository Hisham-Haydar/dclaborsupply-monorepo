"""JAX optimizer placeholder.

JAX is optional and imported lazily only when this placeholder is called.
"""

from __future__ import annotations

from typing import Any


def _load_jax() -> Any:
    try:
        import jax  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "JAX optimizer requested but jax is not installed. "
            "Install with `pip install dclaborsupply[jax]`."
        ) from exc
    return jax


def optimize(*args: Any, **kwargs: Any) -> Any:
    """Placeholder for future JAX optimization."""
    _load_jax()
    raise NotImplementedError("v0.1 skeleton")

