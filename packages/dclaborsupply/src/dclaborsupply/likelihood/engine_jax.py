"""JAX likelihood engine placeholder.

JAX is optional. This module must not import jax at module import time.
"""

from __future__ import annotations

from typing import Any


def _load_jax() -> Any:
    """Import jax lazily so base imports stay lightweight."""
    try:
        import jax  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "JAX backend requested but jax is not installed. "
            "Install with `pip install dclaborsupply[jax]`."
        ) from exc
    return jax


def log_likelihood(*args: Any, **kwargs: Any) -> Any:
    """Placeholder for the future JAX likelihood."""
    _load_jax()
    raise NotImplementedError("v0.1 skeleton")

