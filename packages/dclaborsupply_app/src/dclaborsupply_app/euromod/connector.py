"""EUROMOD pricing connector: injection Protocol + result type + lazy real connector.

The generic runner (``runner.py``) depends only on the :class:`PricingConnector`
Protocol, so it never imports EUROMOD / .NET / Java. The real :class:`EuromodConnector`
imports ``euromod`` LAZILY inside ``run()``; therefore ``import dclaborsupply_app.euromod``
succeeds with no EUROMOD runtime installed, and a documented :class:`ImportError` is
raised only when the real connector is actually invoked without it.

App layer only — no core / france / MNL imports; ``euromod`` is NOT a base dependency.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Protocol, runtime_checkable

import pandas as pd


@dataclass
class PricingConnectorResult:
    """Raw EUROMOD output of a single connector run.

    ``output`` is the per-person EUROMOD output DataFrame (one row per input person,
    carrying the synthetic IDs the runner sent and the raw nominal ``ils_dispy``).
    ``warnings`` / ``errors`` are the non-fatal message strings surfaced by the run
    (fatal failures raise instead).
    """
    output: pd.DataFrame
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@runtime_checkable
class PricingConnector(Protocol):
    """A thing that prices a complete EUROMOD input DataFrame.

    Implementations must NOT mutate ``data``; they return raw per-person output plus
    any warning/error messages. The generic runner is written against this Protocol
    only, so EUROMOD/Java stay out of the runner and out of package import.
    """

    def run(
        self,
        data: pd.DataFrame,
        *,
        country: str,
        system: str,
        dataset: str,
    ) -> PricingConnectorResult:
        ...


_EUROMOD_IMPORT_HELP = (
    "EuromodConnector.run requires the 'euromod' package and its .NET (CoreCLR) runtime, "
    "which are NOT base dependencies of dclaborsupply_app. Install the EUROMOD connector "
    "into this environment (and the EUROMOD model release) to price for real, or inject a "
    "different PricingConnector (e.g. a fake/stub connector) into the runner for tests."
)


class EuromodConnector:
    """Real connector around the ``euromod`` package (validated scratch pattern).

    Mirrors ``Model(root)[country][system].run(df, dataset, outputpath="")`` — output is
    kept in memory (``outputpath=""``) so no EUROMOD Output/Log files are written. The
    ``euromod`` import is lazy (inside :meth:`run`), so importing this module needs no
    EUROMOD runtime.
    """

    def __init__(self, model_root: str, *, runtime: str = "coreclr") -> None:
        self.model_root = str(model_root)
        self.runtime = runtime

    def run(
        self,
        data: pd.DataFrame,
        *,
        country: str,
        system: str,
        dataset: str,
        verbose: bool = True,
        nowarnings: bool = False,
    ) -> PricingConnectorResult:
        os.environ.setdefault("PYTHONNET_RUNTIME", self.runtime)
        try:
            import euromod as em  # noqa: PLC0415  (lazy by design)
        except Exception as exc:  # pragma: no cover - exercised via monkeypatch in tests
            raise ImportError(_EUROMOD_IMPORT_HELP) from exc

        model = em.Model(self.model_root)
        sysobj = model[country][system]
        sim = sysobj.run(data, dataset, outputpath="", verbose=verbose, nowarnings=nowarnings)
        output = sim.outputs[0]
        messages = [str(m) for m in (getattr(sim, "errors", None) or [])]
        warnings = [m for m in messages if m.strip().lower().startswith("warning")]
        errors = [m for m in messages if not m.strip().lower().startswith("warning")]
        return PricingConnectorResult(output=output, warnings=warnings, errors=errors)
