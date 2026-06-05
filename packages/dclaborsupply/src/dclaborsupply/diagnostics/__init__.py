"""Model-generic diagnostics (migration matrix Wave 2.6).

The bundle data structures, metric registry, solver-family classification, the
finite-metric assembly (``build_diagnostics_bundle``), and the generic CONOPT text
parsers. The HTML/Markdown renderers and the artifact writer were NOT lifted
(app/report layer). Imports are stdlib + numpy, so importing this package stays
light (no jax/gamspy/java).
"""

from dclaborsupply.diagnostics.bundle import (
    DEFAULT_PROFILE,
    PROFILE_CHOICES,
    SOLVER_FAMILY_BFGS,
    SOLVER_FAMILY_CONOPT,
    SOLVER_FAMILY_IPOPT,
    SOLVER_FAMILY_KNITRO,
    SOLVER_FAMILY_OTHER,
    SOLVER_FAMILY_TRUST_CONSTR,
    SOLVER_FAMILY_UNKNOWN,
    DiagnosticsBundle,
    MetricSpec,
    Section,
    build_diagnostics_bundle,
    classify_solver_family,
    get_metric,
    parse_conopt_rgmax_from_text,
    parse_conopt_termination_text,
    parse_conopt_trace_from_text,
)

__all__ = [
    "build_diagnostics_bundle",
    "DiagnosticsBundle",
    "Section",
    "MetricSpec",
    "get_metric",
    "classify_solver_family",
    "parse_conopt_rgmax_from_text",
    "parse_conopt_termination_text",
    "parse_conopt_trace_from_text",
    "PROFILE_CHOICES",
    "DEFAULT_PROFILE",
    "SOLVER_FAMILY_CONOPT",
    "SOLVER_FAMILY_IPOPT",
    "SOLVER_FAMILY_KNITRO",
    "SOLVER_FAMILY_BFGS",
    "SOLVER_FAMILY_TRUST_CONSTR",
    "SOLVER_FAMILY_OTHER",
    "SOLVER_FAMILY_UNKNOWN",
]
