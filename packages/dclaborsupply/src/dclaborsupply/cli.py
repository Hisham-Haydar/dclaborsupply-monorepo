"""Command-line interface for dclaborsupply v0.1.

``estimate`` supports ONLY a documented synthetic-fixture config (``cli.mode:
synthetic``): real dataframe/parquet loading (dataframe -> PrecomputedData) is
deferred, because ``models.py`` currently accepts precomputed engine data only.
The CLI dispatches only to ``spec.parser`` + ``models``; no likelihood/optimizer/
score/SE/data-prep math lives here. Heavy deps (numpy/numba via the engine
containers, jax/scipy via the lifted solver/SE) are imported LAZILY inside the
estimate handler, so ``import dclaborsupply.cli`` stays light.

Model dispatch: prefer the explicit ``cli.model: rum|ruro``. If absent, RURO is
inferred ONLY when opportunity blocks (``market_opportunity`` shifters) are present
in the spec; otherwise RUM. Never guessed from file names.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from dclaborsupply.config.loader import load_yaml

SKELETON_MESSAGE = "NotImplementedError: v0.1 skeleton"

# Synthetic-fixture defaults (v0.1 demo path).
_DEFAULT_N_GROUPS = 600
_DEFAULT_N_ALTS = 6


def _cmd_validate_config(args: argparse.Namespace) -> int:
    config = load_yaml(Path(args.path))
    keys = ", ".join(sorted(config.keys())) if config else "(no keys)"
    print(f"Loaded config keys: {keys}")
    return 0


def _build_synthetic_singles(*, is_male: bool, seed: int, n_groups: int, n_alts: int):
    """Small precomputed singles fixture for the v0.1 synthetic estimate path.

    Mirrors the Wave-3.3 test fixture (positive consumption/leisure/log_wage, all
    shifters zero, non-working baseline). PrecomputedData is imported lazily here
    because it pulls numba — keeping ``import dclaborsupply.cli`` light.
    """
    import numpy as np
    from dclaborsupply.likelihood._numpy_primitives import PrecomputedDataSingles

    rng = np.random.default_rng(seed)
    n = n_groups * n_alts
    z = lambda: np.zeros(n)  # noqa: E731
    c = 0.5 + 4.5 * rng.random(n)
    l = 0.5 + 4.5 * rng.random(n)
    st = np.arange(0, n, n_alts)
    return PrecomputedDataSingles(
        consumption=c, leisure=l, log_c=np.log(c), log_l=np.log(l),
        age_norm=z(), age_norm2=z(), n_children=z(), educL=z(), educM=z(), educH=z(),
        working=z(), working_pt1=z(), working_pt2=z(), working_ft=z(), working_lh=z(), gsur=z(),
        female=z(), in_couple=z(), drgn1=z(),
        reg2=z(), reg3=z(), reg4=z(), reg5=z(), reg6=z(), reg7=z(), reg8=z(),
        drgur=z(), drgmd=z(), drgru=z(),
        year_2015_indicator=z(), year_2017_indicator=z(),
        log_wage=1.0 + rng.random(n), pexp_years=None, pexp_years2=None,
        loc4=None, loc4_1=None, loc4_2=None, loc4_3=None, loc4_4=None,
        prior=1.0 + rng.random(n), c_scale=1.0, l_scale=1.0,
        group_ids=np.arange(n_groups), group_starts=st, group_ends=st + n_alts,
        n_groups=n_groups, n_obs=n,
        actual_choice=z(), cluster_ids=np.arange(n_groups), is_male=is_male,
    )


def _resolve_model_kind(cli_block: dict, spec) -> str:
    """Explicit cli.model wins; else infer RURO iff market-opportunity blocks exist."""
    explicit = str(cli_block.get("model", "") or "").strip().lower()
    if explicit in ("rum", "ruro"):
        return explicit
    if explicit:
        raise ValueError(f"cli.model must be 'rum' or 'ruro'; got {explicit!r}.")
    has_opportunity = bool(getattr(spec, "market_opportunity_shifters", None))
    return "ruro" if has_opportunity else "rum"


def _cmd_estimate(args: argparse.Namespace) -> int:
    cfg = load_yaml(Path(args.config))
    cli_block = (cfg.get("cli") or {}) if isinstance(cfg, dict) else {}
    mode = str(cli_block.get("mode", "")).strip().lower()
    if mode != "synthetic":
        raise NotImplementedError(
            "Only cli.mode: synthetic is supported in v0.1. Real-data estimation "
            "(dataframe/parquet -> PrecomputedData loading) is deferred; set "
            "`cli.mode: synthetic` in the config to run the synthetic fixture path."
        )

    from dclaborsupply.spec.parser import EstimationSpec
    from dclaborsupply.models import RUMModel, RUROModel

    spec = EstimationSpec.from_yaml(Path(args.config))
    model_kind = _resolve_model_kind(cli_block, spec)
    compute_se = bool(cli_block.get("compute_se", True))
    seed = int(cli_block.get("seed", 0))
    n_groups = int(cli_block.get("n_groups", _DEFAULT_N_GROUPS))
    n_alts = int(cli_block.get("n_alts", _DEFAULT_N_ALTS))

    sm = _build_synthetic_singles(is_male=True, seed=seed + 1, n_groups=n_groups, n_alts=n_alts)
    sf = _build_synthetic_singles(is_male=False, seed=seed + 2, n_groups=n_groups, n_alts=n_alts)
    data = (sm, sf, None)

    if model_kind == "rum":
        result = RUMModel.from_spec(spec).fit(data, backend=args.backend, compute_se=compute_se)
    else:
        result = RUROModel.from_spec(spec).recover_synthetic(
            data, seed=seed, band=float(cli_block.get("band", 1.0)))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result.to_json(), encoding="utf-8")
    print(f"[estimate] model={model_kind} mode=synthetic backend={args.backend} "
          f"-> {out}  (neg_ll={result.convergence.get('neg_ll')})")
    return 0


def _cmd_summarize(args: argparse.Namespace) -> int:
    from dclaborsupply.models import Result
    result = Result.from_json(Path(args.result).read_text(encoding="utf-8"))
    print(json.dumps(result.summary(), indent=2, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(prog="dcls")
    subparsers = parser.add_subparsers(dest="command")

    validate = subparsers.add_parser("validate-config", help="Load YAML and print top-level keys.")
    validate.add_argument("path", help="Path to a YAML config file.")
    validate.set_defaults(func=_cmd_validate_config)

    estimate = subparsers.add_parser(
        "estimate", help="Estimate from a synthetic-fixture config (v0.1; real data deferred).")
    estimate.add_argument("--config", required=True, help="Path to a spec YAML with a `cli:` block.")
    estimate.add_argument("--out", required=True, help="Path to write the Result JSON.")
    estimate.add_argument("--backend", default="jax", choices=["jax", "numpy"],
                          help="Engine backend label (objective is JAX-built).")
    estimate.set_defaults(func=_cmd_estimate)

    summarize = subparsers.add_parser("summarize", help="Print summary() of a Result JSON.")
    summarize.add_argument("--result", required=True, help="Path to a Result JSON.")
    summarize.set_defaults(func=_cmd_summarize)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
