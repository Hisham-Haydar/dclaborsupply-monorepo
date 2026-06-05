"""Command-line interface for the v0.1 skeleton."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from dclaborsupply.config.loader import load_yaml


SKELETON_MESSAGE = "NotImplementedError: v0.1 skeleton"


def _cmd_validate_config(args: argparse.Namespace) -> int:
    config = load_yaml(Path(args.path))
    keys = ", ".join(sorted(config.keys())) if config else "(no keys)"
    print(f"Loaded config keys: {keys}")
    return 0


def _cmd_estimate(args: argparse.Namespace) -> int:
    print(SKELETON_MESSAGE)
    return 0


def _cmd_summarize(args: argparse.Namespace) -> int:
    print(SKELETON_MESSAGE)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(prog="dcls")
    subparsers = parser.add_subparsers(dest="command")

    validate = subparsers.add_parser("validate-config", help="Load YAML and print top-level keys.")
    validate.add_argument("path", help="Path to a YAML config file.")
    validate.set_defaults(func=_cmd_validate_config)

    estimate = subparsers.add_parser("estimate", help="Placeholder estimation command.")
    estimate.set_defaults(func=_cmd_estimate)

    summarize = subparsers.add_parser("summarize", help="Placeholder summary command.")
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

