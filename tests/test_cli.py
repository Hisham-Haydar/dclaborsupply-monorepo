"""Wave 3.4 tests for the CLI wired to the model front-end (synthetic v0.1 path).

The estimate path uses jax, and test_cli sorts before test_imports, so the
estimate->summarize round-trip runs in a SUBPROCESS (jax must not leak into the
main pytest process). validate-config / help / unsupported-mode never import jax
(unsupported mode raises before the lazy model import), so they run in-process.
"""
import importlib.util
import json
import subprocess
import sys

import pytest

from dclaborsupply.cli import main

_SPEC = (
    "specification:\n  name: synth_joint\n  wage_spec: \"vw\"\n"
    "utility:\n  functional_form: box_cox\n"
    "  consumption:\n    coefficient: beta_c\n    box_cox_exponent: theta_c\n"
    "  leisure:\n    intercept: beta_l0\n    box_cox_exponent: theta_l\n"
    "wage_opportunity:\n  specification: \"log_normal\"\n  variance:\n    parameter: \"sigma\"\n"
    "initial_values:\n"
    "  beta_l0_sm: 0.8\n  beta_c_sm: 0.8\n  theta_l_sm: 0.4\n  theta_c_sm: 0.4\n"
    "  beta_l0_sf: 0.8\n  beta_c_sf: 0.8\n  theta_l_sf: 0.4\n  theta_c_sf: 0.4\n"
    "  beta_l0_m: 1.0\n  theta_l_m: 0.5\n  beta_l0_f: 1.0\n  theta_l_f: 0.5\n"
    "  beta_c: 1.0\n  theta_c: 0.5\n  sigma: 0.5\n"
    "fixed_params:\n  beta_l0_m: 1.0\n  theta_l_m: 0.5\n  beta_l0_f: 1.0\n  theta_l_f: 0.5\n"
    "  beta_c: 1.0\n  theta_c: 0.5\n  sigma: 0.5\n"
)
_CLI_RUM = "cli:\n  mode: synthetic\n  model: rum\n  compute_se: true\n  seed: 0\n  n_groups: 300\n  n_alts: 5\n"


def test_estimate_then_summarize_roundtrip(tmp_path):
    # find_spec, NOT importorskip: importing jax here would leak into the main
    # pytest process (test_cli precedes test_imports' in-process light check).
    if importlib.util.find_spec("jax") is None:
        pytest.skip("jax not installed")
    cfg = tmp_path / "rum.yaml"
    cfg.write_text(_SPEC + _CLI_RUM, encoding="utf-8")
    out = tmp_path / "res.json"
    # Subprocess: keep jax out of the main pytest process (test_cli precedes test_imports).
    code = (
        "from dclaborsupply.cli import main\n"
        f"assert main(['estimate','--config',{str(cfg)!r},'--out',{str(out)!r},'--backend','jax'])==0\n"
        f"assert main(['summarize','--result',{str(out)!r}])==0\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert out.exists()
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["metadata"]["model"] == "RUM"
    assert len(res["param_names"]) == 8 and res["se_hessian"] is not None
    assert '"model": "RUM"' in r.stdout and '"n_free": 8' in r.stdout  # summarize output


def test_validate_config(tmp_path, capsys):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(_SPEC + _CLI_RUM, encoding="utf-8")
    assert main(["validate-config", str(cfg)]) == 0
    out = capsys.readouterr().out
    assert "cli" in out and "specification" in out


def test_help_and_no_command(capsys):
    # no subcommand -> prints help, returns 0
    assert main([]) == 0
    assert "usage" in capsys.readouterr().out.lower()
    # --help -> argparse exits 0
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_unsupported_real_data_mode_raises(tmp_path):
    # mode != synthetic must raise the documented NotImplementedError BEFORE any
    # heavy import (so this stays jax-free and in-process).
    cfg = tmp_path / "real.yaml"
    cfg.write_text(_SPEC + "cli:\n  mode: real\n  model: rum\n", encoding="utf-8")
    with pytest.raises(NotImplementedError, match="synthetic"):
        main(["estimate", "--config", str(cfg), "--out", str(tmp_path / "x.json"), "--backend", "jax"])


def test_cli_import_is_light():
    code = (
        "import sys, dclaborsupply.cli\n"
        "for m in ('jax', 'scipy', 'gamspy', 'numba'):\n"
        "    assert m not in sys.modules, m + ' imported at cli import!'\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
