import subprocess
import sys
import textwrap


def test_core_import_is_light() -> None:
    # Run in a CLEAN subprocess: asserting "import dclaborsupply is light" must be
    # independent of whatever earlier in-process tests imported (e.g. jax for the engine).
    code = textwrap.dedent(
        """
        import sys
        import dclaborsupply
        assert dclaborsupply.__version__ == "0.1.0", dclaborsupply.__version__
        for mod in ("jax", "gamspy", "euromod"):
            assert mod not in sys.modules, mod
        print("OK")
        """
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0 and "OK" in out.stdout, f"stdout={out.stdout}\nstderr={out.stderr}"


def test_app_import() -> None:
    import dclaborsupply_app

    assert dclaborsupply_app.__version__ == "0.1.0"

