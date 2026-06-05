"""Wave 2.4 tests for the GAMSPy vectorized solver (self-contained).

The gate is import-cleanliness, not numerical reproduction: gamspy/gams must NOT
load at import, and calling an entry without gamspy must raise a documented
ImportError. GAMSPy-present LL parity vs scipy/JAX is deferred to app validation.
Subprocesses give a clean sys.modules (and let us simulate gamspy ABSENT even
when it happens to be installed).
"""
import os
import subprocess
import sys


def _run(code: str):
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


def test_gamspy_import_is_clean():
    # (a) importing the module must NOT pull gamspy/gams into sys.modules, even if
    # gamspy is installed (lazy binding).
    code = (
        "import sys\n"
        "import dclaborsupply.solvers\n"
        "import dclaborsupply.solvers.gamspy_vectorized as gv\n"
        "assert 'gamspy' not in sys.modules, 'gamspy imported at module load!'\n"
        "assert 'gams' not in sys.modules, 'gams imported at module load!'\n"
        "assert gv.Container is None, 'gamspy symbol bound at import!'\n"
    )
    r = _run(code)
    assert r.returncode == 0, r.stderr


def test_gamspy_entries_raise_documented_importerror_without_gamspy():
    # (b) with gamspy ABSENT (simulated by blocking the import), every estimation
    # entry raises ImportError pointing at the optional extra (not Attribute/NameError).
    code = (
        "import sys\n"
        "sys.modules['gamspy'] = None\n"          # force ImportError on `import gamspy`
        "sys.modules['gamspy.math'] = None\n"
        "import dclaborsupply.solvers.gamspy_vectorized as gv\n"
        "calls = [\n"
        "    (gv.estimate_singles_vectorized_gamspy, (object(), object(), None)),\n"
        "    (gv.estimate_couples_vectorized_gamspy, (object(), object(), None)),\n"
        "    (gv.estimate_joint_vectorized_gamspy,   (object(), object(), object(), object(), None)),\n"
        "]\n"
        "for fn, args in calls:\n"
        "    try:\n"
        "        fn(*args)\n"
        "        raise SystemExit('no error raised')\n"
        "    except ImportError as e:\n"
        "        assert 'dclaborsupply[gamspy]' in str(e), str(e)\n"
        "    except ImportError:\n"
        "        raise\n"
        "    except Exception as e:\n"
        "        raise SystemExit(f'wrong exception {type(e).__name__}: {e}')\n"
    )
    r = _run(code)
    assert r.returncode == 0, (r.stdout + r.stderr)


def test_set_gamspy_workdir_injected_no_chdir(tmp_path):
    # R4: with an explicit workdir, set GAMSPY_WORKING_DIR + create it, but never chdir.
    from dclaborsupply.solvers.gamspy_vectorized import _set_gamspy_workdir

    cwd_before = os.getcwd()
    target = tmp_path / "gams_work"
    returned = _set_gamspy_workdir(str(target))

    assert returned == str(target)
    assert target.is_dir()
    assert os.environ["GAMSPY_WORKING_DIR"] == str(target)
    assert os.getcwd() == cwd_before  # core does NOT chdir
