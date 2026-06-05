import sys


def test_core_import_is_light() -> None:
    import dclaborsupply

    assert dclaborsupply.__version__ == "0.1.0"
    assert "jax" not in sys.modules
    assert "gamspy" not in sys.modules
    assert "euromod" not in sys.modules


def test_app_import() -> None:
    import dclaborsupply_app

    assert dclaborsupply_app.__version__ == "0.1.0"

