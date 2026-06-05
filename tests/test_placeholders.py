import pytest

from dclaborsupply import RUMModel, RUROModel


def test_rum_model_constructs_and_requires_spec() -> None:
    # Wave 3.3: fit() is implemented; a spec-less model raises a clear ValueError
    # (no longer the v0.1 NotImplementedError skeleton).
    model = RUMModel()
    assert model.spec is None
    with pytest.raises(ValueError, match="no spec"):
        model.fit((None, None, None))


def test_ruro_model_constructs_and_requires_spec() -> None:
    model = RUROModel()
    assert model.spec is None
    with pytest.raises(ValueError, match="no spec"):
        model.fit((None, None, None))
