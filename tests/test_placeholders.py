import pytest

from dclaborsupply import RUMModel, RUROModel


def test_rum_model_constructs_and_fit_raises() -> None:
    model = RUMModel()

    with pytest.raises(NotImplementedError, match="v0.1 skeleton"):
        model.fit([])


def test_ruro_model_constructs_and_fit_raises() -> None:
    model = RUROModel()

    with pytest.raises(NotImplementedError, match="v0.1 skeleton"):
        model.fit([])

