from dclaborsupply import EstimationSpec
from dclaborsupply.config.loader import load_yaml


def test_from_yaml_loads_raw_dict(tmp_path) -> None:
    config = tmp_path / "tiny.yaml"
    config.write_text("specification:\n  name: tiny\n", encoding="utf-8")

    spec = EstimationSpec.from_yaml(config)

    assert spec.raw == {"specification": {"name": "tiny"}}
    assert spec.source == config


def test_load_yaml(tmp_path) -> None:
    config = tmp_path / "tiny.yaml"
    config.write_text("alpha: 1\nbeta: 2\n", encoding="utf-8")

    assert load_yaml(config) == {"alpha": 1, "beta": 2}

