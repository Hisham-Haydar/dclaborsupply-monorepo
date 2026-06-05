from dclaborsupply import EstimationSpec
from dclaborsupply.config.loader import load_yaml


# Wave 1.1: from_yaml now runs the FULL spec parser (lifted from the research
# repo), replacing the v0.1 stub that only stored the raw YAML dict. This test
# was updated from asserting spec.raw/spec.source to asserting the real
# full-parse contract (parsed param list) on a self-contained minimal spec.
def test_from_yaml_full_parse(tmp_path) -> None:
    config = tmp_path / "tiny.yaml"
    config.write_text(
        "specification:\n"
        "  name: tiny\n"
        "  wage_spec: fw\n"
        "utility:\n"
        "  functional_form: box_cox\n"
        "  consumption:\n"
        "    coefficient: beta_c\n"
        "  leisure:\n"
        "    intercept: beta_l0\n"
        "initial_values:\n"
        "  beta_l0_sm: 0.0\n"
        "  beta_c_sm: 0.0\n"
        "  beta_l0_sf: 0.0\n"
        "  beta_c_sf: 0.0\n"
        "  beta_l0_m: 0.0\n"
        "  beta_l0_f: 0.0\n"
        "  beta_c: 0.0\n",
        encoding="utf-8",
    )

    spec = EstimationSpec.from_yaml(config)

    assert isinstance(spec, EstimationSpec)
    assert spec.name == "tiny"
    assert spec.wage_spec == "fw"
    assert set(spec.all_param_names) == {
        "beta_l0_sm", "beta_c_sm",
        "beta_l0_sf", "beta_c_sf",
        "beta_l0_m", "beta_l0_f",
        "beta_c",
    }


def test_load_yaml(tmp_path) -> None:
    config = tmp_path / "tiny.yaml"
    config.write_text("alpha: 1\nbeta: 2\n", encoding="utf-8")

    assert load_yaml(config) == {"alpha": 1, "beta": 2}

