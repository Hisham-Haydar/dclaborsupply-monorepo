# dclaborsupply Monorepo

This is an early v0.1 skeleton for a discrete-choice labour-supply Python
monorepo. It contains package structure, import boundaries, CLI wiring, tests,
and placeholders only.

No likelihood, utility, opportunity-density, EUROMOD, or welfare implementation
logic is included in this scaffold. Public methods that would perform scientific
work raise `NotImplementedError("v0.1 skeleton")`.

## Packages

- `packages/dclaborsupply`: core estimation package. It is microsimulator-free
  and must import without Java, JAX, or GAMSPy installed.
- `packages/dclaborsupply_app`: application package for EUROMOD, France-specific
  preparation, welfare implementations, and reporting placeholders. It depends
  on `dclaborsupply`.

## Editable Install

```bash
pip install -e packages/dclaborsupply
pip install -e packages/dclaborsupply_app
```

## Optional Backends

The base core install depends only on `numpy`, `pandas`, and `pyyaml`.

JAX is the intended default production backend when estimator logic is added,
but it is optional in this skeleton:

```bash
pip install "dclaborsupply[jax]"
```

GAMSPy is an optional solver extra. It is not required for base imports, tests,
or notebook use. Install it only when the GAMSPy solver path is needed:

```bash
pip install "dclaborsupply[gamspy]"
```

## CLI

```bash
dcls --help
dcls validate-config path/to/config.yaml
dcls estimate
dcls summarize
```

## Tests

```bash
pytest
```

