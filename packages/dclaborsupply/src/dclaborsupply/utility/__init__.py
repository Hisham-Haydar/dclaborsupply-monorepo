"""Utility math for the core (migration matrix Wave 3.1 confirm/close).

Box-Cox primitives live in :mod:`dclaborsupply.utility.boxcox`:
``box_cox_transform``, ``box_cox_derivative_x``, ``box_cox_derivative_theta``.

Import them from the submodule, e.g.::

    from dclaborsupply.utility.boxcox import box_cox_transform

This package deliberately does NOT re-export them eagerly: ``boxcox`` attempts an
optional Numba import at module load, so a bare ``import dclaborsupply.utility``
stays light (no jax/gamspy/java/numba) and the acceleration loads only when the
``boxcox`` submodule is imported.
"""
