"""Standard errors: cluster-robust sandwich + T1-T5 checks (lifted Wave 2.1)."""

from dclaborsupply.se.cluster_robust import (
    assemble_meat_matrix,
    compute_cluster_robust_se,
    run_t1_sign_check,
    run_t2_symmetry_check,
    run_t3_cluster_count_check,
    run_t4_se_positivity_check,
    run_t5_vs_hessian_check,
)

__all__ = [
    "assemble_meat_matrix",
    "compute_cluster_robust_se",
    "run_t1_sign_check",
    "run_t2_symmetry_check",
    "run_t3_cluster_count_check",
    "run_t4_se_positivity_check",
    "run_t5_vs_hessian_check",
]
