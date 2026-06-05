"""
==============================================================================
Cluster-Robust Sandwich Standard Errors — RURO MNL
==============================================================================
Implements the clustered sandwich covariance estimator:

    V_cluster = H^{-1} B H^{-1}

where
    H  = Hessian of the negative log-likelihood (bread; same as Hessian-only SE)
    B  = sum_j  s_j s_j'                        (meat)
    s_j = sum_{g in cluster j}  scores_all[g]   (cluster score)
    scores_all[g] = per-choice-set gradient of positive LL (from return_scores=True)

Cluster key: cluster_id = idorighh (original household id across survey years).
9,657 unique clusters in the P3a pooled dataset.

GA15 note:
  Singles consumption derives from ils_dispy_real (non-null for singles only).
  Couples consumption derives from ils_dispy_male + ils_dispy_female.
  The score extractor (compute_scores_joint in estimation_engine.py) handles
  both paths independently through their respective PrecomputedData structs.

Design audit reference: docs/estimation/RURO_cluster_robust_SE_design_audit_v1.md

Author: Enhanced RURO Pipeline
Created: 2026-05-21

Lifted into the dclaborsupply core package (migration matrix Wave 2.1) from
MNL/scripts/enhanced/cluster_robust_se.py — verbatim copy (the module is pure
NumPy with no old-repo imports; nothing to adapt). The CLI/workflow
(run_cluster_robust_se.py) was NOT lifted.

This module performs ONLY the meat assembly + sandwich + T1–T5 checks; it takes
pre-computed `scores_all`/`cluster_ids_all`/`hessian` as inputs and contains NO
chunking. Memory-safe (chunked `jax.jacrev`) per-choice-set score extraction
lives in the estimation workflow (e.g. step4_realdata_baseline `_chunked_scores`
/ `_slice_data_groups`); in core, scores come from the lifted engines
(`engine_jax` per-group `jacrev`, or `engine_numpy.compute_scores_joint`). Do
NOT introduce dense full-data jacrev here.
==============================================================================
"""

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def assemble_meat_matrix(
    scores_all: np.ndarray,
    cluster_ids_all: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """
    Assemble the meat matrix B = sum_j s_j s_j' from per-choice-set scores.

    Parameters
    ----------
    scores_all : np.ndarray, shape (n_groups_total, n_params)
        Per-choice-set score vectors for the POSITIVE log-likelihood.
        Obtained from compute_scores_joint() in estimation_engine.py.
    cluster_ids_all : np.ndarray, shape (n_groups_total,)
        Cluster id (idorighh) for each choice-set row, aligned to scores_all.

    Returns
    -------
    B : np.ndarray, shape (n_params, n_params)
        Meat matrix. Symmetric by construction.
    n_clusters : int
        Number of unique clusters contributing to B.
    """
    n_params = scores_all.shape[1]
    B = np.zeros((n_params, n_params))

    unique_clusters = np.unique(cluster_ids_all)
    n_clusters = len(unique_clusters)

    for cluster_j in unique_clusters:
        mask = cluster_ids_all == cluster_j
        s_j = scores_all[mask].sum(axis=0)  # (n_params,)
        B += np.outer(s_j, s_j)

    return B, n_clusters


def compute_cluster_robust_se(
    hessian: np.ndarray,
    scores_all: np.ndarray,
    cluster_ids_all: np.ndarray,
    free_mask: Optional[np.ndarray] = None,
    apply_finite_sample_correction: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute cluster-robust sandwich standard errors.

    Sandwich formula:
        V_cluster = H^{-1} B H^{-1}

    where H is the Hessian of the negative log-likelihood (bread) and
    B = sum_j s_j s_j' is the meat matrix.

    Parameters
    ----------
    hessian : np.ndarray, shape (n_params, n_params)
        Hessian of the negative log-likelihood. Same matrix used in the
        Hessian-only SE computation. Must be positive semi-definite.
    scores_all : np.ndarray, shape (n_groups_total, n_params)
        Per-choice-set score vectors for the POSITIVE log-likelihood,
        from compute_scores_joint() in estimation_engine.py.
    cluster_ids_all : np.ndarray, shape (n_groups_total,)
        Cluster id (idorighh) aligned to scores_all rows.
    free_mask : np.ndarray of bool, shape (n_params,), optional
        True for estimated (free) parameters. Parameters at bounds receive
        se_robust = 0. If None, all parameters are treated as free.
    apply_finite_sample_correction : bool, default=False
        If True, multiply B by J/(J-1) where J = number of clusters.
        For J = 9,657 this correction is negligible (<0.01%).

    Returns
    -------
    se_robust : np.ndarray, shape (n_params,)
        Cluster-robust standard errors. Zero for fixed/bounded parameters.
    varcov_robust : np.ndarray, shape (n_params, n_params)
        Full sandwich covariance matrix. Rows/cols for fixed parameters = 0.
    B : np.ndarray, shape (n_params, n_params)
        Meat matrix assembled from cluster scores.
    """
    n_params = hessian.shape[0]

    if free_mask is None:
        free_mask = np.ones(n_params, dtype=bool)

    # Assemble meat
    B, n_clusters = assemble_meat_matrix(scores_all, cluster_ids_all)
    logger.info(f"Meat matrix assembled from {n_clusters} unique clusters")

    if apply_finite_sample_correction and n_clusters > 1:
        correction = n_clusters / (n_clusters - 1)
        B = B * correction
        logger.info(f"Finite-sample correction applied: J/(J-1) = {correction:.6f}")

    # Restrict to free parameters
    H_free = hessian[np.ix_(free_mask, free_mask)]
    B_free = B[np.ix_(free_mask, free_mask)]

    # Invert bread (pseudoinverse for robustness)
    try:
        H_free_inv = np.linalg.pinv(H_free, rcond=1e-10)
    except np.linalg.LinAlgError as exc:
        logger.error(f"Hessian pseudoinversion failed: {exc}")
        se_robust = np.full(n_params, np.nan)
        varcov_robust = np.full((n_params, n_params), np.nan)
        return se_robust, varcov_robust, B

    # Sandwich
    VarCov_free = H_free_inv @ B_free @ H_free_inv

    # Symmetrize to remove floating-point asymmetry
    VarCov_free = 0.5 * (VarCov_free + VarCov_free.T)

    # Assemble full-dimension matrix
    varcov_robust = np.zeros((n_params, n_params))
    free_indices = np.where(free_mask)[0]
    for i, fi in enumerate(free_indices):
        for j, fj in enumerate(free_indices):
            varcov_robust[fi, fj] = VarCov_free[i, j]

    se_robust = np.sqrt(np.abs(np.diag(varcov_robust)))

    # Confirm zeros for fixed/bounded parameters
    se_robust[~free_mask] = 0.0

    return se_robust, varcov_robust, B


def run_t1_sign_check(
    scores_all: np.ndarray,
    neg_grad: np.ndarray,
    atol: float = 1e-6,
) -> dict:
    """
    T1: Verify scores_all.sum(axis=0) == -neg_grad (gradient of positive LL).

    Parameters
    ----------
    scores_all : np.ndarray, shape (n_groups, n_params)
    neg_grad : np.ndarray, shape (n_params,)
        Gradient of NEGATIVE log-likelihood from compute_gradient_joint().
    atol : float
        Absolute tolerance for the check.

    Returns
    -------
    dict with keys: passed (bool), max_abs_diff (float), mean_abs_diff (float).
    """
    scores_sum = scores_all.sum(axis=0)
    expected = -neg_grad
    diff = scores_sum - expected
    max_diff = float(np.max(np.abs(diff)))
    mean_diff = float(np.mean(np.abs(diff)))
    passed = max_diff <= atol
    return {
        "passed": passed,
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
        "atol": atol,
    }


def run_t2_symmetry_check(B: np.ndarray, atol: float = 1e-10) -> dict:
    """T2: Verify meat matrix B is symmetric."""
    diff = float(np.max(np.abs(B - B.T)))
    return {"passed": diff <= atol, "max_abs_diff": diff, "atol": atol}


def run_t3_cluster_count_check(cluster_ids_all: np.ndarray, expected: int = 9657) -> dict:
    """T3: Verify the number of unique cluster ids."""
    n = len(np.unique(cluster_ids_all))
    return {"passed": n == expected, "n_unique_clusters": n, "expected": expected}


def run_t4_se_positivity_check(se_robust: np.ndarray, free_mask: np.ndarray) -> dict:
    """T4: All cluster-robust SEs for free parameters are strictly positive."""
    se_free = se_robust[free_mask]
    n_nonpositive = int(np.sum(se_free <= 0))
    return {"passed": n_nonpositive == 0, "n_nonpositive": n_nonpositive, "n_free": int(free_mask.sum())}


def run_t5_vs_hessian_check(
    se_robust: np.ndarray,
    se_hessian: np.ndarray,
    param_names: list,
    free_mask: np.ndarray,
) -> dict:
    """
    T5: Log parameters where robust SE < Hessian SE (informational; does not raise).
    Clustering normally inflates SEs, so robust < hessian suggests low within-cluster
    correlation for that parameter.
    """
    below = []
    for i, name in enumerate(param_names):
        if not free_mask[i]:
            continue
        if se_robust[i] < se_hessian[i]:
            below.append({
                "param": name,
                "se_robust": float(se_robust[i]),
                "se_hessian": float(se_hessian[i]),
                "ratio": float(se_robust[i] / se_hessian[i]) if se_hessian[i] > 0 else None,
            })
    return {"n_below": len(below), "below": below}