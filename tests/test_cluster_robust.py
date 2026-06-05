"""Wave 2.1 tests for cluster-robust SE (self-contained; no jax, no MNL data).

The certified-data gate (reproduce se_clustered for all 47 params on the certified
parquets) runs as a validation script with MNL read-only helpers, not as a
committed test. Here: meat assembly, the sandwich, and T1-T5 on tiny synthetic
inputs with hand-checkable values.
"""
import numpy as np

from dclaborsupply.se import (
    assemble_meat_matrix,
    compute_cluster_robust_se,
    run_t1_sign_check,
    run_t2_symmetry_check,
    run_t3_cluster_count_check,
    run_t4_se_positivity_check,
    run_t5_vs_hessian_check,
)

# 3 choice-set rows, 2 clusters: cluster 10 = rows {0,1}, cluster 20 = row {2}.
SCORES = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
CIDS = np.array([10, 10, 20])
# cluster scores: s_10 = [1,1], s_20 = [1,1]  =>  B = 2*[[1,1],[1,1]]
B_EXPECTED = np.array([[2.0, 2.0], [2.0, 2.0]])


def test_assemble_meat_matrix():
    B, n_clusters = assemble_meat_matrix(SCORES, CIDS)
    assert n_clusters == 2
    assert np.allclose(B, B_EXPECTED)
    assert np.allclose(B, B.T)  # symmetric by construction


def test_sandwich_identity_bread():
    # H = I  =>  V = B, se = sqrt(diag(B))
    H = np.eye(2)
    se, V, B = compute_cluster_robust_se(H, SCORES, CIDS)
    assert np.allclose(B, B_EXPECTED)
    assert np.allclose(V, B_EXPECTED)
    assert np.allclose(se, np.sqrt([2.0, 2.0]))


def test_sandwich_scaled_bread():
    # H = 2I => Hinv = 0.5I => V = 0.25 B
    H = 2.0 * np.eye(2)
    se, V, _ = compute_cluster_robust_se(H, SCORES, CIDS)
    assert np.allclose(V, 0.25 * B_EXPECTED)
    assert np.allclose(se, np.sqrt(0.25 * np.diag(B_EXPECTED)))


def test_free_mask_zeros_fixed_params():
    H = np.eye(2)
    se, V, _ = compute_cluster_robust_se(H, SCORES, CIDS, free_mask=np.array([True, False]))
    assert se[1] == 0.0
    assert np.allclose(se[0], np.sqrt(2.0))
    assert np.all(V[1, :] == 0.0) and np.all(V[:, 1] == 0.0)


def test_finite_sample_correction():
    H = np.eye(2)
    se0, _, _ = compute_cluster_robust_se(H, SCORES, CIDS, apply_finite_sample_correction=False)
    se1, _, _ = compute_cluster_robust_se(H, SCORES, CIDS, apply_finite_sample_correction=True)
    # J/(J-1) = 2/1 = 2 on B => SE scales by sqrt(2)
    assert np.allclose(se1, se0 * np.sqrt(2.0))


def test_t1_sign_check():
    # scores.sum(0) = [2, 2]; passes when neg_grad = -[2,2]
    assert run_t1_sign_check(SCORES, -np.array([2.0, 2.0]))["passed"]
    bad = run_t1_sign_check(SCORES, np.array([0.0, 0.0]))
    assert not bad["passed"] and bad["max_abs_diff"] == 2.0


def test_t2_symmetry_check():
    assert run_t2_symmetry_check(B_EXPECTED)["passed"]
    asym = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert not run_t2_symmetry_check(asym)["passed"]


def test_t3_cluster_count_check():
    assert run_t3_cluster_count_check(CIDS, expected=2)["passed"]
    r = run_t3_cluster_count_check(CIDS, expected=9657)
    assert not r["passed"] and r["n_unique_clusters"] == 2


def test_t4_se_positivity_check():
    fm = np.array([True, True])
    assert run_t4_se_positivity_check(np.array([1.0, 2.0]), fm)["passed"]
    bad = run_t4_se_positivity_check(np.array([1.0, 0.0]), fm)
    assert not bad["passed"] and bad["n_nonpositive"] == 1
    # a non-positive SE on a FIXED param is ignored
    assert run_t4_se_positivity_check(np.array([1.0, 0.0]), np.array([True, False]))["passed"]


def test_t5_vs_hessian_check():
    r = run_t5_vs_hessian_check(
        se_robust=np.array([1.0, 2.0]),
        se_hessian=np.array([1.5, 1.0]),
        param_names=["a", "b"],
        free_mask=np.array([True, True]),
    )
    assert r["n_below"] == 1
    assert r["below"][0]["param"] == "a"
