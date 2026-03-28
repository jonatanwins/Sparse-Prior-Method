import numpy as np
from numpy.testing import assert_allclose

from ..solvers.block_sparse_prior import (
    _build_precision_matrices,
    quad_form,
    sparse_prior_solution,
)
from ..solvers.representations import (
    complex_matrix_to_augmented_real_vector,
    mixing_tensor_to_frequency_major_matrix,
)


def _example_problem():
    A = np.zeros((2, 3, 2), dtype=complex)
    A[:, :, 0] = np.array([[1, 0, 0], [0, 1, 0]], dtype=complex)
    A[:, :, 1] = A[:, :, 0]
    X_true = np.array([[1.0, 2.0], [0.5, -0.25], [0.0, 0.0]], dtype=complex)
    return A, X_true


def test_true_X_is_locally_optimal_along_null_space_directions():
    A, X_true = _example_problem()

    A_big = mixing_tensor_to_frequency_major_matrix(A)
    _, singular_values, Vt = np.linalg.svd(A_big, full_matrices=True)
    rank = np.sum(singular_values > 1e-10)
    B = Vt[rank:].conj().T
    B_real = np.block([[B.real, -B.imag], [B.imag, B.real]])
    X_true_real = complex_matrix_to_augmented_real_vector(X_true)
    precision_operators = _build_precision_matrices(
        X_true, grouping="frequencies", precision=1.0, eps=1e-3
    )

    def score(z: np.ndarray) -> float:
        x = X_true_real + B_real @ z.reshape(-1, 1)
        return sum(np.exp(-quad_form(x, Lg)) for Lg in precision_operators)

    assert rank < A_big.shape[1]
    assert_allclose(A_big @ B, 0.0, atol=1e-12)

    score_true = score(np.zeros(B_real.shape[1]))
    for z in (
        np.array([1e-2, 0.0, 0.0, 0.0]),
        np.array([0.0, -2e-2, 0.0, 0.0]),
        np.array([0.0, 0.0, 1e-2, 0.0]),
        np.array([0.0, 0.0, 0.0, -2e-2]),
        np.array([1e-2, -2e-2, 5e-3, -1e-2]),
    ):
        assert score_true > score(z)


def test_sparse_prior_keeps_true_X_fixed_when_initialized_at_true_X():
    A, X_true = _example_problem()
    X_opt = sparse_prior_solution(
        X_true, A, grouping="frequencies", precision=1.0, eps=1e-3
    )
    assert_allclose(X_opt, X_true, atol=1e-12)


def test_noisy_ridge_initialization_beats_noisy_pseudoinverse_initialization():
    def ridge_pseudoinverse(A_f: np.ndarray, y_f: np.ndarray, ridge_lambda: float):
        return A_f.conj().T @ np.linalg.solve(
            A_f @ A_f.conj().T + ridge_lambda * np.eye(A_f.shape[0]), y_f
        )

    A = np.zeros((2, 3, 2), dtype=complex)
    A_f = np.array([[1.0, 0.9, 1.0], [0.0, 0.2, 1.0]], dtype=complex)
    A[:, :, 0] = A_f
    A[:, :, 1] = A_f
    X_true = np.array([[1.0, 0.8], [0.0, 0.0], [0.0, 0.0]], dtype=complex)

    Y = np.einsum("msf,sf->mf", A, X_true)
    noise = 5 * np.array([[0.08 + 0.03j, -0.06 + 0.02j], [-0.04 + 0.01j, 0.07 - 0.03j]])
    Y_noisy = Y + noise
    ridge_lambda = np.max(np.abs(noise)) ** 2

    X0_pinv = np.zeros_like(X_true)
    X0_ridge = np.zeros_like(X_true)
    for f in range(A.shape[2]):
        X0_pinv[:, f] = np.linalg.pinv(A[:, :, f]) @ Y_noisy[:, f]
        X0_ridge[:, f] = ridge_pseudoinverse(A[:, :, f], Y_noisy[:, f], ridge_lambda)

    X_opt_pinv = sparse_prior_solution(
        X0_pinv, A, grouping="frequencies", precision=1.0, eps=1e-3
    )
    X_opt_ridge = sparse_prior_solution(
        X0_ridge, A, grouping="frequencies", precision=1.0, eps=1e-3
    )

    assert np.linalg.norm(X_opt_ridge - X_true) < np.linalg.norm(X_opt_pinv - X_true)
