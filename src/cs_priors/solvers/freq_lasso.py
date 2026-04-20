import numpy as np
from sklearn.linear_model import Lasso
from typeguard import typechecked
from cs_priors.solvers.representations import (
    augmented_real_vector_to_complex_matrix,
    complex_matrix_to_augmented_real_matrix,
    complex_matrix_to_augmented_real_vector,
    mixing_tensor_to_frequency_major_matrix,
    ensure_frequency_system_shapes,
)


@typechecked
def frequency_lasso_solve(
    Y: np.ndarray, A: np.ndarray, alpha: float = 1e-8, max_iter: int = 10000
) -> np.ndarray:
    """
    Solve the complex LASSO problem by converting to a real-valued problem.
    Minimize ||Y - A X||_2^2 + alpha * ||X||_1
    where A is complex, Y is complex, and X is complex.

    Args:
        Y: (M,), (M, 1), or (M, F) complex array of measurements
        A: (M,S) or (M, S, F) complex mixing matrix
        alpha: regularization strength for LASSO
        max_iter: maximum number of iterations for the LASSO solver

    Returns:
        X_hat: (S, F) complex


    """
    A, Y, _, _ = ensure_frequency_system_shapes(A, Y)
    _, num_sources, num_freqs = A.shape

    A_big = mixing_tensor_to_frequency_major_matrix(A)  # (MF, SF)
    A_real = complex_matrix_to_augmented_real_matrix(A_big)  # (2MF, 2SF)
    Y_real = complex_matrix_to_augmented_real_vector(Y).ravel()  # (2MF,)

    model = Lasso(alpha=alpha, fit_intercept=False, max_iter=max_iter)
    model.fit(A_real, Y_real)

    X_real = model.coef_.reshape(-1, 1)  # (2SF, 1)
    X_hat = augmented_real_vector_to_complex_matrix(
        X_real, num_sources, num_freqs
    )  # (S, F)

    return X_hat

    # ///////////////////////////////////////////////////////////////////////////
    # assert that dimensions match
    assert (
        A.shape[0] == Y.shape[0]
    ), f"Number of rows in A (shape {A.shape}) must match number of rows in Y (shape {Y.shape})"
    assert (
        A.ndim == 2
    ), f"A must be a 2D array, got shape {A.shape}, did you pass Y and A in the correct order?"

    # If A is full rank, we do not need LASSO
    X0 = np.linalg.pinv(A) @ Y  # initial guess for X
    mics, sources = A.shape
    if mics >= sources:
        return X0

    X0_real = to_real_augmented(X0)
    A_real = to_real_augmented(A)
    Y_real = to_real_augmented(Y)

    lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=max_iter)
    lasso.coef_ = X0_real.flatten()
    lasso.fit(A_real, Y_real)
    X_real = lasso.coef_

    X_complex = from_real_augmented(X_real)
    return X_complex


if __name__ == "__main__":
    np.random.seed(0)

    # Multi-frequency underdetermined system
    A = np.zeros((2, 3, 2), dtype=complex)
    A[:, :, 0] = np.array([[1.0, 0.2, 0.0], [0.0, 1.0, 0.3]], dtype=complex)
    A[:, :, 1] = np.array([[1.0, -0.1, 0.0], [0.0, 1.0, 0.4]], dtype=complex)

    X_true = np.array(
        [
            [1.0 + 0.5j, 0.8 - 0.2j],
            [0.0 + 0.0j, 0.0 + 0.0j],
            [0.2 - 0.1j, 0.0 + 0.0j],
        ]
    )
    Y = np.einsum("msf,sf->mf", A, X_true)

    X_hat = frequency_lasso_solve(Y, A)

    print("multi-frequency shapes:", A.shape, Y.shape, X_hat.shape)
    print(
        "multi-frequency residual:",
        np.linalg.norm(np.einsum("msf,sf->mf", A, X_hat) - Y),
    )

    assert X_hat.shape == X_true.shape
    assert np.linalg.norm(np.einsum("msf,sf->mf", A, X_hat) - Y) < 1e-6

    # Single-frequency compatibility test
    A1 = A[:, :, 0]
    Y1 = Y[:, 0]

    X_hat1 = frequency_lasso_solve(Y1, A1)

    print("single-frequency shapes:", A1.shape, Y1.shape, X_hat1.shape)
    print("single-frequency residual:", np.linalg.norm(A1 @ X_hat1[:, 0] - Y1))

    assert X_hat1.shape == (A1.shape[1], 1)
    assert np.linalg.norm(A1 @ X_hat1[:, 0] - Y1) < 1e-6

    print("augmented_real_lasso smoke test passed")
