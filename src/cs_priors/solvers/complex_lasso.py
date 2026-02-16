import numpy as np
from sklearn.linear_model import Lasso
from typeguard import typechecked
from cs_priors.solvers.vectorized_sparse_prior import (
    to_real_augmented,
    from_real_augmented,
)


@typechecked
def complex_lasso(
    Y: np.ndarray, A: np.ndarray, alpha: float = 0.1, max_iter: int = 10000
) -> np.ndarray:
    """
    Solve the complex LASSO problem by converting to a real-valued problem.
    Minimize ||Y - A X||_2^2 + alpha * ||X||_1
    where A is complex, Y is complex, and X is complex.---_
    """
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
