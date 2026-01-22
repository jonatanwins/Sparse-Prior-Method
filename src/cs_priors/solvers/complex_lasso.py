import numpy as np
from sklearn.linear_model import Lasso
from cs_priors.solvers.vectorized_sparse_prior import (
    to_real_augmented,
    from_real_augmented,
)

def complex_lasso(A, Y, alpha=0.1):
    """
    Solve the complex LASSO problem by converting to a real-valued problem.
    Minimize ||Y - A X||_2^2 + alpha * ||X||_1
    where A is complex, Y is complex, and X is complex.---_
    """
    # If A is full rank, we do not need LASSO
    X0 = np.linalg.pinv(A) @ Y  # initial guess for X
    mics, sources = A.shape
    if mics >= sources:
        return X0

    X0_real = to_real_augmented(X0)
    A_real = to_real_augmented(A)
    Y_real = to_real_augmented(Y)

    lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000)
    lasso.coef_ = X0_real.flatten()
    lasso.fit(A_real, Y_real)
    X_real = lasso.coef_

    X_complex = from_real_augmented(X_real)
    return X_complex