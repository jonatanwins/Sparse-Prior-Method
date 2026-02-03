from cs_priors.solvers.real_augmented import (
    to_real_augmented,
    from_real_augmented,
)
from scipy import optimize
import numpy as np


def optimize_objective(
    X0: np.ndarray, A: np.ndarray, Y: np.ndarray, alpha: float, max_iter: int = 1000
):
    # We will need to convert X0 and A to real augmented form for the optimization, but otherwise keep them as complex

    def negative_objective(X_real: np.ndarray) -> float:
        X_complex = from_real_augmented(X_real)
        L2 = 0.5 * np.linalg.norm(Y.reshape(-1, 1) - A @ X_complex.reshape(-1, 1)) ** 2
        L1 = alpha * np.sum(np.abs(X_complex))
        return L2 + L1

    X0_real = to_real_augmented(X0)
    result = optimize.minimize(
        negative_objective,
        X0_real.flatten(),
        method='nelder-mead',
        options={
            'maxiter': max_iter,
            'disp': True,
        },  # Increase iterations and show progress
    )

    return result


def c_lasso(Y: np.ndarray, A: np.ndarray, alpha: float, max_iter=1000):
    """
    Solve the complex lasso problem:
        min_X 0.5 * ||Y - A X||_2^2 + alpha * ||X||_1

    Args:
        Y: response vector  (P x 1)
        A: Mixing matrix (P x N)
        alpha: Regularization parameter
        max_iter: Maximum number of iterations for optimization

    Returns:
        X_opt: Optimized coefficient matrix (N x 1)
    """
    X0 = np.linalg.pinv(A) @ Y  # Initial solution
    result = optimize_objective(X0, A, Y, alpha, max_iter=max_iter)
    X_opt_real = result.x
    X_opt = from_real_augmented(X_opt_real)
    return X_opt


if __name__ == "__main__":
    # Example usage
    P, N = 1, 1
    Y = np.random.randn(P, 1) + 1j * np.random.randn(P, 1)
    A = np.random.randn(P, N) + 1j * np.random.randn(P, N)
    alpha = 0.1

    X0 = np.linalg.pinv(A) @ Y
    print(X0)
    print(optimize_objective(X0, A, Y, alpha))

    # Call the optimization function (to be implemented)
    # X_opt_real = optimize_real_valued_objective(X0_real, A_real, Y, alpha)
