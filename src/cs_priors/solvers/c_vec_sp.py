import numpy as np
import scipy.optimize as optimize
from cs_priors.solvers.real_augmented import from_real_augmented, to_real_augmented


def optimize_real_valued_objective(
    X0,
    B,
    variance: float = 1.0,
    spread: float = 0.005,
    callback=None,
    z_start=None,
):
    delta = variance - spread  # Coefficient difference
    X0_real = to_real_augmented(X0)
    B_real = to_real_augmented(B)

    def negative_objective(z: np.ndarray) -> float:
        x = (X0.reshape(-1, 1) + B @ z.reshape(-1, 1)).flatten()
        x = to_real_augmented(x.reshape(-1, 1)).flatten()

        norm_squared = np.sum(x**2)
        # sum_i exp(-x^T D_i x) = exp(-spread * ||x||^2) * sum_i exp(-delta * x[i]^2)
        global_factor = np.exp(-spread * norm_squared)
        individual_sum = np.sum(np.exp(-delta * x**2))
        objective_value = np.abs(global_factor * individual_sum)

        return -objective_value

    def first_derivative_objective(z: np.ndarray) -> np.ndarray:
        # (-1) e^{-(x_0 + Bz)^T D (x_0+Bz)}(B^T(D + D^T) (x_0 + B z))
        # where D_i is symmetric
        # It can be computed more efficiently as follows:
        # B^T (-2e^{-(-spread * ||x||^2} * [ spread * x * sum_i e^{-delta * x[i]^2} + delta * elementwise(x, e^{-delta * x^2}) ]
        # where delta = variance - spread
        x = X0.reshape(-1, 1) + (B @ z.reshape(-1, 1))
        x = to_real_augmented(x.reshape(-1, 1))
        x_flat = x.flatten()

        # Compute quadratic forms efficiently
        norm_squared = np.sum(x_flat**2)
        global_factor = np.exp(-spread * norm_squared)
        exp_terms = np.exp(-delta * x_flat**2)  # BUG: was x**2
        individual_sum = np.sum(exp_terms)

        dx = -2 * global_factor * (spread * x * individual_sum + delta * x * exp_terms)
        grad = B_real.T @ dx

        return -grad.reshape(-1)

    if z_start is None:
        z_start = np.zeros(B.shape[1], dtype=float)

    result = optimize.minimize(
        negative_objective,
        z_start,
        jac=first_derivative_objective,
        callback=callback,
        method="L-BFGS-B",
    )

    return result


def sparse_prior_solution(X0: np.ndarray, A: np.ndarray):

    U, S, Vh = np.linalg.svd(A)
    rank = np.sum(S > 1e-10)

    # Check if there is a null space
    if rank == A.shape[1]:
        return X0, None

    # Null space basis vectors are the last (num_sources - rank) rows of Vh
    # TODO: this might need to be conjugated
    B = Vh[rank:].conj().T  # Shape (num_sources, dim_null_space)

    # Optimize the objective function in the null space
    result = optimize_real_valued_objective(X0, B, variance=1.0, spread=0.005)

    # Reconstruct the solution
    z_opt = result.x
    X_opt = X0 + B @ z_opt.reshape(-1, 1)

    return X_opt, B


if __name__ == "__main__":
    # Example usage
    P, N = 5, 10
    X = np.random.randn(N, 1) + 1j * np.random.randn(N, 1)
    A = np.random.randn(P, N) + 1j * np.random.randn(P, N)
    Y = A @ X

    X0 = np.linalg.pinv(A) @ Y
    X_sp, B = sparse_prior_solution(X0, A)

    print("Sparse prior solution X_sp-X:")
    print(np.linalg.norm(X_sp - X))
