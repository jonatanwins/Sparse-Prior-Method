import numpy as np
import scipy.optimize as optimize


# Convert to complex vectors and matrices to augmented real form for optimization
def to_real_augmented(x_complex: np.ndarray) -> np.ndarray:
    # X is a row vector, either (N,) or (1,N), or a column vector (N, 1)
    if x_complex.ndim == 1 or (x_complex.ndim == 2 and x_complex.shape[1] == 1):
        x_real = np.array([x_complex.real, x_complex.imag]).reshape(-1, 1)
    else:
        x_real = np.block(
            [[x_complex.real, -x_complex.imag], [x_complex.imag, x_complex.real]]
        )
    return x_real


def from_real_augmented(x_real: np.ndarray) -> np.ndarray:
    n = x_real.shape[0] // 2
    return (x_real[:n] + 1j * x_real[n:]).reshape(-1, 1)


def optimize_real_valued_objective(
    X0_real,
    B_real,
    variance: float,
    spread: float,
    max_iter: int,
    callback=None,
    z_start=None,
):
    delta = variance - spread  # Coefficient difference

    def negative_objective(z: np.ndarray) -> float:
        x = (X0_real.reshape(-1, 1) + B_real @ z.reshape(-1, 1)).flatten()

        norm_squared = np.sum(x**2)
        # sum_i exp(-x^T D_i x) = exp(-spread * ||x||^2) * sum_i exp(-delta * x[i]^2)
        global_factor = np.exp(-spread * norm_squared)
        individual_sum = np.sum(np.exp(-delta * x**2))

        return -global_factor * individual_sum

    def first_derivative_objective(z: np.ndarray) -> np.ndarray:
        # (-1) e^{-(x_0 + Bz)^T D (x_0+Bz)}(B^T(D + D^T) (x_0 + B z))
        # where D_i is symmetric
        # It can be computed more efficiently as follows:
        # B^T (-2e^{-(-spread * ||x||^2} * [ spread * x * sum_i e^{-delta * x[i]^2} + delta * elementwise(x, e^{-delta * x^2}) ]
        # where delta = variance - spread
        x = X0_real.reshape(-1, 1) + (B_real @ z.reshape(-1, 1))
        x_flat = x.flatten()

        # Compute quadratic forms efficiently
        norm_squared = np.sum(x_flat**2)
        global_factor = np.exp(-spread * norm_squared)
        exp_terms = np.exp(-delta * x**2)
        individual_sum = np.sum(exp_terms)

        dx = -2 * global_factor * (spread * x * individual_sum + delta * x * exp_terms)
        grad = B_real.T @ dx

        return -grad.reshape(-1)

    if z_start is None:
        z_start = np.zeros(B_real.shape[1], dtype=float)

    result = optimize.minimize(
        negative_objective,
        z_start,
        jac=first_derivative_objective,
        callback=callback,
        method="L-BFGS-B",
        options={
            "maxiter": max_iter,
            "disp": True,
        },
    )

    return result


def sparse_prior_solution(Y: np.ndarray, A: np.ndarray, max_iter=10000):

    X0 = np.linalg.pinv(A) @ Y

    U, S, Vh = np.linalg.svd(A)
    rank = np.sum(S > 1e-10)

    # Check if there is a null space
    if rank == A.shape[1]:
        return X0

    # Null space basis vectors are the last (num_sources - rank) rows of Vh
    # TODO: this might need to be conjugated
    B = Vh[rank:].conj().T  # Shape (num_sources, dim_null_space)

    X0_real = to_real_augmented(X0)
    B_real = to_real_augmented(B)

    # Optimize the objective function in the null space
    result = optimize_real_valued_objective(
        X0_real, B_real, variance=1.0, spread=0.005, max_iter=max_iter
    )

    # Reconstruct the solution
    z_opt = result.x
    X_opt_real = X0_real + B_real @ z_opt.reshape(-1, 1)
    X_opt = from_real_augmented(X_opt_real)

    return X_opt
