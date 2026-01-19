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


def covariance_diagonals(
    num_sources: int, variance: float = 1.0, spread: float = 0.005
) -> np.ndarray:
    return [
        np.array([variance if j == i else spread for j in range(num_sources)])
        for i in range(num_sources)
    ]


def optimize_real_valued_objective(
    X0_real, B_real, D_diagonals, callback=None, z_start=None
):

    def negative_objective(z: np.ndarray) -> float:
        # numpy broadcasting will make row vectors if z is 1D
        # X0_real has shape (N,) whereas B_real @ z has shape (N, 1)
        x = X0_real.reshape(-1, 1) + (B_real @ z.reshape(-1, 1)).reshape(-1, 1)
        x_flat = x.flatten()  # Convert to 1D for element-wise operations
        return -sum(
            np.exp(-np.dot(d_i * x_flat, x_flat))  # Returns scalar: x^T D x
            for d_i in D_diagonals
        )

    def first_derivative_objective(z: np.ndarray) -> np.ndarray:
        # (-1) e^{-(x_0 + Bz)^T D (x_0+Bz)}(B^T(D + D^T) (x_0 + B z))
        # where D_i is symmetric
        x = X0_real.reshape(-1, 1) + (B_real @ z.reshape(-1, 1))
        grad = np.zeros_like(z)
        for d_i in D_diagonals:
            exp_term = np.exp(-np.sum(d_i * (x**2), axis=0))
            grad += -exp_term * (B_real.T @ (2 * d_i.reshape(-1, 1) * x)).flatten()
        return grad

    if z_start is None:
        z_start = np.zeros(B_real.shape[1])

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
    B = Vh[rank:].conj().T  # Shape (num_sources, dim_null_space)

    X0_real = to_real_augmented(X0)
    B_real = to_real_augmented(B)

    # Save memory by only forming the diagonals of the covariance matrices
    D_diagonals = covariance_diagonals(A.shape[1])

    # Optimize the objective function in the null space
    result = optimize_real_valued_objective(X0_real, B_real, D_diagonals)

    # Reconstruct the solution
    z_opt = result.x
    X_opt_real = X0_real + B_real @ z_opt.reshape(-1, 1)
    X_opt = from_real_augmented(X_opt_real)

    return X_opt, B
