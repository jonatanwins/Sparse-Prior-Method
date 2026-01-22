import numpy as np
import scipy.optimize as optimize

'''
The Home of the Sparse Prior Method
'''


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


def covariance_matrices(
    num_sources: int, variance: float = 1.0, spread: float = 0.005
) -> list[np.ndarray]:
    return [
        np.diag([variance if j == i else spread for j in range(num_sources)])
        for i in range(num_sources)
    ]


# Quadratic form x^T D x for each grid point
def quad_form(points, D_matrix):
    # This function handles both a single column vector (shape N, 1)
    # and a batch of row vectors (shape ..., N)
    if points.shape[-1] == 1 and points.ndim > 1:
        # It's a single column vector, use the mathematical formula
        return (points.T @ D_matrix @ points).item()
    else:
        # It's a batch of row vectors, use the efficient NumPy way
        return (points @ D_matrix * points).sum(axis=-1)


def optimize_objective(X0_real, B_real, D, callback=None, z_start=None):
    # factory function for the objective to minimize
    def negative_objective(z: np.ndarray) -> float:
        # numpy broadcasting will make row vectors if z is 1D
        # X0_real has shape (N,) whereas B_real @ z has shape (N, 1)
        x = X0_real.reshape(-1, 1) + (B_real @ z.reshape(-1, 1)).reshape(-1, 1)
        return -sum(np.exp(-quad_form(x, D_i)) for D_i in D)

    def first_derivative_objective(z: np.ndarray) -> np.ndarray:
        # (-1) e^{-(x_0 + Bz)^T D (x_0+Bz)}(B^T(D + D^T) (x_0 + B z))
        # where D_i is symmetric
        x = X0_real.reshape(-1, 1) + (B_real @ z.reshape(-1, 1))
        grad = np.zeros_like(z)
        for D_i in D:
            qf = quad_form(x, D_i)
            exp_neg_qf = np.exp(-qf)
            grad += 2 * exp_neg_qf * (B_real.T @ (D_i @ x)).flatten()
        return grad  # this is the negative, not actual

    def second_derivative_objective(z: np.ndarray) -> np.ndarray:
        # sum_i e^{-(x_0 + Bz)^T D_i (x_0+Bz)}
        #  * [ (B^T (D_i + D_i^T) (x_0 + B z)) (B^T (D_i + D_i^T) (x_0 + B z))^T
        #      - B^T (D_i + D_i^T) B ]
        x = X0_real.reshape(-1, 1) + (B_real @ z.reshape(-1, 1))
        hess = np.zeros((z.shape[0], z.shape[0]))
        for D_i in D:
            qf = quad_form(x, D_i)
            exp_neg_qf = np.exp(-qf)
            BdX = B_real.T @ (D_i @ x)
            hess += exp_neg_qf * (
                2 * np.outer(BdX, BdX) - 2 * (B_real.T @ D_i @ B_real)
            )
        return -hess  # this is the negative Hessian, not the actual

    if z_start is None:
        z_start = np.zeros(B_real.shape[1])

    res = optimize.minimize(
        negative_objective,
        z_start,
        method='l-BFGS-B',
        callback=callback,
        jac=first_derivative_objective,
        # hess=second_derivative_objective,
    )
    # print(res.message)
    z_opt = res.x
    x_opt = X0_real.reshape(-1, 1) + (B_real @ z_opt.reshape(-1, 1))
    x_opt_complex = from_real_augmented(x_opt)
    return z_opt, x_opt, x_opt_complex, res


def sparse_prior_solution(X0, A) -> tuple[np.ndarray, np.ndarray]:
    """
    Args:
        X0: complex initial solution
        A: complex mixing matrix
    """
    U, S, Vt = np.linalg.svd(A)
    rank = np.sum(S > 1e-10)

    # Check if there is a null space
    if rank == A.shape[1]:  # A is mics x sources
        # No null space, return the pseudoinverse solution
        return X0, None

    # Compute the basis for the null space
    # TODO: Vt might need to be conjugated, i.e. Vh
    B = Vt[rank:].T
    B_real = np.block([[B.real, -B.imag], [B.imag, B.real]])
    X0_real = to_real_augmented(X0)
    D = covariance_matrices(num_sources=X0_real.shape[0])
    z_opt, x_opt, x_opt_complex, res = optimize_objective(X0_real, B_real, D)
    return x_opt_complex, B


if __name__ == "__main__":

    # Example usage
    np.random.seed(2025)
    num_sources = 10
    num_mics = 7

    # Random mixing matrix
    A = np.random.randn(num_mics, num_sources) + 1j * np.random.randn(
        num_mics, num_sources
    )

    # solution
    X = np.zeros((num_sources, 1), dtype=complex)
    for i in [1, 5]:
        X[i] = np.random.randn(1) + 1j * np.random.randn(1)

    # Generate measurements
    Y = A @ X

    # Initial solution using pseudoinverse
    X0 = np.linalg.pinv(A) @ Y

    x_opt_complex, B = sparse_prior_solution(X0, A)

    error_Xopt = np.linalg.norm(X.reshape(-1, 1) - x_opt_complex.reshape(-1, 1))
    error_X0 = np.linalg.norm(X.reshape(-1, 1) - X0.reshape(-1, 1))
    print(f"Reconstruction error (initial): {error_X0:.4f}")
    print(f"Reconstruction error (optimized): {error_Xopt:.4f}")
    print("Optimized solution:\n", x_opt_complex)
