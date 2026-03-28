import numpy as np
import scipy.optimize as optimize
from scipy.sparse import csr_matrix, diags

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

"""
Sparse prior solver using sparse diagonal precision operators.

# Solve min_Z -sum_g exp(-Z^T L^{(g)} Z) subject to A Z = Y
Where Z = X_pinv + B z

This solver now supports two explicit grouping modes for vector problems:
    - "none": mirrors sparse_prior.py and treats each augmented-real coordinate
      as its own prior component.
    - "complex_pairs": groups the real/imaginary pair of each complex
      coefficient together.

The "frequencies" mode is intentionally deferred until the solver gains an
explicit block-vector API.
"""


def _as_column_vector(Z: np.ndarray) -> np.ndarray:
    Z = np.asarray(Z)
    if Z.ndim == 1:
        return Z.reshape(-1, 1)
    if Z.ndim == 2 and Z.shape[1] == 1:
        return Z
    raise ValueError(f"Expected a vector or column vector, got shape {Z.shape}")


def _augmented_coordinate_groups(num_augmented: int) -> list[np.ndarray]:
    return [np.array([i], dtype=int) for i in range(num_augmented)]


def _complex_pair_groups(num_complex: int) -> list[np.ndarray]:
    return [np.array([i], dtype=int) for i in range(num_complex)]


def _lift_complex_groups_to_augmented(
    complex_groups: list[np.ndarray], num_complex: int
) -> list[np.ndarray]:
    return [
        np.concatenate([group, group + num_complex]).astype(int)
        for group in complex_groups
    ]


def _groups_to_precision_diagonals(
    size: int,
    groups: list[np.ndarray],
    precision: float,
    eps: float,
) -> list[np.ndarray]:
    diagonals = []
    for group in groups:
        diagonal = np.full(size, eps, dtype=float)
        diagonal[group] = precision
        diagonals.append(diagonal)
    return diagonals


def _diagonals_to_sparse_operators(diagonals: list[np.ndarray]) -> list[csr_matrix]:
    return [diags(diagonal, offsets=0, format="csr") for diagonal in diagonals]


def _build_precision_operators(
    X0: np.ndarray,
    grouping: str,
    precision: float,
    eps: float,
) -> list[csr_matrix]:
    num_complex = X0.shape[0]
    num_augmented = 2 * num_complex

    if grouping == "none":
        groups = _augmented_coordinate_groups(num_augmented)
        diagonals = _groups_to_precision_diagonals(
            size=num_augmented,
            groups=groups,
            precision=precision,
            eps=eps,
        )
    elif grouping == "complex_pairs":
        complex_groups = _complex_pair_groups(num_complex)
        augmented_groups = _lift_complex_groups_to_augmented(
            complex_groups, num_complex
        )
        diagonals = _groups_to_precision_diagonals(
            size=num_augmented,
            groups=augmented_groups,
            precision=precision,
            eps=eps,
        )
    elif grouping == "frequencies":
        raise NotImplementedError(
            "grouping='frequencies' is intentionally deferred. "
            "It requires block-vector input plus explicit block metadata "
            "(num_sources and num_frequencies)."
        )
    else:
        raise ValueError(
            "Unknown grouping strategy "
            f"{grouping!r}. Supported modes are 'none', 'complex_pairs', "
            "and 'frequencies' (deferred)."
        )

    return _diagonals_to_sparse_operators(diagonals)


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


def optimize_objective(X0_real, B_real, precision_operators, callback=None, z_start=None):
    # factory function for the objective to minimize
    def negative_objective(z: np.ndarray) -> float:
        # numpy broadcasting will make row vectors if z is 1D
        # X0_real has shape (N,) whereas B_real @ z has shape (N, 1)
        x = X0_real.reshape(-1, 1) + (B_real @ z.reshape(-1, 1)).reshape(-1, 1)
        return -sum(np.exp(-quad_form(x, D_i)) for D_i in precision_operators)

    def first_derivative_objective(z: np.ndarray) -> np.ndarray:
        # (-1) e^{-(x_0 + Bz)^T D (x_0+Bz)}(B^T(D + D^T) (x_0 + B z))
        # where D_i is symmetric
        x = X0_real.reshape(-1, 1) + (B_real @ z.reshape(-1, 1))
        grad = np.zeros_like(z)
        for D_i in precision_operators:
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
        for D_i in precision_operators:
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


def core_sparse_prior(X0, A, precision_operators) -> tuple[np.ndarray, np.ndarray]:
    """
    Args:
        X0: complex initial solution
        A: complex mixing matrix
    Returns:
        x_opt_complex: optimized solution with sparse prior
        B: basis for the null space of A (if it exists), otherwise None
    """
    X0 = _as_column_vector(X0)

    U, S, Vt = np.linalg.svd(A)
    rank = np.sum(S > 1e-10)

    # Check if there is a null space
    if rank == A.shape[1]:  # A is mics x sources
        # No null space, return the pseudoinverse solution
        return X0, None

    # Compute the basis for the null space
    B = Vt[rank:].conj().T
    B_real = np.block([[B.real, -B.imag], [B.imag, B.real]])
    X0_real = to_real_augmented(X0)
    z_opt, x_opt, x_opt_complex, res = optimize_objective(
        X0_real, B_real, precision_operators
    )
    return x_opt_complex, B


def sparse_prior_solution(
    X0: np.ndarray,
    A: np.ndarray,
    grouping: str,
    precision: float = 1.0,
    eps: float = 0.005,
) -> np.ndarray:
    """
    Args:
        X0: complex initial solution
        A: complex mixing matrix
        grouping: grouping strategy for the prior. Supported modes are
            'none', 'complex_pairs', and 'frequencies' (deferred)
        precision: precision value for the sparse prior
        eps: small value indicating how sparse the prior tries to enforce the solution to be (lower means more sparse)

    Returns:
        x_opt_complex: optimized solution with sparse prior
    """
    X0 = _as_column_vector(X0)
    precision_operators = _build_precision_operators(X0, grouping, precision, eps)
    x_opt_complex, _ = core_sparse_prior(X0, A, precision_operators)

    return x_opt_complex


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

    for grouping in ["none", "complex_pairs"]:
        x_opt_complex = sparse_prior_solution(
            X0, A, grouping=grouping, precision=1.0, eps=0.005
        )

        error_Xopt = np.linalg.norm(X.reshape(-1, 1) - x_opt_complex.reshape(-1, 1))
        error_X0 = np.linalg.norm(X.reshape(-1, 1) - X0.reshape(-1, 1))
        print(f"[{grouping}] Reconstruction error (initial): {error_X0:.4f}")
        print(f"[{grouping}] Reconstruction error (optimized): {error_Xopt:.4f}")
        print(f"[{grouping}] Optimized solution:\n", x_opt_complex)
