import numpy as np
import scipy.optimize as optimize
from scipy.sparse import csr_matrix, diags
from cs_priors.solvers.representations import (
    augmented_real_vector_to_complex_matrix,
    complex_matrix_to_augmented_real_vector,
    mixing_tensor_to_frequency_major_matrix,
)


"""
Sparse prior solver using sparse diagonal precision operators.

# Solve min_Z -sum_g exp(-Z^T L^{(g)} Z) subject to A Z = Y
Where Z = X_pinv + B z

This solver now supports two explicit grouping modes for vector problems:
    - "none": mirrors sparse_prior.py and treats each augmented-real coordinate
      as its own prior component.
    - "complex_pairs": groups the real/imaginary pair of each complex
      coefficient together.
    - "frequencies": groups the coefficients across frequencies for each source

explicit block-vector API.
"""


def _singleton_groups(num_augmented: int) -> list[np.ndarray]:
    return [np.array([i], dtype=int) for i in range(num_augmented)]


def _source_groups_frequency_major(
    num_sources: int, num_freqs: int
) -> list[np.ndarray]:
    return [
        np.arange(s, num_sources * num_freqs, num_sources, dtype=int)
        for s in range(num_sources)
    ]


def _lift_groups_to_real_imag(
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


def _build_precision_matrices(
    X0: np.ndarray,
    grouping: str,
    precision: float,
    eps: float,
) -> list[csr_matrix]:

    S, F = X0.shape  # sources and frequencies

    if grouping == "none":
        groups = _singleton_groups(2 * S * F)
        diagonals = _groups_to_precision_diagonals(
            size=2 * S * F,
            groups=groups,
            precision=precision,
            eps=eps,
        )
    elif grouping == "complex_pairs":
        complex_groups = _singleton_groups(S * F)
        augmented_groups = _lift_groups_to_real_imag(complex_groups, S * F)
        diagonals = _groups_to_precision_diagonals(
            size=2 * S * F,
            groups=augmented_groups,
            precision=precision,
            eps=eps,
        )
    elif grouping == "frequencies":
        # Lg[g] = Lg[SF+g] = precision*I_F, for all other groups, Lg = eps*I_F
        # Lg = diag(eps*I_F, eps*I_F, ..., precision*I_F at the g-th group, ..., eps*I_F, precision*I_F at the SF+g-th group, ..., eps*I_F)
        complex_groups = _source_groups_frequency_major(S, F)
        augmented_groups = _lift_groups_to_real_imag(complex_groups, S * F)
        diagonals = _groups_to_precision_diagonals(
            size=2 * S * F,
            groups=augmented_groups,
            precision=precision,
            eps=eps,
        )

    else:
        raise ValueError(
            "Unknown grouping strategy "
            f"{grouping!r}. Supported modes are 'none', 'complex_pairs', "
            "and 'frequencies'."
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


def optimize_objective(
    X0_real, B_real, precision_operators, callback=None, z_start=None
):
    # factory function for the objective to minimize
    def negative_objective(z: np.ndarray) -> float:
        # numpy broadcasting will make row vectors if z is 1D
        # X0_real has shape (N,) whereas B_real @ z has shape (N, 1)
        x = X0_real.reshape(-1, 1) + (B_real @ z.reshape(-1, 1)).reshape(-1, 1)
        return -sum(np.exp(-quad_form(x, Lg)) for Lg in precision_operators)

    def first_derivative_objective(z: np.ndarray) -> np.ndarray:
        # (-1) e^{-(x_0 + Bz)^T D (x_0+Bz)}(B^T(D + D^T) (x_0 + B z))
        # where D_i is symmetric
        x = X0_real.reshape(-1, 1) + (B_real @ z.reshape(-1, 1))
        grad = np.zeros_like(z)
        for Lg in precision_operators:
            qf = quad_form(x, Lg)
            exp_neg_qf = np.exp(-qf)
            grad += 2 * exp_neg_qf * (B_real.T @ (Lg @ x)).flatten()
        return grad  # this is the negative, not actual

    def second_derivative_objective(z: np.ndarray) -> np.ndarray:
        # sum_i e^{-(x_0 + Bz)^T D_i (x_0+Bz)}
        #  * [ (B^T (D_i + D_i^T) (x_0 + B z)) (B^T (D_i + D_i^T) (x_0 + B z))^T
        #      - B^T (D_i + D_i^T) B ]
        x = X0_real.reshape(-1, 1) + (B_real @ z.reshape(-1, 1))
        hess = np.zeros((z.shape[0], z.shape[0]))
        for Lg in precision_operators:
            qf = quad_form(x, Lg)
            exp_neg_qf = np.exp(-qf)
            BdX = B_real.T @ (Lg @ x)
            hess += exp_neg_qf * (2 * np.outer(BdX, BdX) - 2 * (B_real.T @ Lg @ B_real))
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

    return z_opt, x_opt, res


def core_sparse_prior(X0, A, precision_operators) -> tuple[np.ndarray, np.ndarray]:
    """
    Args:
        X0: (S,F) complex initial solution
        A: (M,S,F) complex mixing matrix
    Returns:
        x_opt_complex: optimized solution with sparse prior
        B: basis for the null space of A (if it exists), otherwise None
    """
    M, S, F = A.shape

    # 1. block structure to find null space
    # TODO: never form A big explicitly to conserve memory
    A_big = mixing_tensor_to_frequency_major_matrix(A)  # (MF, SF)
    U, SingularValues, Vt = np.linalg.svd(A_big)
    rank = np.sum(SingularValues > 1e-10)

    # Check if there is a null space
    if rank == S * F:  # A_big is (MF, SF), so shape[1] is SF
        # No null space, return the pseudoinverse solution
        return X0, None

    # Compute the basis for the null space
    B = Vt[rank:].conj().T  # (SF, SF-rank) -> (SF, K) where K is the nullity of A_big

    # Convert X0 and B to augmented-real form for optimization
    B_real = np.block([[B.real, -B.imag], [B.imag, B.real]])
    X0_real = complex_matrix_to_augmented_real_vector(X0)

    # Optimize in augmented-real coordinates
    z_opt, x_opt_real, res = optimize_objective(X0_real, B_real, precision_operators)

    x_opt_complex = augmented_real_vector_to_complex_matrix(x_opt_real, S, F)

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
            'none', 'complex_pairs', and 'frequencies'
        precision: precision value for the sparse prior
        eps: small value indicating how sparse the prior tries to enforce the solution to be (lower means more sparse)

    Returns:
        x_opt_complex: optimized solution with sparse prior
    """

    precision_operators = _build_precision_matrices(X0, grouping, precision, eps)
    x_opt_complex, _ = core_sparse_prior(X0, A, precision_operators)

    return x_opt_complex


if __name__ == "__main__":

    # Example usage
    np.random.seed(2025)
    num_sources = 3
    num_mics = 3
    num_frequencies = 2

    # Random mixing matrix
    A = np.random.randn(num_mics, num_sources, num_frequencies) + 1j * np.random.randn(
        num_mics, num_sources, num_frequencies
    )

    # solution
    X = np.zeros((num_sources, num_frequencies), dtype=complex)
    for i in [0, num_sources - 1]:
        X[i, :] = np.random.randn(num_frequencies) + 1j * np.random.randn(
            num_frequencies
        )

    # Generate measurements
    Y = np.einsum("m s f, s f -> m f", A, X)

    # Initial solution using pseudoinverse
    X0 = np.zeros_like(X)
    for f in range(num_frequencies):
        A_f = A[:, :, f]
        Y_f = Y[:, f]
        X0[:, f] = np.linalg.pinv(A_f) @ Y_f

    from cs_priors.plotting.plot_complex import plot_matrices
    from matplotlib import pyplot as plt

    # plot_matrices(
    #     [X, X0],
    #     titles=["True solution X", "Initial solution X0"],
    # )

    for grouping in ["none", "complex_pairs", "frequencies"]:
        L = _build_precision_matrices(X0, grouping, precision=1.0, eps=0.2)

        plot_matrices(
            [Li.toarray().astype(complex) for Li in L],
            titles=[f"Precmat {grouping=} {X0.shape=}"] * len(L),
        )

        x_opt_complex = sparse_prior_solution(
            X0, A, grouping=grouping, precision=1.0, eps=0.005
        )

        error_Xopt = np.linalg.norm(X.reshape(-1, 1) - x_opt_complex.reshape(-1, 1))
        error_X0 = np.linalg.norm(X.reshape(-1, 1) - X0.reshape(-1, 1))
        print(f"[{grouping}] Reconstruction error (initial): {error_X0:.4f}")
        print(f"[{grouping}] Reconstruction error (optimized): {error_Xopt:.4f}")
        print(f"[{grouping}] Optimized solution:\n", x_opt_complex)
        plt.show()