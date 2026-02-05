"""
Group LASSO for frequency-domain source localization.

Groups frequencies from the same direction/angle together to exploit
the co-occurrence phenomenon where multiple frequencies originate from
the same physical source.

Reference: https://group-lasso.readthedocs.io/en/latest/
"""

import numpy as np

# from typing import Tuple, Optional
from group_lasso import GroupLasso


def create_frequency_groups(num_sources: int, num_frequencies: int) -> np.ndarray:
    """
    Create group indices for augmented real form where each source (direction)
    has multiple frequencies grouped together.

    Structure in augmented real form X_R:
    [Re(X_1) Re(X_2) ... Re(X_N)
    Im(X_1) Im(X_2) ... Im(X_N)]
    where each X_j has F frequencies:
    X_j = [X_j[1], X_j[2], ..., X_j[F]]


    Args:
        num_sources: Number of sources/directions (N)
        num_frequencies: Number of frequency bins (F)

    Returns:
        groups: Array of shape (2*N*F,) where groups[i] indicates which group
                coefficient i belongs to. Groups are numbered 0 to N-1.
    """
    groups = np.zeros(2 * num_sources * num_frequencies, dtype=int)

    # Each source has 2*F coefficients (F real + F imaginary)
    group_size = 2 * num_frequencies

    # maps the F consecutive indices of the real part and F consecutive indices of the imaginary part to the same group
    # G[iF:(i+1)F] = i
    # G[(N+i)F:(N+i+1)F] = i
    for source_idx in range(num_sources):
        start_real = source_idx * num_frequencies
        start_imag = (num_sources + source_idx) * num_frequencies
        groups[start_real : start_real + num_frequencies] = source_idx
        groups[start_imag : start_imag + num_frequencies] = source_idx

    return groups


def complex_block_to_augmented_real(
    X_complex: np.ndarray, num_sources: int, num_frequencies: int
) -> np.ndarray:
    """
    Convert complex block vector to augmented real form.

    Input X_complex has shape (N*F, 1) structured as:
    [X_1[1], ..., X_1[F], X_2[1], ..., X_2[F], ..., X_N[1], ..., X_N[F]]

    Output X_R has shape (2*N*F, 1) structured as:
    [Re(X_1[1]), ..., Re(X_1[F]), Im(X_1[1]), ..., Im(X_1[F]),
     Re(X_2[1]), ..., Re(X_2[F]), Im(X_2[1]), ..., Im(X_2[F]),
     ...,
     Re(X_N[1]), ..., Re(X_N[F]), Im(X_N[1]), ..., Im(X_N[F])]

    Args:
        X_complex: Complex block vector (N*F, 1)
        num_sources: Number of sources (N)
        num_frequencies: Number of frequencies (F)

    Returns:
        X_R: Augmented real vector (2*N*F, 1)
    """
    X_R = np.zeros((2 * num_sources * num_frequencies, 1))

    for source_idx in range(num_sources):
        # Extract frequencies for this source
        freq_start = source_idx * num_frequencies
        freq_end = freq_start + num_frequencies
        X_source = X_complex[freq_start:freq_end]

        # Place real and imaginary parts in augmented form
        aug_start = source_idx * 2 * num_frequencies
        aug_real_end = aug_start + num_frequencies
        aug_imag_end = aug_real_end + num_frequencies

        X_R[aug_start:aug_real_end] = X_source.real
        X_R[aug_real_end:aug_imag_end] = X_source.imag

    return X_R


def augmented_real_to_complex_block(
    X_R: np.ndarray, num_sources: int, num_frequencies: int
) -> np.ndarray:
    """
    Convert augmented real form back to complex block vector.

    Args:
        X_R: Augmented real vector (2*N*F, 1)
        num_sources: Number of sources (N)
        num_frequencies: Number of frequencies (F)

    Returns:
        X_complex: Complex block vector (N*F, 1)
    """
    X_complex = np.zeros((num_sources * num_frequencies, 1), dtype=complex)

    for source_idx in range(num_sources):
        # Extract real and imaginary parts for this source
        aug_start = source_idx * 2 * num_frequencies
        aug_real_end = aug_start + num_frequencies
        aug_imag_end = aug_real_end + num_frequencies

        X_real = X_R[aug_start:aug_real_end]
        X_imag = X_R[aug_real_end:aug_imag_end]

        # Reconstruct complex values
        freq_start = source_idx * num_frequencies
        freq_end = freq_start + num_frequencies
        X_complex[freq_start:freq_end] = X_real + 1j * X_imag

    return X_complex


def block_mixing_matrix_to_augmented_real(
    A_complex: np.ndarray, num_mics: int, num_sources: int, num_frequencies: int
) -> np.ndarray:
    """
    Convert complex block mixing matrix to augmented real form.

    Args:
        A_complex: Complex block matrix (P*F, N*F)
        num_mics: Number of microphones (P)
        num_sources: Number of sources (N)
        num_frequencies: Number of frequencies (F)

    Returns:
        A_R: Augmented real matrix (2*P*F, 2*N*F)
    """
    A_real = A_complex.real
    A_imag = A_complex.imag

    # Build augmented real form: [Re(A), -Im(A); Im(A), Re(A)]
    A_R = np.block([[A_real, -A_imag], [A_imag, A_real]])

    return A_R


def frequency_group_lasso(
    Y_complex: np.ndarray,
    A_complex: np.ndarray,
    num_mics: int,
    num_sources: int,
    num_frequencies: int,
    alpha: float = 1.0,
    fit_intercept: bool = False,
) -> np.ndarray:
    """
    Solve group LASSO problem with frequency grouping.

    Minimizes: 0.5 * ||Y - A*X||_2^2 + alpha * sum_j ||X_j||_2

    where X_j contains all frequencies (real and imaginary) for source j.

    Args:
        Y_complex: Complex measurements (P*F, 1)
        A_complex: Complex block mixing matrix (P*F, N*F)
        num_mics: Number of microphones (P)
        num_sources: Number of sources (N)
        num_frequencies: Number of frequencies (F)
        alpha: Regularization parameter
        fit_intercept: Whether to fit an intercept term

    Returns:
        X_opt: Optimized complex block vector (N*F, 1)
    """
    # Convert to augmented real form
    Y_R = np.concatenate([Y_complex.real, Y_complex.imag], axis=0)
    A_R = block_mixing_matrix_to_augmented_real(
        A_complex, num_mics, num_sources, num_frequencies
    )

    # Create groups: each source (direction) is one group
    groups = create_frequency_groups(num_sources, num_frequencies)

    # Solve group LASSO
    model = GroupLasso(
        groups=groups,
        group_reg=alpha,
        l1_reg=0.0,  # Pure group LASSO, no additional L1
        fit_intercept=fit_intercept,
        scale_reg="group_size",  # Scale by sqrt(group_size) as is standard
        supress_warning=True,
        n_iter=100000,
    )

    model.fit(A_R, Y_R.ravel())
    X_R_opt = model.coef_.reshape(-1, 1)

    # Convert back to complex form
    X_opt = augmented_real_to_complex_block(X_R_opt, num_sources, num_frequencies)

    return X_opt


def tensor_to_block_matrix(A: np.ndarray) -> np.ndarray:
    """
    Args
        A: 3D tensor of shape (P, N, F)

    Returns:
        A_block: 2D block matrix of shape (P*F, N*F) where each block is a diagonal matrix of size (F, F) of frequencies for each source
    """
    P, N, F = A.shape
    A_block = np.zeros((P * F, N * F), dtype=A.dtype)

    for p in range(P):
        for n in range(N):
            A_block[p * F : (p + 1) * F, n * F : (n + 1) * F] = np.diag(A[p, n, :])

    return A_block


def matrix_to_block_vector(X: np.ndarray) -> np.ndarray:
    """
    Args:
        X: 2D matrix of shape (N, F)

    Returns:
        X_block: 2D block vector of shape (N*F, 1) where each block is a column vector of size (F, 1) of frequencies for each source
    """
    N, F = X.shape
    X_block = np.zeros((N * F, 1), dtype=X.dtype)

    for n in range(N):
        X_block[n * F : (n + 1) * F, 0] = X[n, :]

    return X_block


def color_groups(X: np.ndarray):
    """
    Interprets the block vector of X and returns a vector of same shape with a distinct complex number in all entries of each group.

    Args:
        X: complex matrix of shape (N, F)

    Returns:
        group_matrix: A matrix of shape (N, F) where each group has a distinct complex number.
        group_block_vector: A block vector of shape (N*F, 1) where each group has a distinct complex number.
        group_augmented_real: A matrix of shape (2*N*F, 1) where each group has a distinct complex number.
    """
    num_sources, num_frequencies = X.shape
    groups = create_frequency_groups(num_sources, num_frequencies)
    # create a mapping as groups, but to a distinct complex color for each group. There are num_sources groups.
    group_color = {i: np.exp(2j * np.pi * i / num_sources) for i in range(num_sources)}

    X_block = matrix_to_block_vector(X)
    X_R = complex_block_to_augmented_real(X_block, num_sources, num_frequencies)
    group_matrix = np.zeros_like(X, dtype=complex)
    group_block_vector = np.zeros_like(X_block, dtype=complex)
    group_augmented = np.zeros_like(X_R, dtype=complex)

    # group_block_vector[i] = groups[i]
    for r in range(2):
        for n in range(X.shape[0]):
            for f in range(X.shape[1]):
                group_idx = r * num_sources * num_frequencies + n * num_frequencies + f
                idx = group_idx
                if r == 0:
                    group_matrix[n, f] = group_color[groups[idx]]
                    group_block_vector[n * num_frequencies + f, 0] = group_color[
                        groups[idx]
                    ]
                    group_augmented[idx, 0] = group_color[groups[idx]]
                else:
                    group_augmented[idx, 0] = group_color[groups[idx]]
    return group_matrix, group_block_vector, group_augmented


def real_augmented_to_matrix(
    X_R: np.ndarray, num_sources: int, num_frequencies: int
) -> np.ndarray:
    """
    Convert augmented real form block matrix back to complex non-block matrix.

    Args:
        X_R: Augmented real vector (2*N*F, 1)
        num_sources: Number of sources (N)
        num_frequencies: Number of frequencies (F)
        structured as [Re(X_1) Re(X_2) ... Re(X_N)
                      Im(X_1) Im(X_2) ... Im(X_N)]
        Where each X_j has F frequencies:
        X_j = [X_j[1], X_j[2], ..., X_j[F]]

    Returns:
        X_complex: Complex matrix (N, F)
    """
    assert X_R.shape == (2 * num_sources * num_frequencies, 1)
    X_complex = np.zeros((num_sources, num_frequencies), dtype=complex)

    for n in range(num_sources):
        # Extract real and imaginary parts for this source
        # i:i+F, i+NF: i+NF+F
        real_start = n * num_frequencies
        real_end = real_start + num_frequencies
        imag_start = (num_sources + n) * num_frequencies
        imag_end = imag_start + num_frequencies
        X_real = X_R[real_start:real_end]
        X_imag = X_R[imag_start:imag_end]

        # Reconstruct complex values
        X_complex[n, :] = (X_real + 1j * X_imag).flatten()

    return X_complex


def block_vector_to_matrix(
    X_block: np.ndarray, num_sources: int, num_frequencies: int
) -> np.ndarray:
    """
    Convert block vector back to complex non-block matrix.

    Args:
        X_block: Block vector (N*F, 1)
        num_sources: Number of sources (N)
        num_frequencies: Number of frequencies (F)

    Returns:
        X_complex: Complex matrix (N, F)
    """
    assert X_block.shape == (num_sources * num_frequencies, 1)
    X_complex = np.zeros((num_sources, num_frequencies), dtype=complex)

    for n in range(num_sources):
        freq_start = n * num_frequencies
        freq_end = freq_start + num_frequencies
        X_complex[n, :] = X_block[freq_start:freq_end].flatten()

    return X_complex


if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path
    import matplotlib.pyplot as plt

    # Add src directory to path for absolute imports when running as script
    src_dir = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(src_dir))

    from cs_priors.simulation.mixing_model import run_simulation
    from cs_priors.plotting.plotting import plot_matrices

    np.random.seed(42)

    num_mics = 10
    num_sources = 10
    s_sparse = 2

    sim = run_simulation(num_mics=num_mics, num_sources=num_sources, s_sparse=s_sparse)

    group_matrix, group_block_vector, group_augmented = color_groups(sim.X)

    plot_matrices(
        [sim.X, group_matrix, group_block_vector, group_augmented],
        titles=[
            "Original X",
            "Group Matrix",
            "Group Block Vector",
            "Group Augmented Real",
        ],
        show_values=True,
    )

    A_block = tensor_to_block_matrix(sim.C)

    P, N, F = sim.C.shape

    A_R = block_mixing_matrix_to_augmented_real(A_block, P, N, F)
    Y_block = matrix_to_block_vector(sim.Y)
    X0_block = np.linalg.pinv(A_block) @ Y_block
    X0 = block_vector_to_matrix(X0_block, N, F)

    X_true = matrix_to_block_vector(sim.X)

    X_gl = frequency_group_lasso(
        Y_complex=Y_block,
        A_complex=A_block,
        num_mics=P,
        num_sources=N,
        num_frequencies=F,
        alpha=1e-90,
    )

    X_gl_matrix = block_vector_to_matrix(X_gl, N, F)

    plot_matrices(
        [A_block, Y_block, sim.Y, X0, sim.X, X_gl_matrix],
        titles=[
            "Mixing Matrix A (block)",
            "Measurements Y (block)",
            "Measurements Y (original)",
            "Initial Estimate X0",
            "True X",
            "Group LASSO X",
        ],
        show_values=True,
    )
    plt.show()
