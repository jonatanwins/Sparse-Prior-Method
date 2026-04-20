"""
Group LASSO solver for frequency-domain source recovery.

Supports multiple grouping strategies via the ``grouping`` keyword:

    "frequency"  - All freq bins of a source share a group (co-occurrence prior).
    "none"       - Every coefficient is its own group (equivalent to plain LASSO).
    "real_imag"  - Each (Re, Im) pair is a size-2 group (straw-man structure).
    "random"     - Randomly assign coefficients to S equal-sized groups.

Public API:
    group_lasso_solve(sim, alpha, grouping, max_iter, seed) -> X_pred (S x N)
"""

import numpy as np
from group_lasso import GroupLasso

from cs_priors.solvers.representations import (
    _as_column_vector,
    augmented_real_vector_to_complex_matrix,
    complex_matrix_to_augmented_real_matrix,
    complex_matrix_to_augmented_real_vector,
    frequency_major_vector_to_matrix,
    matrix_to_frequency_major_vector,
    mixing_tensor_to_frequency_major_matrix,
    ensure_frequency_system_shapes,
)


def frequency_groups(num_sources: int, num_freqs: int, **_) -> np.ndarray:
    groups_half = np.tile(np.arange(num_sources), num_freqs)  # [0 1 2 3 0 1 2 3 ...]
    return np.concatenate([groups_half, groups_half])  # real imag


def no_groups(num_sources: int, num_freqs: int, **_) -> np.ndarray:
    return np.arange(2 * num_sources * num_freqs)  # [0, 1, 2, ..., 2SF-1]


def real_imag_groups(num_sources: int, num_freqs: int, **_) -> np.ndarray:
    labels = np.arange(num_sources * num_freqs)  # [0, 1, 2, ..., SF-1]
    return np.concatenate([labels, labels])  # [0, 1, 2, ..., SF-1, 0, 1, 2, ..., SF-1]


def random_groups(
    num_sources: int, num_freqs: int, *, seed: int = 0, **_
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    groups_half = np.tile(
        np.arange(num_sources), num_freqs
    )  # start with frequency grouping
    rng.shuffle(groups_half)  # shuffle source/frequency information
    return np.concatenate(
        [groups_half, groups_half]
    )  # keep real/imag to separate out frequency knowledge


_GROUP_STRATEGIES = {
    "frequency": frequency_groups,
    "none": no_groups,
    "real_imag": real_imag_groups,
    "random": random_groups,
}

# ---------------------------------------------------------------------------
# Public solver
# ---------------------------------------------------------------------------


# TODO: split into core_group_lasso_solve and a convenience wrapper once you understand the proximal operator math.
# TODO: Change
def frequency_group_lasso_solve(
    Y: np.ndarray,
    A: np.ndarray,
    alpha: float = 1e-4,
    grouping: str = "frequency",
    max_iter: int = 5000,
    seed: int = 0,
) -> np.ndarray:
    A, Y, _, _ = ensure_frequency_system_shapes(A, Y)
    _, num_sources, num_freqs = A.shape

    if grouping not in _GROUP_STRATEGIES:
        raise ValueError(
            f"Unknown grouping {grouping!r}. "
            f"Choose from {list(_GROUP_STRATEGIES.keys())}"
        )

    A_big = mixing_tensor_to_frequency_major_matrix(A)  # (MF, SF)
    A_real = complex_matrix_to_augmented_real_matrix(A_big)  # (2MF, 2SF)
    Y_real = complex_matrix_to_augmented_real_vector(Y).ravel()  # (2MF,)

    groups = _GROUP_STRATEGIES[grouping](num_sources, num_freqs, seed=seed)

    # TODO: dive deep in the math and understand how to control group lasso.
    group_reg = alpha  # A_big.shape[0] * alpha

    model = GroupLasso(
        groups=groups,
        group_reg=group_reg,
        n_iter=max_iter,
        fit_intercept=False,
        supress_warning=True,
    )
    model.fit(A_real, Y_real)

    X_real = model.coef_.reshape(-1, 1)  # (2SF, 1)
    X_hat = augmented_real_vector_to_complex_matrix(
        X_real, num_sources, num_freqs
    )  # (S, F)

    return X_hat


# ---------------------------------------------------------------------------
# More flexible implementation, Work in progress.
# ---------------------------------------------------------------------------
def group_lasso_custom_solve(
    Y: np.ndarray,
    A: np.ndarray,
    group_reg: float,
    l1_reg: float,
    frobenius_lipschitz: bool,
    scale_reg="inverse_group_size",
    subsampling_scheme=1,
    n_iter=1000,
    tol=1e-3,
    warm_start=True,
    grouping: str = "frequency",
    seed: int = 0,
) -> np.ndarray:

    if grouping not in _GROUP_STRATEGIES:
        raise ValueError(
            f"Unknown grouping {grouping!r}. "
            f"Choose from {list(_GROUP_STRATEGIES.keys())}"
        )

    A_block, Y_block, S, N = _to_block_system(sim)

    # Real-augmented form for the real-valued GroupLasso optimizer
    A_real = to_real_augmented(A_block)
    Y_real = to_real_augmented(Y_block)

    groups = _GROUP_STRATEGIES[grouping](S, N, seed=seed)

    model = GroupLasso(
        groups=groups,
        group_reg=group_reg,
        l1_reg=l1_reg,
        n_iter=n_iter,
        tol=tol,
        scale_reg=scale_reg,
        subsampling_scheme=subsampling_scheme,
        fit_intercept=False,
        frobenius_lipschitz=frobenius_lipschitz,
        warm_start=warm_start,
        supress_warning=True,
    )
    model.fit(A_real, Y_real.ravel())

    # Back to complex block vector, then reshape to (S, N)
    X_block_complex = from_real_augmented(model.coef_.reshape(-1, 1))
    X_pred = X_block_complex.reshape(S, N)

    return X_pred


if __name__ == "__main__":
    np.random.seed(0)

    M, S, F = 2, 3, 2

    # 1. frequency grouping: same source across frequencies, and same labels in real/imag halves
    g = frequency_groups(S, F)
    assert np.array_equal(g, np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]))

    # 2. no grouping: every augmented-real coordinate gets its own label
    g = no_groups(S, F)
    assert np.array_equal(g, np.arange(2 * S * F))

    # 3. real_imag grouping: each complex coefficient gets one shared Re/Im label
    g = real_imag_groups(S, F)
    assert np.array_equal(g, np.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]))

    # 4. random grouping: same multiset as frequency grouping, but shuffled
    g = random_groups(S, F, seed=0)
    assert np.array_equal(np.sort(g[: S * F]), np.array([0, 0, 1, 1, 2, 2]))
    assert np.array_equal(g[: S * F], g[S * F :])

    # 5. solver smoke test
    A = np.zeros((M, S, F), dtype=complex)
    A[:, :, 0] = np.array([[1.0, 0.2, 0.0], [0.0, 1.0, 0.3]], dtype=complex)
    A[:, :, 1] = np.array([[1.0, -0.1, 0.0], [0.0, 1.0, 0.4]], dtype=complex)

    X_true = np.array(
        [
            [1.0 + 0.5j, 0.8 - 0.2j],
            [0.0 + 0.0j, 0.0 + 0.0j],
            [0.2 - 0.1j, 0.0 + 0.0j],
        ]
    )
    Y = np.einsum("msf,sf->mf", A, X_true)

    X_hat = frequency_group_lasso_solve(
        Y, A, alpha=1e-800, grouping="frequency", max_iter=5000
    )

    residual = np.linalg.norm(np.einsum("msf,sf->mf", A, X_hat) - Y)
    print("X_hat.shape:", X_hat.shape)
    print("residual:", residual)

    assert X_hat.shape == X_true.shape
    from cs_priors.plotting.plot_complex import plot_matrices
    import matplotlib.pyplot as plt

    X_pinv = np.linalg.pinv(
        mixing_tensor_to_frequency_major_matrix(A)
    ) @ matrix_to_frequency_major_vector(Y)
    X_pinv = frequency_major_vector_to_matrix(X_pinv, S, F)
    plot_matrices([X_true, X_pinv, X_hat], ["X_true", "X_pinv", "X_hat"])
    plt.show()

    assert residual < 1e-3

    print("frequency_group_lasso smoke tests passed")
