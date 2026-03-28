"""
Group LASSO solver for frequency-domain source recovery.

Supports multiple grouping strategies via the ``grouping`` keyword:

    "frequency"  – All freq bins of a source share a group (co-occurrence prior).
    "none"       – Every coefficient is its own group (equivalent to plain LASSO).
    "real_imag"  – Each (Re, Im) pair is a size-2 group (straw-man structure).
    "random"     – Randomly assign coefficients to S equal-sized groups.

Public API:
    group_lasso_solve(sim, alpha, grouping, max_iter, seed) -> X_pred (S x N)
"""

import numpy as np
from group_lasso import GroupLasso
from ..simulation.Simulation import Simulation
from .real_augmented import to_real_augmented, from_real_augmented


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _to_block_system(sim: Simulation):
    """
    Stack the per-frequency-bin system Y[:,n] = A[:,:,n] @ X[:,n]
    into one large linear system:  Y_block = A_block @ X_block.

    Source-major layout (M x S grid of N x N diagonal blocks):
        A_block[(m,n), (s,n)] = A[m, s, n]
        Row index: m*N + n   (mic m, freq n)
        Col index: s*N + n   (source s, freq n)

    X_block is source-major: [X_0[0..N-1], X_1[0..N-1], ..., X_{S-1}[0..N-1]]
    Y_block is mic-major:    [Y_0[0..N-1], Y_1[0..N-1], ..., Y_{M-1}[0..N-1]]

    Returns:
        A_block, Y_block, S, N
    """
    M, S, N = sim.A.shape

    # A_block: M x S grid of N x N diagonal blocks
    # Block (m, s) = diag(A[m, s, :]) at row-block m, col-block s
    A_block = np.zeros((M * N, S * N), dtype=sim.A.dtype)
    for m in range(M):
        for s in range(S):
            idx_row = m * N + np.arange(N)
            idx_col = s * N + np.arange(N)
            # populate each whole block diagonal at a time
            A_block[idx_row, idx_col] = sim.A[m, s, :]

    # Y_block: stack Y[0,:], Y[1,:], ..., Y[M-1,:]
    Y_block = sim.Y.reshape(-1, 1)  # (M*N, 1)

    return A_block, Y_block, S, N


# ---------------------------------------------------------------------------
# Grouping strategies  (S, N) -> groups array of shape (2*S*N,)
# ---------------------------------------------------------------------------


def frequency_groups(S: int, N: int, **_) -> np.ndarray:
    """
    All N frequency bins of source s -> group s  (S groups of size N).
    Both real and imaginary halves get the same assignment.

    This is the co-occurrence prior: a source is active at all
    frequencies or none.
    """
    groups_half = np.repeat(np.arange(S), N)
    return np.concatenate([groups_half, groups_half])


def no_groups(S: int, N: int, **_) -> np.ndarray:
    """
    Every coefficient gets its own group (equivalent to plain LASSO).
    2*S*N unique group labels.
    """
    return np.arange(2 * S * N)


def real_imag_groups(S: int, N: int, **_) -> np.ndarray:
    """
    Each (Re, Im) pair of the same complex coefficient shares a group.
    S*N groups of size 2.

    This is a straw-man: it couples real and imaginary parts but
    ignores the frequency co-occurrence structure entirely.
    """
    labels = np.arange(S * N)
    return np.concatenate([labels, labels])


def random_groups(S: int, N: int, *, seed: int = 0, **_) -> np.ndarray:
    """
    Randomly assign 2*S*N coefficients to S equal-sized groups.
    Both real and imaginary halves share the same random assignment.

    Shows that arbitrary grouping does not help — only the *right*
    groups (frequency) improve recovery.
    """
    rng = np.random.default_rng(seed)
    groups_half = np.tile(np.arange(S), N)  # S groups, each size N
    rng.shuffle(groups_half)
    return np.concatenate([groups_half, groups_half])


_GROUP_STRATEGIES = {
    "frequency": frequency_groups,
    "none": no_groups,
    "real_imag": real_imag_groups,
    "random": random_groups,
}


# ---------------------------------------------------------------------------
# Public solver
# ---------------------------------------------------------------------------


def group_lasso_solve(
    sim: Simulation,
    alpha: float = 1e-4,
    grouping: str = "frequency",
    max_iter: int = 5000,
    seed: int = 0,
) -> np.ndarray:
    """
    Solve Y = A @ X via Group LASSO with configurable grouping.

    Args:
        sim:      Simulation object (provides A, Y).
        alpha:    Regularisation strength.
        grouping: One of "frequency", "none", "real_imag", "random".
        max_iter: Maximum iterations for the GroupLasso solver.
        seed:     RNG seed (only used when grouping="random").

    Returns:
        X_pred: (S x N) complex array, same shape as sim.X.
    """
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
    group_reg = A_block.shape[0] * alpha

    model = GroupLasso(
        groups=groups,
        group_reg=group_reg,
        n_iter=max_iter,
        supress_warning=True,
    )
    model.fit(A_real, Y_real.ravel())

    # Back to complex block vector, then reshape to (S, N)
    X_block_complex = from_real_augmented(model.coef_.reshape(-1, 1))
    X_pred = X_block_complex.reshape(S, N)

    return X_pred


def group_lasso_custom_solve(
    sim: Simulation,
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
