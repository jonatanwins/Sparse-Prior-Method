import numpy as np

from cs_priors.solvers.freq_group_lasso import _GROUP_STRATEGIES
from cs_priors.solvers.representations import (
    augmented_real_source_microphone_major_vector_to_complex_matrix,
    complex_matrix_to_augmented_real_matrix,
    complex_matrix_to_augmented_real_source_microphone_major_vector,
    mixing_tensor_to_source_microphone_major_matrix,
    ensure_frequency_system_shapes,
)


try:
    from groupyr import SGL
except ImportError as e:
    SGL = None
    _GROUPYR_IMPORT_ERROR = e
else:
    _GROUPYR_IMPORT_ERROR = None


"""
Groupyr expects the features to be in contingent blocks,
Therefore we have to use source-major representation. 
"""


def _labels_to_groups(labels: np.ndarray) -> list[np.ndarray]:
    return [np.flatnonzero(labels == label) for label in np.unique(labels)]


def _frequency_major_labels_to_source_microphone_major(
    labels: np.ndarray, num_sources: int, num_freqs: int
) -> np.ndarray:
    """
    Reorder augmented-real labels from the existing frequency-major convention
    to the source/microphone-major convention.

    Real and imaginary halves are kept separate; only the ordering within each
    half changes from:
        [f0 all rows, f1 all rows, ...]
    to:
        [row0 all freqs, row1 all freqs, ...]
    """
    labels = np.asarray(labels)
    half = num_sources * num_freqs
    expected_size = 2 * half
    if labels.shape != (expected_size,):
        raise ValueError(
            f"Expected label vector of length {expected_size}, got {labels.shape}"
        )

    perm_half = (
        np.arange(half)
        .reshape((num_sources, num_freqs), order="F")
        .reshape(-1, order="C")
    )
    perm = np.concatenate([perm_half, perm_half + half])
    return labels[perm]


def frequency_sparse_group_lasso_solve(
    Y: np.ndarray,
    A: np.ndarray,
    alpha: float = 1e-4,
    grouping: str = "frequency",
    max_iter: int = 5000,
    seed: int = 0,
    X_start: np.ndarray | None = None,
) -> np.ndarray:
    if SGL is None:
        raise ImportError(
            "frequency_sparse_group_lasso_solve requires groupyr. "
            "Install it with `pip install groupyr`."
        ) from _GROUPYR_IMPORT_ERROR

    A, Y, X_start, _ = ensure_frequency_system_shapes(A, Y, X_start)
    _, num_sources, num_freqs = A.shape

    if X_start is None:
        X_start = np.zeros((num_sources, num_freqs), dtype=complex)
        for f in range(num_freqs):
            X_start[:, f] = np.linalg.lstsq(A[:, :, f], Y[:, f], rcond=None)[0]

    if grouping not in _GROUP_STRATEGIES:
        raise ValueError(
            f"Unknown grouping {grouping!r}. "
            f"Choose from {list(_GROUP_STRATEGIES.keys())}"
        )

    A_big = mixing_tensor_to_source_microphone_major_matrix(A)
    A_real = complex_matrix_to_augmented_real_matrix(A_big)
    Y_real = complex_matrix_to_augmented_real_source_microphone_major_vector(Y).ravel()
    X_start_real = complex_matrix_to_augmented_real_source_microphone_major_vector(
        X_start
    ).ravel()

    labels_frequency_major = _GROUP_STRATEGIES[grouping](
        num_sources, num_freqs, seed=seed
    )
    labels_source_microphone_major = _frequency_major_labels_to_source_microphone_major(
        labels_frequency_major, num_sources, num_freqs
    )
    groups = _labels_to_groups(labels_source_microphone_major)

    model = SGL(
        groups=groups,
        alpha=alpha,
        l1_ratio=0.0,
        fit_intercept=False,
        max_iter=max_iter,
        warm_start=True,
    )
    model.coef_ = X_start_real.copy()
    model.intercept_ = 0.0
    model.fit(A_real, Y_real)

    X_real = np.asarray(model.coef_).reshape(-1, 1)
    return augmented_real_source_microphone_major_vector_to_complex_matrix(
        X_real, num_sources, num_freqs
    )
