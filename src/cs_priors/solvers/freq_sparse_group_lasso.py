import numpy as np

from cs_priors.solvers.freq_group_lasso import _GROUP_STRATEGIES
from cs_priors.solvers.representations import (
    augmented_real_vector_to_complex_matrix,
    complex_matrix_to_augmented_real_matrix,
    complex_matrix_to_augmented_real_vector,
    mixing_tensor_to_frequency_major_matrix,
    ensure_frequency_system_shapes,
)

try:
    from groupyr import SGL
except ImportError as e:
    SGL = None
    _GROUPYR_IMPORT_ERROR = e
else:
    _GROUPYR_IMPORT_ERROR = None


def _labels_to_groups(labels: np.ndarray) -> list[np.ndarray]:
    return [np.flatnonzero(labels == label) for label in np.unique(labels)]


def frequency_sparse_group_lasso_solve(
    Y: np.ndarray,
    A: np.ndarray,
    alpha: float = 1e-4,
    grouping: str = "frequency",
    max_iter: int = 5000,
    seed: int = 0,
) -> np.ndarray:
    if SGL is None:
        raise ImportError(
            "frequency_sparse_group_lasso_solve requires groupyr. "
            "Install it with `pip install groupyr`."
        ) from _GROUPYR_IMPORT_ERROR

    A, Y, _, _ = ensure_frequency_system_shapes(A, Y)
    _, num_sources, num_freqs = A.shape

    if grouping not in _GROUP_STRATEGIES:
        raise ValueError(
            f"Unknown grouping {grouping!r}. "
            f"Choose from {list(_GROUP_STRATEGIES.keys())}"
        )

    A_big = mixing_tensor_to_frequency_major_matrix(A)
    A_real = complex_matrix_to_augmented_real_matrix(A_big)
    Y_real = complex_matrix_to_augmented_real_vector(Y).ravel()
    labels = _GROUP_STRATEGIES[grouping](num_sources, num_freqs, seed=seed)
    groups = _labels_to_groups(labels)

    model = SGL(
        groups=groups,
        alpha=alpha,
        l1_ratio=0.0,
        fit_intercept=False,
        max_iter=max_iter,
    )
    model.fit(A_real, Y_real)

    X_real = np.asarray(model.coef_).reshape(-1, 1)
    return augmented_real_vector_to_complex_matrix(X_real, num_sources, num_freqs)
