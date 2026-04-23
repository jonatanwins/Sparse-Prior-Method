"""
Sparse-recovery metrics for comparing X_true vs X_pred.

Source-level detection: a source is "active" when its energy across all
frequency bins exceeds a threshold (‖X[i,:]‖₂ > tol).

Element-level helpers (active_elements) are kept for debugging.
"""

import numpy as np
from ..simulation.Simulation import Simulation


# ---------------------------------------------------------------------------
# Threshold helpers
# ---------------------------------------------------------------------------


def noise_threshold(X: np.ndarray, factor: float = 0.1) -> float:
    """Threshold = factor * max per-source energy.

    Computes ‖X[i,:]‖₂ for each source row, returns factor * max.
    For 1-D input, falls back to factor * ‖X‖₂.
    """
    if X.ndim >= 2:
        energies = np.linalg.norm(X, axis=1)
    else:
        energies = np.array([np.linalg.norm(X)])
    return float(factor * energies.max())


# ---------------------------------------------------------------------------
# Source-level detection (primary API)
# ---------------------------------------------------------------------------


def source_energies(X: np.ndarray) -> np.ndarray:
    """Per-source energy: ‖X[i,:]‖₂ for each row.  Returns (S,) array."""
    if X.ndim < 2:
        return np.array([np.linalg.norm(X)])
    return np.linalg.norm(X, axis=1)


def active_sources(X: np.ndarray, tol: float) -> np.ndarray:
    """Source indices where ‖X[i,:]‖₂ > tol."""
    return np.flatnonzero(source_energies(X) > tol)


# ---------------------------------------------------------------------------
# Element-level detection (debugging)
# ---------------------------------------------------------------------------


def active_elements(X: np.ndarray, tol: float) -> np.ndarray:
    """1-D array of flat indices where |X| > tol."""
    return np.flatnonzero(np.abs(X) > tol)


# ---------------------------------------------------------------------------
# Detection metrics (source-level)
# ---------------------------------------------------------------------------


def detection_scores(
    X_true: np.ndarray,
    X_pred: np.ndarray,
    tol: float | None = None,
) -> dict[str, float]:
    """
    Compare which *sources* are detected as active.

    A source i is active when ‖X[i,:]‖₂ > tol.

    Parameters
    ----------
    X_true, X_pred : (S x N) arrays
    tol : float, optional
        If None, uses ``noise_threshold(X_true)``.

    Returns
    -------
    dict with keys: tp, fp, fn, precision, recall, f1
    """
    if tol is None:
        tol = noise_threshold(X_true)

    true_set = set(active_sources(X_true, tol).tolist())
    pred_set = set(active_sources(X_pred, tol).tolist())

    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1)


# ---------------------------------------------------------------------------
# Support-leakage metrics
# ---------------------------------------------------------------------------


def source_leakage_metrics(
    X_true: np.ndarray,
    X_pred: np.ndarray,
    tol: float | None = None,
    row_floor_ratio: float = 0.1,
) -> dict[str, float]:
    """
    Quantify how much predicted energy leaks into source rows that should be inactive.

    The true support is taken from ``X_true`` using ``active_sources``. The returned
    metrics are energy-based, so they can distinguish "clean lines on the correct
    rows" from solutions that spread mass across inactive rows.

    Parameters
    ----------
    X_true, X_pred : (S x N) arrays
        True and predicted source spectra.
    tol : float, optional
        Threshold used to decide which rows in ``X_true`` are active. If omitted,
        uses ``noise_threshold(X_true)``.
    row_floor_ratio : float, default 0.1
        An inactive row counts as visibly active when its row norm exceeds this
        fraction of the strongest true active row norm.

    Returns
    -------
    dict with keys:
        active_energy_ratio, inactive_energy_ratio,
        inactive_to_active_energy_ratio,
        inactive_rows_above_floor, inactive_row_fraction,
        max_inactive_row_ratio, row_floor_ratio
    plus the source-detection scores and relative_error.
    """
    if not 0.0 <= row_floor_ratio <= 1.0:
        raise ValueError("row_floor_ratio must lie in [0, 1]")

    X_true_2d = np.atleast_2d(np.asarray(X_true))
    X_pred_2d = np.atleast_2d(np.asarray(X_pred))
    if X_true_2d.shape != X_pred_2d.shape:
        raise ValueError(
            "X_true and X_pred must have the same shape after conversion to (S, N)"
        )

    if tol is None:
        tol = noise_threshold(X_true_2d)

    active = active_sources(X_true_2d, tol)
    inactive = np.setdiff1d(np.arange(X_true_2d.shape[0]), active, assume_unique=True)

    row_true = source_energies(X_true_2d)
    row_pred = source_energies(X_pred_2d)
    strongest_true_active = float(row_true[active].max()) if active.size else 0.0

    total_energy = float(np.linalg.norm(X_pred_2d, ord="fro") ** 2)
    active_energy = (
        float(np.linalg.norm(X_pred_2d[active], ord="fro") ** 2) if active.size else 0.0
    )
    inactive_energy = (
        float(np.linalg.norm(X_pred_2d[inactive], ord="fro") ** 2)
        if inactive.size
        else 0.0
    )

    active_energy_ratio = active_energy / total_energy if total_energy > 0 else 0.0
    inactive_energy_ratio = (
        inactive_energy / total_energy if total_energy > 0 else 0.0
    )

    if active_energy > 0:
        inactive_to_active_energy_ratio = inactive_energy / active_energy
    elif inactive_energy > 0:
        inactive_to_active_energy_ratio = float("inf")
    else:
        inactive_to_active_energy_ratio = 0.0

    if inactive.size and strongest_true_active > 0:
        inactive_rows_above_floor = int(
            np.sum(row_pred[inactive] > row_floor_ratio * strongest_true_active)
        )
        max_inactive_row_ratio = float(row_pred[inactive].max() / strongest_true_active)
    else:
        inactive_rows_above_floor = 0
        max_inactive_row_ratio = 0.0

    detection = detection_scores(X_true_2d, X_pred_2d, tol=tol)
    return {
        **detection,
        "relative_error": relative_error(X_true_2d, X_pred_2d),
        "active_energy_ratio": active_energy_ratio,
        "inactive_energy_ratio": inactive_energy_ratio,
        "inactive_to_active_energy_ratio": inactive_to_active_energy_ratio,
        "inactive_rows_above_floor": inactive_rows_above_floor,
        "inactive_row_fraction": (
            inactive_rows_above_floor / inactive.size if inactive.size else 0.0
        ),
        "max_inactive_row_ratio": max_inactive_row_ratio,
        "row_floor_ratio": row_floor_ratio,
    }


def format_source_leakage_summary(
    metrics: dict[str, float], digits: int = 1
) -> str:
    """
    Format source-leakage metrics for compact display in plot titles.

    The second line reports how many inactive rows rise above the configured floor
    and how large the worst inactive row is relative to the strongest true row.
    """
    floor_pct = 100.0 * metrics["row_floor_ratio"]
    on_pct = 100.0 * metrics["active_energy_ratio"]
    off_pct = 100.0 * metrics["inactive_energy_ratio"]
    peak_off_pct = 100.0 * metrics["max_inactive_row_ratio"]
    inactive_rows = int(metrics["inactive_rows_above_floor"])

    return (
        f"on={on_pct:.{digits}f}%, off={off_pct:.{digits}f}%\n"
        f"off-rows>{floor_pct:.0f}%={inactive_rows}, max off/true={peak_off_pct:.{digits}f}%"
    )


# ---------------------------------------------------------------------------
# Reconstruction error
# ---------------------------------------------------------------------------


def relative_error(X_true: np.ndarray, X_pred: np.ndarray) -> float:
    """||X_true - X_pred|| / ||X_true||.  Returns inf if X_true is zero."""
    norm_true = np.linalg.norm(X_true)
    if norm_true == 0:
        return float("inf")
    return float(np.linalg.norm(X_true - X_pred) / norm_true)


# ---------------------------------------------------------------------------
# One-call summary
# ---------------------------------------------------------------------------


def score(
    sim: Simulation,
    X_pred: np.ndarray,
    tol: float | None = None,
) -> dict[str, float]:
    """
    Full scorecard for a single prediction against sim.X.

    Returns dict with: tp, fp, fn, precision, recall, f1, relative_error
    """
    scores = detection_scores(sim.X, X_pred, tol=tol)
    scores["relative_error"] = relative_error(sim.X, X_pred)
    return scores
