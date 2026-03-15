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
