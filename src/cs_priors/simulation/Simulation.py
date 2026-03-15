import numpy as np
from dataclasses import dataclass


@dataclass
class Simulation:
    """Result of a frequency-domain simulation: Y = A @ X."""

    Y: np.ndarray  # (M x F) observed signals in freq domain
    A: np.ndarray  # (M x S x F) mixing matrix
    X: np.ndarray  # (S x F) source spectra
    X_pinv: np.ndarray  # (S x F) pseudoinverse recovery: pinv(A) @ Y
    x: np.ndarray  # (S x N) source time-domain signals
    freqs: np.ndarray  # (F,) frequency bins
    sources: list  # list of SoundSource
    mics: np.ndarray  # (M x 2) mic positions
    active_indices: list  # which sources are active
    sampling_rate: float
    duration: float
