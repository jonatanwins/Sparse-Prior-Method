import numpy as np
from dataclasses import dataclass


@dataclass
class Simulation:
    t: np.ndarray
    composite_waveforms: np.ndarray
    individual_waveforms: dict
    delays_dict: dict
    x_time: np.ndarray
    y_time: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    freqs: np.ndarray
    A: np.ndarray
    A_pinv: np.ndarray
    Y_pred: np.ndarray
    X_pred: np.ndarray
    x_pred: np.ndarray
    sources: list
    mics: list
    walls: list
    sampling_rate: float
    duration: float
    N: int
    active_indices: list
