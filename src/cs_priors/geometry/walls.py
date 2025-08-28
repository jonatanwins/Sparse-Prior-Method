from dataclasses import dataclass
import numpy as np


@dataclass
class Wall:
    p1: np.ndarray
    p2: np.ndarray
