import sys
from pathlib import Path
from pyparsing import alphas
import scipy.optimize as optimize
import seaborn as sns
from sklearn.linear_model import Lasso

# Add the src directory to the Python path
# This allows us to import modules from cs_priors
script_path = Path(__file__).resolve()
project_root = script_path.parents[1]  # Go up two levels from script to project root
src_path = project_root / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import numpy as np
import matplotlib.pyplot as plt
from cs_priors.simulation.mixing_model import run_simulation
from cs_priors.plotting.plotting import (
    plot_equation,
    plot_two_line_equation,
    plot_overview,
)
from cs_priors.solvers.sparse_prior import (
    to_real_augmented,
    from_real_augmented,
    sparse_prior_solution,
)

def compare_phase_difference(X_true, X_recovered, tol=1e-6):
    """
    Compare the phase difference between the true and recovered signals.
    Only compares phases for elements where both values are above the threshold.
    Returns the mean phase difference in radians for significant values.
    """
    # Only compare phases where both values are above threshold
    mask = (np.abs(X_true) > tol) & (np.abs(X_recovered) > tol)
    
    if not np.any(mask):
        return 0.0  # No significant values to compare
    
    phase_true = np.angle(X_true[mask])
    phase_recovered = np.angle(X_recovered[mask])
    phase_diff = phase_true - phase_recovered
    # Wrap phase difference to [-pi, pi]
    phase_diff = (phase_diff + np.pi) % (2 * np.pi) - np.pi
    mean_phase_diff = np.mean(np.abs(phase_diff))
    return mean_phase_diff