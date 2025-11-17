import numpy as np
import matplotlib.pyplot as plt
from .mixing_model import run_simulation
from ..plotting.plot_geometry import plot_mics, plot_sources
from ..plotting.plotting import (
    plot_composite_overview,
    plot_overview,
    plot_equation,
    plot_two_line_equation,
)
from scipy.linalg import null_space

# python -m src.cs_priors.simulation.beamforming_with_priors

# sim = run_simulation(num_mics=1, num_sources=2, s_sparse=1)
# fig, ax = plt.subplots(1)

# plot_mics(ax, sim.mics)
# plot_sources(ax, sim.sources)
# plot_overview(sim)
freq_index = 1
# plot_two_line_equation(
#     sim.Y[:, freq_index],
#     sim.C[:, :, freq_index],
#     sim.X[:, freq_index],
#     ["Y =", "C", "X"],
#     sim.Y[:, freq_index],
#     sim.C_pinv[:, :, freq_index],
#     sim.X_pred[:, freq_index],
#     titles2=["Y", r"$C^{\dagger}$", "= X_pred"],
# )

X = np.array([[0.0], [3.0]])  # 2x1
C = np.array([[1.0, 2.0]])  # 1x2
Y = C @ X  # 1x2 @ 2x1 = 1x1
C_pinv = np.linalg.pinv(C)  # 1x2
X_pred = C_pinv @ Y  # 1x2 @ 1x1 =   1x1
B = null_space(C)  # 2x1
print("B =\n", B / B[1])  # normalize
print("Check: C @ B =", C @ B)


plot_two_line_equation(
    Y,
    C,
    X,
    ("Y = ", "C", "X"),
    Y,
    C_pinv,
    X_pred,
    ("Y", r"$C^{\dagger}$", "X_pred"),
)
