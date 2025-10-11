import numpy as np
import matplotlib.pyplot as plt
from .mixing_model import run_simulation
from ..plotting.plot_geometry import plot_mics, plot_sources
from ..plotting.plotting import plot_composite_overview, plot_overview, plot_equation

# python -m src.cs_priors.simulation.beamforming_with_priors

sim = run_simulation(num_mics=3, num_sources=8, s_sparse=5)
# fig, ax = plt.subplots(1)

# plot_mics(ax, sim.mics)
# plot_sources(ax, sim.sources)
# plot_overview(sim)
plot_equation(
    sim.X_pred[1],
    sim.C_pinv[:, :, 1],
    sim.Y[1],
    ratios=(1, 10, 1),
    titles=["X_pred", "C", "Y"],
)

# plt.show()
