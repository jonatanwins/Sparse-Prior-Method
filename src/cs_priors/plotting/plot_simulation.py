import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import ifft

from ..simulation.Simulation import Simulation
from .plot_geometry import plot_geometry_on_ax
from .plot_signals import plot_signals


def plot_simulation_report(
    sim: Simulation, scale: float = 0.7, pad_factor: float = 0.3
):
    """
    Plot a sequential overview of a simulation:
      1. Geometry (mics + sources)
      2. Source signals x  (S subplots)
      3. Observed signals y = IFFT(Y)  (M subplots)

    Each section is its own figure, suitable for vertical scrolling in a notebook.

    Args:
        sim: Simulation result.
        scale: Multiplier for all figure dimensions (e.g. 0.5 for half-size).
    """

    N = sim.x.shape[1]
    t = np.linspace(0, sim.duration, N, endpoint=False)

    # 1. Geometry
    fig_geo, ax_geo = plt.subplots(figsize=(5 * scale, 5 * scale))
    plot_geometry_on_ax(ax_geo, sim.mics, sim.sources, pad_factor=pad_factor)
    fig_geo.tight_layout()

    # 2. Source signals x
    S = sim.x.shape[0]
    plot_signals(
        sim.x,
        t=t,
        labels=[f"Source {i}" for i in range(S)],
        title="Source signals x",
        figsize_width=10 * scale,
        height_per_signal=2.0 * scale,
    )

    # 3. Observed signals y = IFFT(Y)
    y = ifft(sim.Y, axis=1).real  # expected to be real, rest is noise
    M = y.shape[0]
    plot_signals(
        y,
        t=t,
        labels=[f"Mic {m}" for m in range(M)],
        title="Observed signals y",
        figsize_width=10 * scale,
        height_per_signal=2.0 * scale,
    )

    plt.show()
