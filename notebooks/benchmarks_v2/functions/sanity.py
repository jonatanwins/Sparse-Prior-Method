import matplotlib.pyplot as plt
import numpy as np

from cs_priors.plotting.plot_complex import plot_matrices
from cs_priors.plotting.plot_geometry import plot_sim_geometry
from figures import save_pdf
from single_frequency_benchmarks import METHOD_LABELS


def _auto_view_limits(sim):
    source_positions = np.array([src.get_position() for src in sim.sources])
    points = np.vstack([sim.mics, source_positions, np.zeros((1, 2))])

    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    x_pad = 0.05 * (x_max - x_min)
    y_pad = 0.10 * (y_max - y_min)

    return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)


def run_sanity_check(
    sim,
    *,
    solutions,
    save_path,
    view_limits=None,
    grid_step=None,
    unit="cm",
    dpi=90,
):
    view_limits = _auto_view_limits(sim) if view_limits is None else view_limits
    fig, ax = plot_sim_geometry(
        sim,
        dpi=dpi,
        pad_factor=0.1,
        show=False,
        unit=unit,
        view_limits=view_limits,
        grid_step=grid_step,
    )
    save_pdf(fig, save_path)
    plt.show()

    plot_matrices(
        [sim.X, *solutions.values()],
        [r"$\boldsymbol{x}$ true", *[METHOD_LABELS.get(k, k) for k in solutions]],
        dpi=60,
        show_values=True,
    )
