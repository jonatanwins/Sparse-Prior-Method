from typeguard import typechecked
import numpy as np


@typechecked
def plot_sources(ax, sources, source_color: str = "dodgerblue"):
    # Plot each source
    for idx, src in enumerate(sources):
        sx, sy = src.get_position()
        ax.scatter(
            sx,
            sy,
            s=40,
            marker="^",
            edgecolor="k",
            color=source_color,
            label="Sources" if idx == 0 else None,
        )
        if src.label is not None:
            # Annotate at the desired offset
            ax.text(
                sx + 0.05,
                sy + 0.05,
                src.label,
                fontsize=8,
                color=source_color,
            )


@typechecked
def plot_mics(ax, mics, mic_color="crimson", annotate=False):
    # Plot microphones
    ax.scatter(
        mics[:, 0],
        mics[:, 1],
        s=20,
        marker="o",
        color=mic_color,
        edgecolor="k",
        label="Microphones",
    )
    if annotate:
        # Annotate each microphone with its microphone number (starting from 0)
        for idx, (x, y) in enumerate(zip(mics[:, 0], mics[:, 1])):
            ax.text(x - 0.01, y + 0.05, f"{idx}", fontsize=8, color=mic_color)


def plot_walls(ax, walls, wall_color="gray"):
    """Plot the walls on the given Axes object."""
    for wall in walls:
        ax.plot([wall.p1[0], wall.p2[0]], [wall.p1[1], wall.p2[1]], color=wall_color)


@typechecked
def plot_geometry_on_ax(
    ax,
    mics: np.ndarray,
    sources,
    walls: list | None = None,
    mic_color: str = "crimson",
    source_color: str = "dodgerblue",
    pad_factor: float = 0.3,
):
    """
    Plot microphones, sources, and origin on the given Axes object.
    """

    plot_mics(ax, mics, mic_color=mic_color)
    plot_sources(ax, sources, source_color=source_color)
    if walls is not None:
        plot_walls(ax, walls, wall_color="gray")

    ax.set_title("Microphone Array & Sound Sources")
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    # ax.axis("equal")  # This causes problems with xlim and ylim
    ax.grid(True)
    ax.legend()

    # ----------------------------
    # 2) Compute auto-limits for x and y
    # ----------------------------
    # Collect all x, all y: mics, sources, origin
    mic_and_src_x = np.concatenate(
        [mics[:, 0], np.array([src.get_position()[0] for src in sources]), [0.0]]
    )
    mic_and_src_y = np.concatenate(
        [mics[:, 1], np.array([src.get_position()[1] for src in sources]), [0.0]]
    )

    x_min, x_max = mic_and_src_x.min(), mic_and_src_x.max()
    y_min, y_max = mic_and_src_y.min(), mic_and_src_y.max()

    # Determine required span to make the plot square
    x_span = x_max - x_min
    y_span = y_max - y_min
    max_span = max(x_span, y_span)

    # Pad by pad_factor * 100%
    lower_bound = min(x_min, y_min) - pad_factor * max_span
    upper_bound = max(x_max, y_max) + pad_factor * max_span
    ax.set(xlim=(lower_bound, upper_bound), ylim=(lower_bound, upper_bound))


def plot_sim_geometry(sim, dpi: int = 70, pad_factor: float = 0.2, show: bool = True):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(dpi=dpi)
    plot_geometry_on_ax(ax, sim.mics, sim.sources, pad_factor=pad_factor)

    for k, i in enumerate(sim.active_indices):
        sx, sy = sim.sources[i].get_position()
        ax.scatter(
            sx,
            sy,
            s=80,
            marker="^",
            color="gold",
            edgecolor="k",
            label="Active sources" if k == 0 else None,
        )

    ax.legend()
    if show:
        plt.show()
    else:
        return fig, ax


# ------ Functions related to walls ----------------------------------------
# TODO: These are not in use yet (5.mar 2026)
from ..geometry.walls import compute_reflections


def plot_reflected_sources(ax, walls, sources, color="orange"):
    """Plot the reflected sources across the given walls."""

    reflections = compute_reflections(sources, walls)
    for reflection in reflections:
        ax.scatter(
            reflection.pos[0],
            reflection.pos[1],
            s=50,
            marker="^",
            edgecolor="k",
            color=color,
        )
    return reflections


def plot_intersections(ax, walls, mics, reflections):
    """Plot the intersections of the reflections with the walls."""
    for mic in mics:
        paths = compute_path(reflections, mic)
        for path in paths:
            for intersection in path.intersection_seq:
                ax.scatter(
                    intersection[0],
                    intersection[1],
                    s=50,
                    marker="x",
                    color="purple",
                )


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
