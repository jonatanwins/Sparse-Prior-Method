def plot_sources(ax, sources, show_frequency=True, source_color="dodgerblue"):
    # Plot each source
    for idx, src in enumerate(sources):
        sx, sy = src.get_position()
        ax.scatter(
            sx,
            sy,
            s=50,
            marker="^",
            edgecolor="k",
            color=source_color,
            label="Sources" if idx == 0 else None,
        )
        if show_frequency:

            # Construct the annotation string with frequency and (if applicable) phase
            label = f"{src.frequency:.0f} Hz" + (
                f"\n{src.phase:.2f} rad" if src.phase != 0.0 else ""
            )

            # Annotate at the desired offset
            ax.text(
                sx + 0.05,
                sy + 0.05,
                label,
                fontsize=9,
                color=source_color,
            )


def plot_mics(ax, mics, mic_color="crimson"):
    # Plot microphones
    ax.scatter(
        mics.x,
        mics.y,
        s=30,
        marker="o",
        color=mic_color,
        edgecolor="k",
        label="Microphones",
    )

    # Annotate each microphone with its microphone number (starting from 0)
    for idx, (x, y) in enumerate(zip(mics.x, mics.y)):
        ax.text(x - 0.01, y + 0.05, f"{idx}", fontsize=9, color=mic_color)


def plot_walls(ax, walls, wall_color="gray"):
    """Plot the walls on the given Axes object."""
    for wall in walls:
        ax.plot([wall.p1[0], wall.p2[0]], [wall.p1[1], wall.p2[1]], color=wall_color)


def plot_geometry_on_ax(
    ax,
    mics,
    sources,
    walls=[],
    show_frequency=True,
    mic_color="crimson",
    source_color="dodgerblue",
):
    """
    Plot microphones, sources, and origin on the given Axes object.
    """

    plot_mics(ax, mics, mic_color=mic_color)
    plot_sources(ax, sources, show_frequency=show_frequency, source_color=source_color)
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
        [mics.x, np.array([src.get_position()[0] for src in sources]), [0.0]]
    )
    mic_and_src_y = np.concatenate(
        [mics.y, np.array([src.get_position()[1] for src in sources]), [0.0]]
    )

    x_min, x_max = mic_and_src_x.min(), mic_and_src_x.max()
    y_min, y_max = mic_and_src_y.min(), mic_and_src_y.max()

    # Determine required span to make the plot square
    x_span = x_max - x_min
    y_span = y_max - y_min
    max_span = max(x_span, y_span)

    # Pad by 30%
    pad_factor = 0.8

    lower_bound = min(x_min, y_min) - pad_factor * max_span
    upper_bound = max(x_max, y_max) + pad_factor * max_span
    ax.set(xlim=(lower_bound, upper_bound), ylim=(lower_bound, upper_bound))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    from ..simulation.mixing_model import run_simulation
    from ..geometry.walls import Wall

    sim = run_simulation(no_sources=1, num_mics=1)

    fig, ax = plt.subplots()
    plot_geometry_on_ax(
        ax,
        sim.mics,
        sim.sources,
        walls=[
            Wall(p1=np.array([-0.3, -0.5]), p2=np.array([-0.3, 1.7])),
            Wall(p1=np.array([-0.3, 1.7]), p2=np.array([1.5, 1.7])),
        ],
    )
    plt.show()
