import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

from itertools import cycle
import numpy as np

colors = cycle(["#FFA343", "#FF1744", "#9A4DFF", "#00BFA5", "#3498DB"])
yellow = "#FFA343"
red = "#FF1744"
purple = "#9A4DFF"
green = "#00BFA5"
blue = "#3498DB"


# ------------------------
# Plotting
# ------------------------


def plot_array_and_sources(x_positions, y_positions, sources):
    """Plot the microphone array and the positions of all sound sources."""
    plt.figure(figsize=(8, 8))
    plt.scatter(x_positions, y_positions, label="Microphones", color=next(colors))
    for idx, source in enumerate(sources):
        source_x, source_y = source.get_position()
        plt.scatter(source_x, source_y, label=f"Source {idx+1}", color=next(colors))
    plt.scatter(0, 0, label="Origin", color=next(colors))
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title("Microphone Array and Sound Sources")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()


def plot_waveforms(t, composite_waveforms, delays_dict):
    """Plot the composite waveforms and delay information for each microphone."""
    num_mics = composite_waveforms.shape[0]

    # Plot composite waveforms
    plt.figure(figsize=(10, 6))
    for mic in range(num_mics):
        plt.plot(t, composite_waveforms[mic], label=f"Mic {mic+1}")
    plt.title("Composite Waveformds at Microphones")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot delays for each source
    for source_label, delays in delays_dict.items():
        plt.figure(figsize=(8, 4))
        plt.stem(range(num_mics), delays)
        plt.title(f"Delays from {source_label} to Each Microphone")
        plt.xlabel("Microphone Index")
        plt.ylabel("Delay (s)")
        plt.grid(True)
        plt.show()


def plot_delays(x_positions, y_positions, sources, t, composite_waveforms, delays_dict):
    """
    Create a single figure that includes:
      1. Microphone array and sound source positions.
      2. Composite waveforms at each microphone.
      3. Delay plots for each sound source.
    """
    n_delay_plots = len(delays_dict)
    total_rows = (
        2 + n_delay_plots
    )  # one row for array geometry, one for waveform, and one per source delay plot

    fig, axes = plt.subplots(total_rows, 1, figsize=(10, 4 * total_rows))

    # Subplot 1: Array and Source Positions
    ax = axes[0]
    ax.scatter(x_positions, y_positions, label="Microphones", color=next(colors))
    for idx, source in enumerate(sources):
        source_x, source_y = source.get_position()
        ax.scatter(source_x, source_y, label=f"Source {idx+1}", color=next(colors))
    ax.scatter(0, 0, label="Origin", color=next(colors))
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_title("Microphone Array and Sound Sources")
    ax.legend()
    ax.grid(True)
    ax.axis("equal")

    # Subplot 2: Composite Waveforms
    ax = axes[1]
    num_mics = composite_waveforms.shape[0]
    for mic in range(num_mics):
        ax.plot(t, composite_waveforms[mic], label=f"Mic {mic+1}")
    ax.set_title("Composite Waveforms at Microphones")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid(True)

    # Subsequent Subplots: Delay Plots for Each Source
    for i, (source_label, delays) in enumerate(delays_dict.items(), start=2):
        ax = axes[i]
        ax.stem(range(num_mics), delays)
        ax.set_title(f"Delays from {source_label} to Each Microphone")
        ax.set_xlabel("Microphone Index")
        ax.set_ylabel("Delay (s)")
        ax.grid(True)

    plt.tight_layout()
    plt.show()


def plot_geometry(ax, x_positions, y_positions, sources):
    """
    Plot microphones, sources, and origin on the given Axes object.
    """
    # Plot microphones
    ax.scatter(
        x_positions,
        y_positions,
        s=60,
        marker="o",
        color="blue",
        edgecolor="k",
        label="Microphones",
    )

    # Plot each source
    for idx, src in enumerate(sources):
        sx, sy = src.get_position()
        ax.scatter(
            sx,
            sy,
            s=80,
            marker="^",
            edgecolor="k",
            label=f"Source {idx+1}",
        )

    # Plot origin
    # ax.scatter(
    #     0,
    #     0,
    #     s=60,
    #     marker="s",
    #     color="black",
    #     edgecolor="k",
    #     label="Origin",
    # )

    ax.set_title("Microphone Array & Sound Sources")
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.axis("equal")
    ax.grid(True)
    ax.legend()

    # ----------------------------
    # 2) Compute auto-limits for x and y
    # ----------------------------
    # Collect all x, all y: mics, sources, origin
    mic_and_src_x = np.concatenate(
        [x_positions, np.array([src.get_position()[0] for src in sources]), [0.0]]
    )
    mic_and_src_y = np.concatenate(
        [y_positions, np.array([src.get_position()[1] for src in sources]), [0.0]]
    )

    x_min, x_max = mic_and_src_x.min(), mic_and_src_x.max()
    y_min, y_max = mic_and_src_y.min(), mic_and_src_y.max()

    # Determine required span to make the plot square
    x_span = x_max - x_min
    y_span = y_max - y_min
    # print(f"{x_max=}, {x_min=}, {y_max=}, {y_min}")
    # print([src.get_position() for src in sources])
    max_span = max(x_span, y_span)

    # Pad by 10% (this is completely hacked atm)
    pad_factor = 0.2
    half_span = 1.5 * max_span * (1 + pad_factor)

    # Final limits
    ax.set_xlim(-half_span, half_span)
    ax.set_ylim(-half_span, half_span)
    # print(f"{half_span=}")

    # quick fix, something weird about xlim on plots straight to axis
    # ax.set_xlim(-5, 5)
    # ax.set_ylim(-5, 5)


def plot_waveform_column(
    fig,
    gs,
    col_index,
    t,
    composite_waveforms,
    individual_waveforms,
    first_mic_colors=None,
):
    """
    Plots the 'composite' waveforms in row 0 and each source’s waveforms
    in subsequent rows, all inside the specified 'col_index' of the GridSpec.

    first_mic_colors: list of colors used to highlight the first mic in each source
    """

    if first_mic_colors is None:
        # Default highlight palette for first mic in each source
        first_mic_colors = ["#FF1744", "#00BFA5", "#FFA343", "#9A4DFF", "#3498DB"]

    # Number of sources determines how many subplot rows we need
    n_sources = len(individual_waveforms)
    # total_rows = 1 + n_sources is handled by whichever function calls this

    # -----------------------------------------------
    # Row 0: Composite waveforms
    # -----------------------------------------------
    ax_composite = fig.add_subplot(gs[0, col_index])

    num_mics = composite_waveforms.shape[0]

    # Plot all but the first mic in light gray
    for mic_idx in range(1, num_mics):
        ax_composite.plot(
            t,
            composite_waveforms[mic_idx],
            color="lightgray",
            label=f"Mic {mic_idx}" if mic_idx == 1 else None,  # minimal legend
        )

    # Now plot the first mic in black
    ax_composite.plot(t, composite_waveforms[0], color="black", label="Mic 1")

    ax_composite.set_title("Composite Waveforms (All Sources Summed)")
    ax_composite.set_xlabel("Time (s)")
    ax_composite.set_ylabel("Amplitude")
    ax_composite.grid(True)
    ax_composite.legend(loc="upper right")

    # -----------------------------------------------
    # Rows 1..n_sources: individual source waveforms
    # -----------------------------------------------
    for i, (src_label, waveforms_for_mics) in enumerate(
        individual_waveforms.items(), start=1
    ):
        ax_source = fig.add_subplot(gs[i, col_index])

        # Pick a highlight color for this source’s first mic
        highlight_color = first_mic_colors[(i - 1) % len(first_mic_colors)]

        # Plot all but first mic in light gray
        for mic_idx in range(1, num_mics):
            ax_source.plot(t, waveforms_for_mics[mic_idx], color="lightgray")

        # The first mic in highlight color
        ax_source.plot(t, waveforms_for_mics[0], color=highlight_color, label="Mic 1")

        ax_source.set_title(f"{src_label} Waveforms")
        ax_source.set_xlabel("Time (s)")
        ax_source.set_ylabel("Amplitude")
        ax_source.grid(True)
        ax_source.legend(loc="upper right")


def plot_overview(
    x_positions,
    y_positions,
    sources,
    t,
    composite_waveforms,
    individual_waveforms,
):
    """
    Create a single figure with:
      - Left column (spans all rows): Microphone array & source positions
      - Right column:
         Row 0: Composite waveforms (all sources summed)
         Rows 1..N: Each source's waveforms in its own subplot
    Highlight only the first mic in each waveform plot with a distinct color,
    and make all other mic waveforms light gray.
    """

    # We'll have N+1 rows on the right: 1 for composite + N for individual sources
    n_sources = len(sources)
    total_rows = n_sources + 1  # 1 composite row + 1 row per source

    fig = plt.figure(figsize=(14, 2.5 * total_rows))
    gs = fig.add_gridspec(
        total_rows, 2, width_ratios=[1, 1], height_ratios=[1] * total_rows
    )

    # -------------------------------------------------------------------------
    # 1) Left column: geometry (spans all rows in the first column)
    # -------------------------------------------------------------------------
    ax_geometry = fig.add_subplot(gs[:, 0])
    plot_geometry(ax_geometry, x_positions, y_positions, sources)

    # -------------------------------------------------------------------------
    # 2) Right column, row 0: Composite waveforms
    #    Highlight Mic #1 in black, others in light gray
    # -------------------------------------------------------------------------
    ax_composite = fig.add_subplot(gs[0, 1])

    num_mics = composite_waveforms.shape[0]
    # The rest in light gray
    for mic_idx in range(1, num_mics):
        ax_composite.plot(
            t,
            composite_waveforms[mic_idx],
            color="lightgray",
            # label=f"Mic {mic_idx}",
            # consider no label to avoid legend clutter
        )
    # First mic in black
    ax_composite.plot(t, composite_waveforms[0], color="black", label="Mic 1")

    ax_composite.set_title("Composite Waveforms (All Sources Summed)")
    ax_composite.set_xlabel("Time (s)")
    ax_composite.set_ylabel("Amplitude")
    ax_composite.grid(True)
    ax_composite.legend(loc="upper right")

    # -------------------------------------------------------------------------
    # 3) Right column, rows 1..n_sources: individual source waveforms
    #    Different highlight color for each source's first mic,
    #    all other mics in light gray
    # -------------------------------------------------------------------------
    # Define a small palette of highlight colors for each source's first mic
    first_mic_colors = ["#FF1744", "#00BFA5", "#FFA343", "#9A4DFF", "#3498DB"]
    # If there are more sources than colors, it will cycle

    # Enumerate each source's waveform dictionary entry:
    #   "Source 1" -> waveforms_for_mics, etc.
    for i, (src_label, waveforms_for_mics) in enumerate(
        individual_waveforms.items(), start=1
    ):
        ax_source = fig.add_subplot(gs[i, 1])

        # Pick a highlight color for this source
        highlight_color = first_mic_colors[(i - 1) % len(first_mic_colors)]

        # Other mics in gray
        for mic_idx in range(1, num_mics):
            ax_source.plot(t, waveforms_for_mics[mic_idx], color="lightgray")
        # Mic #0 in highlight color
        ax_source.plot(t, waveforms_for_mics[0], color=highlight_color, label="Mic 1")

        ax_source.set_title(f"Source {src_label} Waveforms")
        ax_source.set_xlabel("Time (s)")
        ax_source.set_ylabel("Amplitude")
        ax_source.grid(True)
        ax_source.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


def plot_Y_comparison(Y_f0, Y_pred_f0):
    indices = np.arange(len(Y_f0))

    # Determine colors based on the sign for real and imaginary parts.
    colors_real_Y = [green if val >= 0 else red for val in Y_f0.real]
    colors_real_Ypred = [green if val >= 0 else red for val in Y_pred_f0.real]
    colors_imag_Y = [yellow if val >= 0 else purple for val in Y_f0.imag]
    colors_imag_Ypred = [yellow if val >= 0 else purple for val in Y_pred_f0.imag]

    # Create a 2x2 subplot: lef column for real parts, right column for imaginary parts.
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Plot real parts of Y_f0.
    axs[0, 0].bar(indices, Y_f0.real, color=colors_real_Y)
    axs[0, 0].set_title(r"$Y_{f_0}$ Real")
    axs[0, 0].set_xticks(indices)

    # Plot real parts of Y_pred_f0.
    axs[1, 0].bar(indices, Y_pred_f0.real, color=colors_real_Ypred)
    axs[1, 0].set_title(r"$Y^{pred}_{f_0}$ Real")
    axs[1, 0].set_xticks(indices)

    # Plot imaginary parts of Y_f0.
    axs[0, 1].bar(indices, Y_f0.imag, color=colors_imag_Y)
    axs[0, 1].set_title(r"$Y_{f_0}$ Imag")
    axs[0, 1].set_xticks(indices)

    # Plot imaginary parts of Y_pred_f0.
    axs[1, 1].bar(indices, Y_pred_f0.imag, color=colors_imag_Ypred)
    axs[1, 1].set_title(r"$Y^{pred}_{f_0}$ Imag")
    axs[1, 1].set_xticks(indices)

    # Annotate bars with values (rounded to 2 decimals)
    for ax in axs.flat:
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f")

    fig.suptitle("Comparison of Fourier Coefficients at f0", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def complex_to_rgb(matrix):
    """
    Convert a complex matrix to an RGB image using an HSV mapping:
      - Hue: phase (normalized from -pi to pi → 0 to 1)
      - Saturation: fixed at 1.
      - Value: magnitude (normalized to 0-1).
    """
    # Ensure matrix is two-dimensional.
    mat = np.atleast_2d(matrix)

    # Compute phase and magnitude.
    phase = np.angle(mat)  # Range: [-pi, pi]
    magnitude = np.abs(mat)

    # Normalize phase to [0,1].
    norm_phase = (phase + np.pi) / (2 * np.pi)
    # Normalize magnitude to [0,1] using its maximum.
    max_mag = magnitude.max() if magnitude.max() != 0 else 1
    norm_mag = magnitude / max_mag

    # Build an HSV image.
    hsv = np.zeros(mat.shape + (3,))
    hsv[..., 0] = norm_phase  # Hue from phase.
    hsv[..., 1] = 1  # Full saturation.
    hsv[..., 2] = norm_mag  # Value from normalized magnitude.

    # Convert HSV to RGB.
    rgb = hsv_to_rgb(hsv)
    return rgb


def plot_complex_matrix_on_ax(ax, matrix, title="", show_values=True):
    """
    Plot a complex matrix on the provided Axes object using an HSV mapping.

    The cell color is determined by:
      - Hue: phase of the complex number.
      - Value: magnitude.

    Optionally, the numerical complex value is overlaid.
    """
    # Convert the complex matrix to an RGB image.
    rgb = complex_to_rgb(matrix)
    ax.imshow(rgb, interpolation="none", aspect="auto")
    ax.set_title(title)
    ax.axis("off")

    # Expand if single column vector
    # not sure if this causes problems
    if np.ndim(matrix) < 2:
        matrix = np.expand_dims(matrix, axis=0)

    if show_values:
        n, m = matrix.shape
        for i in range(n):
            for j in range(m):
                # Format the complex number with 2 decimal precision.
                # If the imaginary part is negative, the sign will appear automatically.
                value_str = f"{matrix[i, j].real:.1f}{matrix[i, j].imag:+.1f}j"
                ax.text(
                    j,
                    i,
                    value_str,
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=10,
                )


def plot_complex_matrix(matrix, title="Complex Matrix", show_values=True):
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_complex_matrix_on_ax(ax, matrix, title=title, show_values=show_values)
    plt.show()


def plot_equation(Y, C, X, titles=("Y", "C", "X"), show_values=True):
    """
    Plot the matrices Y, C, and X side by side as if in the equation:
         Y = C × X

    Each matrix is plotted using an HSV-based visualization with overlaid values.
    If Y or X have fewer than 2 dimensions, they are automatically expanded.
    The equation symbols are positioned using the axes' positions to avoid overlapping.
    """
    # Automatically expand dimensions if necessary.
    # if np.ndim(Y) < 2:
    #     Y = np.expand_dims(Y, axis=1)
    # if np.ndim(X) < 2:
    #     X = np.expand_dims(X, axis=1)

    # Use a gridspec with some horizontal space between axes.
    fig, axs = plt.subplots(
        1, 3, figsize=(15, 5), gridspec_kw={"width_ratios": [1, 1, 1], "wspace": 0.1}
    )

    # Plot each matrix on its own axis.
    plot_complex_matrix_on_ax(axs[0], Y, title=titles[0], show_values=show_values)
    plot_complex_matrix_on_ax(axs[1], C, title=titles[1], show_values=show_values)
    plot_complex_matrix_on_ax(axs[2], X, title=titles[2], show_values=show_values)

    # Get the positions of the axes in figure coordinates.
    pos0 = axs[0].get_position()  # Position of first subplot
    pos1 = axs[1].get_position()  # Position of second subplot
    pos2 = axs[2].get_position()  # Position of third subplot

    # Compute x positions for the symbols:
    # Place "=" halfway between the right edge of the first and the left edge of the second.
    eq_x = (pos0.x1 + pos1.x0) / 2.0
    # Place "×" halfway between the right edge of the second and the left edge of the third.
    mult_x = (pos1.x1 + pos2.x0) / 2.0
    # Use the vertical center of the middle axis.
    eq_y = (pos1.y0 + pos1.y1) / 2.0

    # Add the equation symbols using figure coordinates.
    fig.text(eq_x, eq_y, "=", fontsize=30, ha="center", va="center")
    fig.text(mult_x, eq_y, "×", fontsize=30, ha="center", va="center")

    fig.suptitle(f"Equation: {titles[0]} = {titles[1]}{titles[2]}", fontsize=32)
    # plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.show()
