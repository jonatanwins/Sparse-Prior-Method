from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb

colors = cycle(["#FFA343", "#FF1744", "#9A4DFF", "#00BFA5", "#3498DB"])
yellow = "#FFA343"
red = "#FF1744"
purple = "#9A4DFF"
green = "#00BFA5"
blue = "#3498DB"


# ------------------------
# Plotting
# ------------------------


def plot_waveforms(t, composite_waveforms, delays_dict):
    """Plot the composite waveforms and delay information for each microphone."""
    num_mics = composite_waveforms.shape[0]

    # Plot composite waveforms
    plt.figure(figsize=(10, 6))
    for mic in range(num_mics):
        plt.plot(t, composite_waveforms[mic], label=f"Mic {mic}")
    plt.title("Composite Waveforms at Microphones")
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
        ax.scatter(source_x, source_y, label=f"Source {idx}", color=next(colors))
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
        ax.plot(t, composite_waveforms[mic], label=f"Mic {mic}")
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


def plot_geometry_on_ax(
    ax,
    x_positions,
    y_positions,
    sources,
    show_frequency=True,
    mic_color="crimson",
    source_color="dodgerblue",
    pad_factor=0.3,
    skip_labels=False,
):
    """
    Plot microphones, sources, and origin on the given Axes object.
    """

    # Plot microphones
    ax.scatter(
        x_positions,
        y_positions,
        s=30,
        marker="o",
        color=mic_color,
        edgecolor="k",
        label="Microphones",
    )

    # Annotate each microphone with its microphone number (starting from 0)
    if not skip_labels:
        for idx, (x, y) in enumerate(zip(x_positions, y_positions)):
            ax.text(x - 0.01, y + 0.05, f"{idx}", fontsize=9, color=mic_color)
    else:
        show_frequency = False  # if skipping labels, also skip frequency annotations

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
    max_span = max(x_span, y_span)

    # Here we use the specified pad_factor to add padding around the plot
    lower_bound = min(x_min, y_min) - pad_factor * max_span
    upper_bound = max(x_max, y_max) + pad_factor * max_span
    ax.set(xlim=(lower_bound, upper_bound), ylim=(lower_bound, upper_bound))


def plot_geometry_auto(
    x_positions,
    y_positions,
    sources,
    show_frequency=True,
    figsize=(8, 8),
    fontsize=16,
    pad_factor=0.4,
    skip_labels=False,
):

    fig, ax = plt.subplots(figsize=figsize)
    # Call your existing plot_geometry function
    plot_geometry_on_ax(
        ax,
        x_positions,
        y_positions,
        sources,
        show_frequency,
        pad_factor=pad_factor,
        skip_labels=skip_labels,
    )
    plt.axis("equal")
    plt.setp(ax.get_xticklabels(), fontsize=fontsize)
    plt.setp(ax.get_yticklabels(), fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.title("Microphone Array & Sound Sources", fontsize=fontsize + 2)
    plt.xlabel("X Position (m)", fontsize=fontsize)
    plt.ylabel("Y Position (m)", fontsize=fontsize)
    plt.show()


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
    ax_composite.plot(t, composite_waveforms[0], color="black", label="Mic 0")

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
        ax_source.plot(t, waveforms_for_mics[0], color=highlight_color, label="Mic 0")

        ax_source.set_title(f"{src_label} Waveforms")
        ax_source.set_xlabel("Time (s)")
        ax_source.set_ylabel("Amplitude")
        ax_source.grid(True)
        ax_source.legend(loc="upper right")


def plot_composite_overview(
    t, composite_waveforms, figsize=(10, 6), sample_color="crimson"
):
    """
    Create a figure that displays the composite waveforms (all sources summed) at the microphones.

    Parameters:
        t (array-like): Time axis values.
        composite_waveforms (np.array): 2D array (num_mics x len(t)) representing the composite signals
                                        at each microphone.
    """
    num_mics = composite_waveforms.shape[0]
    fig, ax = plt.subplots(figsize=figsize)

    # Plot all microphones except the first in light gray.
    for mic_idx in range(1, num_mics):
        ax.plot(
            t,
            composite_waveforms[mic_idx],
            color="lightgray",
            label=f"Other Microphones" if mic_idx == 1 else None,
        )  # Only add label once.

    # Plot the first microphone in black and highlight its points in gold.
    ax.plot(t, composite_waveforms[0], color="black", label="Mic 0")
    ax.scatter(t, composite_waveforms[0], color=sample_color, zorder=3)

    ax.set_title("Composite Waveforms (All Sources Summed)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_waveform_column_auto(
    t, composite_waveforms, individual_waveforms, width=8, height_per_cell=2.5
):
    """
    Create a figure that plots only the waveform plots:

      - Row 0: Composite waveforms (all sources summed)
      - Rows 1..N: Each individual source's waveforms (each in its own subplot)

    For every waveform plot:
      - The first microphone (mic #0) is highlighted (in black for composite and in a unique color for each source).
      - All other microphone waveforms are plotted in light gray.

    Parameters:
      t : array-like
          Time vector for the x-axis.
      composite_waveforms : numpy.ndarray
          2D array of shape (num_mics, len(t)). Represents the composite waveform data.
      individual_waveforms : dict
          Dictionary with keys representing source labels and values being 2D arrays (one per source)
          of shape (num_mics, len(t)) that contain the waveform data from each microphone.
    """
    # Determine the number of sources and number of rows (1 composite + 1 per source)
    n_sources = len(individual_waveforms)
    total_rows = n_sources + 1

    # Create a figure with a grid of subplots (one column only)
    fig = plt.figure(figsize=(width, height_per_cell * total_rows))
    gs = fig.add_gridspec(total_rows, 1, height_ratios=[1] * total_rows)

    # ------------------------------
    # 1) Composite Waveform (Row 0)
    # ------------------------------
    num_mics = composite_waveforms.shape[0]
    ax_composite = fig.add_subplot(gs[0, 0])

    # Plot all mics except the first in light gray
    for mic_idx in range(1, num_mics):
        ax_composite.plot(t, composite_waveforms[mic_idx], color="lightgray")

    # Plot the first mic in black with highlighted markers
    ax_composite.plot(t, composite_waveforms[0], color="black", label="Mic 0")
    ax_composite.scatter(t, composite_waveforms[0], color="gold")

    ax_composite.set_title("Composite Waveforms (All Sources Summed)")
    ax_composite.set_xlabel("Time (s)")
    ax_composite.set_ylabel("Amplitude")
    ax_composite.grid(True)
    ax_composite.legend(loc="upper right")

    # ---------------------------------------------------------
    # 2) Individual Source Waveforms (Rows 1..n_sources)
    # ---------------------------------------------------------
    # Define a small palette of highlight colors for each source's first mic.
    # These will be cycled if there are more sources than colors.
    first_mic_colors = ["#FF1744", "#00BFA5", "#FFA343", "#9A4DFF", "#3498DB"]

    for i, (src_label, waveforms_for_mics) in enumerate(
        individual_waveforms.items(), start=1
    ):
        ax_source = fig.add_subplot(gs[i, 0])

        # Choose a highlight color for this source's first mic
        highlight_color = first_mic_colors[(i - 1) % len(first_mic_colors)]

        # Plot all microphones except the first in light gray
        for mic_idx in range(1, num_mics):
            ax_source.plot(t, waveforms_for_mics[mic_idx], color="lightgray")

        # Plot the first mic in the chosen highlight color
        ax_source.plot(t, waveforms_for_mics[0], color=highlight_color, label="Mic 0")

        ax_source.set_title(f"Source {src_label} Waveforms")
        ax_source.set_xlabel("Time (s)")
        ax_source.set_ylabel("Amplitude")
        ax_source.grid(True)
        ax_source.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


def plot_overview(sim):
    # TODO reuse plot_waveform_column_auto
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
    n_sources = len(sim.sources)
    total_rows = n_sources + 1  # 1 composite row + 1 row per source

    fig = plt.figure(figsize=(18, 2.5 * total_rows))
    gs = fig.add_gridspec(
        total_rows, 2, width_ratios=[1, 1], height_ratios=[1] * total_rows
    )

    # -------------------------------------------------------------------------
    # 1) Left column: geometry (spans all rows in the first column)
    # -------------------------------------------------------------------------
    ax_geometry = fig.add_subplot(gs[:, 0])
    plot_geometry_on_ax(ax_geometry, sim.mics[:, 0], sim.mics[:, 1], sim.sources)

    # -------------------------------------------------------------------------
    # 2) Right column, row 0: Composite waveforms
    #    Highlight Mic #1 in black, others in light gray
    # -------------------------------------------------------------------------
    num_mics = sim.composite_waveforms.shape[0]

    ax_composite = fig.add_subplot(gs[0, 1])
    # The rest in light gray
    for mic_idx in range(1, num_mics):
        ax_composite.plot(
            sim.t,
            sim.composite_waveforms[mic_idx],
            color="lightgray",
            # label=f"Mic {mic_idx}",
            # consider no label to avoid legend clutter
        )

    # First mic in black
    ax_composite.plot(sim.t, sim.composite_waveforms[0], color="black", label="Mic 0")
    # Plot the measurement points
    ax_composite.scatter(sim.t, sim.composite_waveforms[0], color="gold")

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
        sim.individual_waveforms.items(), start=1
    ):
        ax_source = fig.add_subplot(gs[i, 1])

        # Pick a highlight color for this source
        highlight_color = first_mic_colors[(i - 1) % len(first_mic_colors)]

        # Other mics in gray
        for mic_idx in range(1, num_mics):
            ax_source.plot(sim.t, waveforms_for_mics[mic_idx], color="lightgray")
        # Mic #0 in highlight color
        ax_source.plot(
            sim.t, waveforms_for_mics[0], color=highlight_color, label="Mic 0"
        )

        ax_source.set_title(f"Source {src_label} Waveforms")
        ax_source.set_xlabel("Time (s)")
        ax_source.set_ylabel("Amplitude")
        ax_source.grid(True)
        ax_source.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


def plot_signals_side_by_side(
    t,
    x_signals,
    y_signals,
    x_sources,
    labels=("x Signals", "y Signals"),
    first_mic_colors=None,
):
    """
    Plot two sets of signals (x and y) side by side in two columns, annotating only the X signals with frequency and phase,
    ensuring each pair of signals (x,y) shares the same axis scale, and setting labels as titles for each column.

    Parameters:
        t (array): Time array for the x-axis.
        x_signals (np.array): 2D array of shape (num_signals, len(t)) for the left column.
        y_signals (np.array): 2D array of shape (num_signals, len(t)) for the right column.
        x_sources (list): List of source objects for x_signals, each with frequency and phase attributes.
        labels (tuple): Titles for the left and right columns.
        first_mic_colors (list, optional): Highlight colors for the first signal.
    """
    num_signals = max(x_signals.shape[0], y_signals.shape[0])

    fig = plt.figure(figsize=(14, 2.5 * num_signals))
    gs = fig.add_gridspec(num_signals, 2, width_ratios=[1, 1])

    if first_mic_colors is None:
        first_mic_colors = ["#FF1744", "#00BFA5", "#FFA343", "#9A4DFF", "#3498DB"]

    # Add column titles explicitly
    fig.text(0.25, 0.96, labels[0], ha="center", fontsize=16, fontweight="bold")
    fig.text(0.75, 0.96, labels[1], ha="center", fontsize=16, fontweight="bold")

    for i in range(num_signals):
        highlight_color = first_mic_colors[i % len(first_mic_colors)]

        # Determine the shared axis limits
        y_max = max(np.max(np.abs(x_signals[i])), np.max(np.abs(y_signals[i])))

        # Left column: x Signals
        ax_x = fig.add_subplot(gs[i, 0])
        ax_x.plot(t, x_signals[i], color=highlight_color, label=f"Source {i}")
        freq = x_sources[i].frequency
        phase = x_sources[i].phase
        ax_x.set_title(f"Freq: {freq:.0f} Hz, Phase: {phase:.2f} rad")
        ax_x.set_xlabel("Time (s)")
        ax_x.set_ylabel("Amplitude")
        ax_x.set_ylim(-y_max, y_max)
        ax_x.grid(True)
        ax_x.legend(loc="upper right")

        # Right column: y Signals
        ax_y = fig.add_subplot(gs[i, 1])
        ax_y.plot(t, y_signals[i], color=highlight_color, label=f"{i}")
        ax_y.set_xlabel("Time (s)")
        ax_y.set_ylabel("Amplitude")
        ax_y.set_ylim(-y_max, y_max)
        ax_y.grid(True)
        ax_y.legend(loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
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
      - Value: magnitude (normalized to 0-1), but set to 0 (black) for small magnitudes regardless of angle.
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

    # Set values with small magnitude to black (Value = 0) regardless of angle.
    # Use a threshold of 0.0001 for magnitude, matching the display logic elsewhere.
    threshold = 0.0001
    small_mask = magnitude < threshold
    hsv[small_mask, 2] = 0  # Set Value to 0 for small magnitudes.

    # Convert HSV to RGB.
    rgb = hsv_to_rgb(hsv)
    return rgb


def plot_complex_matrix_on_ax(
    ax, matrix, title="", show_values=True, polar=False, font_size=8
):
    """
    Plot a complex matrix on the provided Axes object using an HSV mapping.

    The cell color is determined by:
      - Hue: phase of the complex number.
      - Value: magnitude.

    Optionally, the numerical complex value is overlaid.
    """
    # Expand if single column vector
    # not sure if this causes problems
    # essential that it is before complex to rgb
    if np.ndim(matrix) < 2:
        matrix = np.expand_dims(matrix, axis=1)

    # Convert the complex matrix to an RGB image.
    rgb = complex_to_rgb(matrix)
    ax.imshow(rgb, interpolation="none", aspect="auto")
    ax.set_title(title)
    # set the title size
    ax.title.set_fontsize(font_size + 20)
    ax.axis("off")

    if show_values:
        n, m = matrix.shape
        for i in range(n):
            for j in range(m):
                if abs(matrix[i, j]) < 0.01:
                    value_str = "0"
                elif polar:
                    # Compute magnitude and phase (in degrees)
                    r = np.abs(matrix[i, j])
                    if polar == "absolute":
                        value_str = f"{r:.1f}"
                    else:
                        theta = np.angle(matrix[i, j], deg=True)
                        if theta == 180:
                            value_str = f"{-r:.1f}"
                        elif theta == 0:
                            value_str = f"{r:.1f}"
                        else:
                            value_str = f"{r:.1f}∠{theta:.0f}°"
                else:
                    value_str = f"{matrix[i, j].real:.1f}{matrix[i, j].imag:+.1f}j"
                ax.text(
                    j,
                    i,
                    value_str,
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=font_size,
                )


def plot_complex_matrix(matrix, title="Complex Matrix", show_values=True):
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_complex_matrix_on_ax(ax, matrix, title=title, show_values=show_values)
    plt.show()


def plot_equation(
    Y,
    C,
    X,
    titles=("Y", "C", "X"),
    show_values=True,
    polar=True,
    ratios=[1, 1, 1],
    symbols=False,
    font_size=8,
):
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
        1, 3, figsize=(15, 5), gridspec_kw={"width_ratios": ratios, "wspace": 0.1}
    )

    # Plot each matrix on its own axis.
    plot_complex_matrix_on_ax(
        axs[0],
        Y,
        title=titles[0],
        show_values=show_values,
        polar=polar,
        font_size=font_size,
    )
    plot_complex_matrix_on_ax(
        axs[1],
        C,
        title=titles[1],
        show_values=show_values,
        polar=polar,
        font_size=font_size,
    )
    plot_complex_matrix_on_ax(
        axs[2],
        X,
        title=titles[2],
        show_values=show_values,
        polar=polar,
        font_size=font_size,
    )

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
    if symbols:
        fig.text(eq_x, eq_y, "=", fontsize=30, ha="center", va="center")
        fig.text(mult_x, eq_y, "×", fontsize=30, ha="center", va="center")

        fig.suptitle(f"Equation: {titles[0]} = {titles[1]}{titles[2]}", fontsize=32)
    plt.show()


def plot_two_line_equation(
    Y1,
    C1,
    X1,
    titles1=("Y1", "C1", "X1"),
    Y2=None,
    C2=None,
    X2=None,
    titles2=("Y2", "C2", "X2"),
    show_values=True,
    polar=True,
    ratios=[1, 1, 1],
    symbols=False,
    font_size=8,
):
    """
    Plot two equations of three matrices each, one above the other.

    Each equation is plotted as Y = C × X, with matrices side by side.
    """
    fig, axs = plt.subplots(
        2,
        3,
        figsize=(15, 10),
        gridspec_kw={"width_ratios": ratios, "wspace": 0.1, "hspace": 0.2},
    )

    # First equation (top row)
    plot_complex_matrix_on_ax(
        axs[0, 0],
        Y1,
        title=titles1[0],
        show_values=show_values,
        polar=polar,
        font_size=font_size,
    )
    plot_complex_matrix_on_ax(
        axs[0, 1],
        C1,
        title=titles1[1],
        show_values=show_values,
        polar=polar,
        font_size=font_size,
    )
    plot_complex_matrix_on_ax(
        axs[0, 2],
        X1,
        title=titles1[2],
        show_values=show_values,
        polar=polar,
        font_size=font_size,
    )

    # Second equation (bottom row), if provided
    if Y2 is not None and C2 is not None and X2 is not None:
        plot_complex_matrix_on_ax(
            axs[1, 0],
            Y2,
            title=titles2[0],
            show_values=show_values,
            polar=polar,
            font_size=font_size,
        )
        plot_complex_matrix_on_ax(
            axs[1, 1],
            C2,
            title=titles2[1],
            show_values=show_values,
            polar=polar,
            font_size=font_size,
        )
        plot_complex_matrix_on_ax(
            axs[1, 2],
            X2,
            title=titles2[2],
            show_values=show_values,
            polar=polar,
            font_size=font_size,
        )
    else:
        # If not provided, perhaps hide or leave blank
        for j in range(3):
            axs[1, j].axis('off')

    if symbols:
        # For first row
        pos0 = axs[0, 0].get_position()
        pos1 = axs[0, 1].get_position()
        pos2 = axs[0, 2].get_position()
        eq_x = (pos0.x1 + pos1.x0) / 2.0
        mult_x = (pos1.x1 + pos2.x0) / 2.0
        eq_y1 = (pos1.y0 + pos1.y1) / 2.0
        fig.text(eq_x, eq_y1, "=", fontsize=30, ha="center", va="center")
        fig.text(mult_x, eq_y1, "×", fontsize=30, ha="center", va="center")

        # For second row, if applicable
        if Y2 is not None:
            eq_y2 = (axs[1, 1].get_position().y0 + axs[1, 1].get_position().y1) / 2.0
            fig.text(eq_x, eq_y2, "=", fontsize=30, ha="center", va="center")
            fig.text(mult_x, eq_y2, "×", fontsize=30, ha="center", va="center")

    plt.show()


def plot_C_pinv(C, selected_frequency):
    C_f0 = C[:, :, selected_frequency]
    C_pinv = np.linalg.pinv(C_f0)
    id_f0 = C_pinv @ C_f0

    plot_equation(
        id_f0,
        C_f0,
        C_pinv,
        titles=(
            rf"$C_{{{selected_frequency}}}^{{\dagger}} \cdot C_{{{selected_frequency}}} \approx I$",
            rf"$C_{{{selected_frequency}}}$ ",
            rf"$C_{{{selected_frequency}}}^{{\dagger}}$",
        ),
        ratios=(id_f0.shape[1], C_f0.shape[1], C_pinv.shape[1]),
        polar="absolute",
    )


def plot_time_and_frequency(
    function_list,
    fourier_list,
    time_axis=None,
    frequency_axis=None,
    title=None,
    absolute=True,
):
    """
    Plots a time-domain function and its Fourier transform in a two-panel figure.

    Parameters:
        function_values (array-like): Array of values representing the function in the time domain.
        fourier_values (array-like): Array representing the Fourier transform of the function.
        time_axis (array-like, optional): Array of time points corresponding to function_values.
            Defaults to np.arange(len(function_values)) if None.
        frequency_axis (array-like, optional): Array of frequency points corresponding to fourier_values.
            Defaults to np.arange(len(fourier_values)) if None.
    """
    # # Ensure function_list and fourier_list are lists
    # if not isinstance(function_list, list):
    #     function_list = [function_list]
    # if not isinstance(fourier_list, list):
    #     fourier_list = [fourier_list]

    # num_functions = len(function_list)
    if np.ndim(function_list) < 2:
        function_list = np.expand_dims(function_list, axis=0)
    if np.ndim(fourier_list) < 2:
        fourier_list = np.expand_dims(fourier_list, axis=0)

    # Set default axes if not provided
    if time_axis is None:
        time_axis = np.arange(len(function_values))
    if frequency_axis is None:
        frequency_axis = np.arange(len(fourier_values))

    # Create a figure with two subplots: time domain on top, frequency domain below
    fig, (ax_time, ax_freq) = plt.subplots(2, 1, figsize=(10, 8))

    if title is not None:
        fig.suptitle(title, fontsize=16)

    # Define colors for plotting
    colors = ["orange", "blue", "green", "red", "purple", "cyan", "magenta"]

    # Plot the time-domain functions
    for i, function_values in enumerate(function_list):
        color = colors[i % len(colors)]
        label = f"Function {i}"
        ax_time.plot(time_axis, function_values, color=color, lw=2, label=label)

    # Plot the time-domain function
    ax_time.plot(time_axis, function_values, color="orange", lw=2)
    ax_time.set_title("Function Graph (Time Domain)")
    ax_time.set_xlabel("Time")
    ax_time.set_ylabel("Amplitude")
    ax_time.grid(True)

    # Plot the Fourier transforms
    for i, fourier_values in enumerate(fourier_list):
        color = colors[i % len(colors)]
        label = f"Fourier {i}"
        if absolute:
            ax_freq.plot(
                frequency_axis,
                abs(fourier_values),
                color=color,
                lw=2,
                label=f"|{label}|",
            )
        else:
            ax_freq.plot(
                frequency_axis,
                fourier_values.real,
                color=color,
                lw=2,
                linestyle="--",
                label=f"Re({label})",
            )
            ax_freq.plot(
                frequency_axis,
                fourier_values.imag,
                color=color,
                lw=2,
                linestyle=":",
                label=f"Im({label})",
            )
    ax_freq.legend(loc="upper left")
    ax_freq.set_title("Fourier Transform (Frequency Domain)")
    ax_freq.set_xlabel("Frequency")
    ax_freq.set_ylabel("Magnitude")
    ax_freq.grid(True)

    plt.tight_layout()
    plt.show()


def plot_matrix_3D(C):
    """
    Plot a 3D visualization of the complex-valued C matrix where color represents phase and magnitude.

    Parameters:
    - C: A 3D numpy array (num_mics, num_sources, N), where each element is a complex number.
    """
    num_mics, num_sources, N = C.shape

    # Create a figure and 3D axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Create grid positions
    X, Y, Z = np.meshgrid(
        np.arange(num_sources), np.arange(num_mics), np.arange(N), indexing="ij"
    )

    # Flatten arrays for plotting
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_flat = Z.flatten()
    C_flat = C.flatten()

    # Convert complex values to RGB colors
    colors = complex_to_rgb(C)
    colors_flat = colors.reshape(-1, 3)  # Reshape to a list of RGB colors

    # Plot the 3D scatter with colors
    ax.scatter(X_flat, Y_flat, Z_flat, c=colors_flat, marker="o", s=20)

    # Labels and title
    ax.set_xlabel("Sources")
    ax.set_ylabel("Microphones")
    ax.set_zlabel("Frequency Index")
    ax.set_title("3D Visualization of Mixing Matrix C (Color: Phase & Magnitude)")

    plt.show()


def wrapper_plot_geometry(
    sim, figsize=(8, 8), fontsize=16, pad_factor=0.4, skip_labels=False
):
    plot_geometry_auto(
        sim.mics[:, 0],
        sim.mics[:, 1],
        sim.sources,
        figsize=figsize,
        fontsize=fontsize,
        pad_factor=pad_factor,
        skip_labels=skip_labels,
    )
