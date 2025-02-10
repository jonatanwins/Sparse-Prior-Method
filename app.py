import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

# Constants
colors = cycle(["#FFA343", "#FF1744", "#9A4DFF", "#00BFA5", "#3498DB"])
speed_of_sound = 343  # m/s


def initialize_linear_array(array_size, microphone_spacing):
    microphone_positions = (
        np.linspace(-(array_size) / 2, (array_size) / 2, array_size)
        * microphone_spacing
    )
    return microphone_positions


def initialize_circular_array(array_size, radius):
    angle_spacing = 2 * np.pi / array_size

    x_microphone_positions = np.array(
        [radius * np.cos(x * angle_spacing) for x in range(array_size)]
    )
    y_microphone_positions = np.array(
        [radius * np.sin(x * angle_spacing) for x in range(array_size)]
    )
    return x_microphone_positions, y_microphone_positions


# ------------------------
# Sound source class
# ------------------------


class SoundSource:
    def __init__(self, distance, angle, frequency, amplitude=1.0):
        self.distance = distance
        self.angle = angle
        self.frequency = frequency
        self.amplitude = amplitude

    def get_position(self):
        x = self.distance * np.sin(self.angle)
        y = self.distance * np.cos(self.angle)
        return x, y


# ------------------------
# Simulation
# ------------------------


def calculate_delays(x_positions, y_positions, source: SoundSource):
    source_x, source_y = source.get_position()
    distances = np.sqrt((x_positions - source_x) ** 2 + (y_positions - source_y) ** 2)
    delays = distances / speed_of_sound
    return delays


def simulate_waveform_for_source(x_positions, y_positions, source, t):
    """
    Compute the waveform at each microphone due to one sound source.

    Returns:
        waveforms: A 2D array (mic index x time samples) for the source.
        delays: Delay for each microphone.
    """

    delays = calculate_delays(x_positions, y_positions, source)
    phase_shifts = (
        2 * np.pi * source.frequency * delays
    )  # angle frequency times respective delay

    waveforms = np.array(
        [
            source.amplitude * np.sin(2 * np.pi * source.frequency * t + shift)
            for shift in phase_shifts
        ]
    )

    return waveforms, delays


def simulate_waveforms_multiple_sources(
    x_positions, y_positions, sources, sampling_rate=10_000, duration=None
):
    """
    TODO
    composite waveforms, uses same t for all
    """
    if duration is None:
        min_freq = min(source.frequency for source in sources)
        duration = 1 / min_freq

    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    composite_waveforms = np.zeros((len(x_positions), len(t)))
    individual_waveforms = {}
    delays_dict = {}

    for idx, source in enumerate(sources):
        waveform, delays = simulate_waveform_for_source(
            x_positions, y_positions, source, t
        )  # same t
        composite_waveforms += waveform
        individual_waveforms[f"Source {idx+1}"] = waveform
        delays_dict[f"Source {idx+1}"] = delays

    return t, composite_waveforms, individual_waveforms, delays_dict


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
    ax.scatter(
        0,
        0,
        s=60,
        marker="s",
        color="black",
        edgecolor="k",
        label="Origin",
    )

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
    print(f"{x_max=}, {x_min=}, {y_max=}, {y_min}")
    print([src.get_position() for src in sources])
    max_span = max(x_span, y_span)

    # We'll pad by 10%
    pad_factor = 0.2
    half_span = 1.5 * max_span * (1 + pad_factor)

    # Final limits
    ax.set_xlim(-half_span, half_span)
    ax.set_ylim(-half_span, half_span)
    print(f"{half_span=}")

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


def plot_all(
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

        ax_source.set_title(f"{src_label} Waveforms")
        ax_source.set_xlabel("Time (s)")
        ax_source.set_ylabel("Amplitude")
        ax_source.grid(True)
        ax_source.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


# ------------------------
# Experiments
# ------------------------


def experiment_1():
    array_size = 8
    microphone_spacing = 0.1
    microphone_radius = 0.2

    if microphone_radius is not None:
        x_mics, y_mics = initialize_circular_array(array_size, microphone_radius)
    else:
        x_mics = initialize_linear_array(array_size, microphone_spacing)
        y_mics = np.zeros_like(x_mics)

    sources = [
        SoundSource(distance=2.0, angle=np.pi / 4, frequency=100, amplitude=1.0),
        SoundSource(distance=3.0, angle=np.pi / 3, frequency=150, amplitude=0.8),
        SoundSource(distance=3.0, angle=np.pi / 5, frequency=300, amplitude=0.8),
    ]

    t, composite_waveforms, individual_waveforms, delays_dict = (
        simulate_waveforms_multiple_sources(x_mics, y_mics, sources)
    )

    # Plotting
    # plot_array_and_sources(x_mics, y_mics, sources)
    # plot_waveforms(t, composite_waveforms, delays_dict)

    plot_all(x_mics, y_mics, sources, t, composite_waveforms, individual_waveforms)


if __name__ == "__main__":
    experiment_1()
