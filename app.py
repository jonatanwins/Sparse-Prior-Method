import numpy as np 
import matplotlib.pyplot as plt
from itertools import cycle

# Constants
colors = cycle(['#FFA343', '#FF1744', '#9A4DFF', '#00BFA5', '#3498DB'])
speed_of_sound = 343 # m/s

def initialize_linear_array(array_size, microphone_spacing):
    microphone_positions = np.linspace(-(array_size) / 2, (array_size) / 2, array_size) * microphone_spacing
    return microphone_positions

def initialize_circular_array(array_size, radius):
    angle_spacing = 2 * np.pi / array_size

    x_microphone_positions = np.array( [ radius * np.cos(x * angle_spacing) for x in range(array_size)] )
    y_microphone_positions = np.array( [ radius * np.sin(x * angle_spacing) for x in range(array_size)] )
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
    distances = np.sqrt((x_positions - source_x)**2 + (y_positions - source_y)**2)
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
    phase_shifts = 2 * np.pi * source.frequency * delays # angle frequency times respective delay
    
    waveforms = np.array([
        source.amplitude * np.sin(2 * np.pi * source.frequency * t + shift)
        for shift in phase_shifts
    ])

    return waveforms, delays

def simulate_waveforms_multiple_sources(x_positions, y_positions, sources, sampling_rate=10_000, duration=None):
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
        waveform, delays = simulate_waveform_for_source(x_positions, y_positions, source, t) # same t
        composite_waveforms += waveform
        individual_waveforms[f"Source {idx+1}"] = waveform
        delays_dict[f"Source {idx+1}"] = delays

    return t, composite_waveforms, individual_waveforms, delays_dict


# ------------------------
# Plotting
# ------------------------
def plot_array_and_sources(x_positions, y_positions, sources):
    """Plot the microphone array and the positions of all sound sources."""
    plt.figure(figsize=(8,8))
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
    plt.axis('equal')
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

def plot_all(x_positions, y_positions, sources, t, composite_waveforms, delays_dict):
    """
    Create a single figure that includes:
      1. Microphone array and sound source positions.
      2. Composite waveforms at each microphone.
      3. Delay plots for each sound source.
    """
    n_delay_plots = len(delays_dict)
    total_rows = 2 + n_delay_plots  # one row for array geometry, one for waveform, and one per source delay plot

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
    ax.axis('equal')

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


# def plot_array_and_source(array_size=8, source_distance=2.0, source_angle=(1/4)*np.pi, microphone_spacing=0.2, microphone_radius=False):

#     if (microphone_radius):
#         x_microphone_positions, y_microphone_positions = initialize_circular_array(array_size, microphone_radius)
#     else:
#         x_microphone_positions = initialize_linear_array(array_size, microphone_spacing)
#         y_microphone_positions = np.zeros_like(x_microphone_positions)
    
#     # Source position
#     source_x = source_distance * np.sin(source_angle)
#     source_y = source_distance * np.cos(source_angle)

#     # Plot the array and source
#     lim = max(array_size * microphone_spacing / 2, source_distance) * 1.05 # migth create issues wih circular array
#     plt.scatter(x_microphone_positions, y_microphone_positions, label="Microphones", color=next(colors))
#     plt.scatter(source_x, source_y, label=f"Sound Source", color=next(colors))
#     plt.scatter(0, 0, label="Origo", color=next(colors))
#     plt.axhline(0, color="gray", linestyle="--", linewidth=0.5)
#     plt.title("Microphone Array and Sound Source")
#     plt.xlabel("X Position (m)")
#     plt.ylabel("Y Position (m)")
#     plt.legend()
#     plt.grid()
#     plt.xlim(-lim, lim)
#     plt.ylim(-lim, lim)
#     plt.show()

# def plot_waveforms(frequency, sampling_rate=100000, array_size=8, microphone_spacing=0.1, microphone_radius=False, source_distance=2.0, source_angle= (1/4)*np.pi):
    
#     if (microphone_radius):
#         x_microphone_positions, y_microphone_positions = initialize_circular_array(array_size, microphone_radius)
#     else:
#         x_microphone_positions = initialize_linear_array(array_size, microphone_spacing)
#         y_microphone_positions = np.zeros_like(x_microphone_positions)

#     phase_shifts, delays = calculate_delays(x_microphone_positions, y_microphone_positions, source_distance, source_angle, frequency)
#     time, waveforms = simulate_waveforms(phase_shifts, frequency, sampling_rate)
    
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,12))

#     # waveforms
#     for i, waveform in enumerate(waveforms):
#         ax1.plot(time, waveform, label=f"Mic {i+1} (Delay: {delays[i]*1e6:.0f} µs)")
    
#     ax1.set_title("Simulated Waveforms at Each Microphone")
#     ax1.set_xlabel("Time (s)")
#     ax1.set_ylabel("Amplitude")
#     ax1.legend()
#     ax1.grid()

#     # phase shifts
#     ax2.stem(range(array_size), phase_shifts, )
#     ax2.set_title("Phase Shifts for Each Microphone")
#     ax2.set_xlabel("Microphone Index")
#     ax2.set_ylabel("Phase Shift (radians)")
#     ax2.grid()

#     plt.show()

# ------------------------
# Experiments
# ------------------------

def experiment_1():
    array_size = 8
    microphone_spacing = 0.1
    microphone_radius = None
    
    if microphone_radius is not None:
        x_mics, y_mics = initialize_circular_array(array_size, microphone_radius)
    else:
        x_mics = initialize_linear_array(array_size, microphone_spacing)
        y_mics = np.zeros_like(x_mics)
    
    sources = [
        SoundSource(distance=2.0, angle=np.pi/4, frequency=100, amplitude=1.0),
        SoundSource(distance=3.0, angle=np.pi/3, frequency=150, amplitude=0.8),
        SoundSource(distance=3.0, angle=np.pi/5, frequency=300, amplitude=0.8)

    ]
    
    
    t, composite_waveforms, individual_waveforms, delays_dict = simulate_waveforms_multiple_sources(
        x_mics, y_mics, sources
    )
    
    # Plotting
    # plot_array_and_sources(x_mics, y_mics, sources)
    # plot_waveforms(t, composite_waveforms, delays_dict)

    plot_all(x_mics, y_mics, sources, t, composite_waveforms, delays_dict), 
    


if __name__ == "__main__":
    experiment_1()

