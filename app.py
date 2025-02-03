import numpy as np 
import matplotlib.pyplot as plt
from itertools import cycle

# Constants
colors = cycle(['#FFA343', '#FF1744', '#9A4DFF', '#00BFA5', '#3498DB'])
speed_of_sound = 343 # m/s

def initialize_array(array_size, microphone_spacing):
    microphone_positions = np.linspace(-(array_size) / 2, (array_size) / 2, array_size) * microphone_spacing
    return microphone_positions

def initialize_circular_array(array_size, radius):
    angle_spacing = 2 * np.pi / array_size

    x_microphone_positions = np.array( [ radius * np.cos(x * angle_spacing) for x in range(array_size)] )
    y_microphone_positions = np.array( [ radius * np.sin(x * angle_spacing) for x in range(array_size)] )
    return x_microphone_positions, y_microphone_positions

def plot_array_and_source(array_size=8, source_distance=2.0, source_angle=(1/4)*np.pi, microphone_spacing=0.2, microphone_radius=False):

    if (microphone_radius):
        x_microphone_positions, y_microphone_positions = initialize_circular_array(array_size, microphone_radius)
    else:
        x_microphone_positions = initialize_array(array_size, microphone_spacing)
        y_microphone_positions = np.zeros_like(x_microphone_positions)
    
    # Source position
    source_x = source_distance * np.sin(source_angle)
    source_y = source_distance * np.cos(source_angle)

    # Plot the array and source
    lim = max(array_size * microphone_spacing / 2, source_distance) * 1.05 # migth create issues wih circular array
    plt.scatter(x_microphone_positions, y_microphone_positions, label="Microphones", color=next(colors))
    plt.scatter(source_x, source_y, label=f"Sound Source", color=next(colors))
    plt.scatter(0, 0, label="Origo", color=next(colors))
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    plt.title("Microphone Array and Sound Source")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.legend()
    plt.grid()
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.show()

def calculate_delays(x_microphone_positions, y_microphone_positions, source_distance, source_angle, frequency):
    source_x = source_distance * np.sin(source_angle)
    source_y = source_distance * np.cos(source_angle)

    distances = np.sqrt((x_microphone_positions - source_x)**2 + (y_microphone_positions - source_y)**2)
    delays = distances / speed_of_sound #s
    phase_shifts = 2 * np.pi * frequency * delays # angle frequency times respective delay

    return phase_shifts, delays

def simulate_waveforms(phase_shifts, frequency, sampling_rate):
    t = np.linspace(0,1 / frequency, int(sampling_rate / frequency), endpoint=False)
    waveforms = []
    for shift in phase_shifts:
        waveforms.append(np.sin(2 * np.pi * frequency * t + shift))
    
    return t, np.array(waveforms)

def plot_waveforms(frequency, sampling_rate=100000, array_size=8, microphone_spacing=0.1, microphone_radius=False, source_distance=2.0, source_angle= (1/4)*np.pi):
    
    if (microphone_radius):
        x_microphone_positions, y_microphone_positions = initialize_circular_array(array_size, microphone_radius)
    else:
        x_microphone_positions = initialize_array(array_size, microphone_spacing)
        y_microphone_positions = np.zeros_like(x_microphone_positions)

    phase_shifts, delays = calculate_delays(x_microphone_positions, y_microphone_positions, source_distance, source_angle, frequency)
    time, waveforms = simulate_waveforms(phase_shifts, frequency, sampling_rate)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,12))

    # waveforms
    for i, waveform in enumerate(waveforms):
        ax1.plot(time, waveform, label=f"Mic {i+1} (Delay: {delays[i]*1e6:.0f} µs)")
    
    ax1.set_title("Simulated Waveforms at Each Microphone")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.legend()
    ax1.grid()

    # phase shifts
    ax2.stem(range(array_size), phase_shifts, )
    ax2.set_title("Phase Shifts for Each Microphone")
    ax2.set_xlabel("Microphone Index")
    ax2.set_ylabel("Phase Shift (radians)")
    ax2.grid()

    plt.show()


    


if __name__ == "__main__":
    r = 0.3
    plot_array_and_source(array_size=8, microphone_spacing=0.1, source_angle=1*np.pi/4, microphone_radius=r)
    plot_waveforms(frequency=100, microphone_radius=r)

