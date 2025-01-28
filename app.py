import numpy as np 
import matplotlib.pyplot as plt
from itertools import cycle

# Constants
colors = cycle(['#FFA343', '#FF1744', '#9A4DFF', '#00BFA5', '#3498DB'])




def initialize_array(array_size, microphone_spacing):
    microphone_positions = np.linspace(-(array_size) / 2, (array_size) / 2, array_size) * microphone_spacing
    return microphone_positions

def plot_array_and_source(microphone_positions, source_distance, source_angle):
    
    # Source position
    source_x = source_distance * np.sin(source_angle)
    source_y = source_distance * np.cos(source_angle)

    # Plot the array and source
    plt.scatter(microphone_positions, np.zeros_like(microphone_positions), label="Microphones", color=next(colors))
    plt.scatter(source_x, source_y, label=f"Sound Source", color=next(colors))
    plt.scatter(0, 0, label="Origo", color=next(colors))
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    plt.title("Microphone Array and Sound Source")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.legend()
    plt.grid()
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.show()


if __name__ == "__main__":
    
    array_size = 8
    microphone_spacing = 0.1
    source_distance = 2.0
    source_angle = (1/4)*np.pi

    # Initialize the array and plot
    microphone_positions = initialize_array(array_size, microphone_spacing)
    plot_array_and_source(microphone_positions, source_distance, source_angle)