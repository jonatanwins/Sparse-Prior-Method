import numpy as np
from dataclasses import dataclass


# Microphone arrays are numpy arrays of xy-coordinates


def linear_array(array_size, microphone_spacing):
    # Edge case
    if array_size == 1:
        return np.array([[0.0, 0.0]])

    x_microphone_positions = (
        np.linspace(-(array_size) / 2, (array_size) / 2, array_size)
        * microphone_spacing
    )
    return np.column_stack(
        (x_microphone_positions, np.zeros_like(x_microphone_positions))
    )


def circular_array(array_size, radius):

    if array_size == 1:
        return np.array([[0.0, 0.0]])

    angle_spacing = 2 * np.pi / array_size

    x_microphone_positions = np.array(
        [radius * np.cos(x * angle_spacing) for x in range(array_size)]
    )
    y_microphone_positions = np.array(
        [radius * np.sin(x * angle_spacing) for x in range(array_size)]
    )
    return np.column_stack((x_microphone_positions, y_microphone_positions))
