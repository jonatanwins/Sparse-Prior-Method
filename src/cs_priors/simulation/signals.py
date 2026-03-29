import numpy as np

""" Unused as of now, because we define white noise signals and sines directly in the simulation code."""


def sinusoidal_signal(t, frequency, amplitude=1.0, phase=0.0):
    """Generate a sinusoidal time-domain signal."""
    return amplitude * np.sin(2 * np.pi * frequency * t + phase)


def broadband_signal(t, frequencies, amplitudes, phases):
    """Sum of sinusoids."""
    return sum(
        a * np.sin(2 * np.pi * f * t + p)
        for f, a, p in zip(frequencies, amplitudes, phases)
    )


def white_noise_signal(t, amplitude=1.0, seed=None):
    """Generate white noise."""
    rng = np.random.default_rng(seed)
    return amplitude * rng.standard_normal(len(t))


def custom_signal(signal_array):
    """Pass through a pre-computed signal as-is."""
    return signal_array
