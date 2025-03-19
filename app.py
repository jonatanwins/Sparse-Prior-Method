import random
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq, fftshift, ifft, rfftfreq

from DFT import DFT, DFT_matrix
from plotting import (
    plot_C_pinv,
    plot_complex_matrix,
    plot_equation,
    plot_matrix_3D,
    plot_overview,
    plot_signals_side_by_side,
    plot_time_and_frequency,
    plot_Y_comparison,
)

# Constants
speed_of_sound = 343  # m/s


def initialize_linear_array(array_size, microphone_spacing):
    # Edge case
    if array_size == 1:
        return np.array([0.0]), np.array([0.0])

    x_microphone_positions = (
        np.linspace(-(array_size) / 2, (array_size) / 2, array_size)
        * microphone_spacing
    )
    return x_microphone_positions, np.zeros_like(x_microphone_positions)


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
    def __init__(self, distance, angle, frequency, amplitude=1.0, phase=0.0):
        self.distance = distance
        self.angle = angle
        self.frequency = frequency
        self.amplitude = amplitude
        self.phase = phase

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


def simulate_waveform_for_source(x_positions, y_positions, src, t, cosine=False):
    """
    Compute the waveform at each microphone due to one sound source.

    Returns:
        waveforms: A 2D array (mic index x time samples) for the source.
        delays: Delay for each microphone.
    """

    delays = calculate_delays(x_positions, y_positions, src)
    # BUG

    f = np.cos if cosine else np.sin

    # at the microphone, hence the shift
    # BUG SQUASH: the shift is negative as we are at an earlier stage of the signal
    waveforms = np.array(
        [
            src.amplitude * f(2 * np.pi * src.frequency * (t - delay + src.phase))
            for delay in delays
        ]
    )

    return waveforms, delays


def simulate_waveforms_multiple_sources(
    x_positions, y_positions, sources, sampling_rate=10_000, duration=None, cosine=False
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
            x_positions, y_positions, source, t, cosine=cosine
        )  # same t
        composite_waveforms += waveform
        individual_waveforms[idx] = waveform
        delays_dict[idx] = delays

    return t, composite_waveforms, individual_waveforms, delays_dict


def s_sparse_sources(s, sources, seed=2025):
    sparse_sources = [
        SoundSource(
            distance=src.distance,
            angle=src.angle,
            frequency=src.frequency,
            amplitude=0,
            phase=src.phase,
        )
        for src in sources
    ]
    random.seed(seed)
    indicies = random.sample(range(len(sources)), s)
    for i in indicies:
        sparse_sources[i] = sources[i]
    return sparse_sources, indicies


# ------------------------
# Experiments
# ------------------------


def experiment_7(plot=False):
    # TODO multiple sources and add noise

    no_sources = 8
    s_sparse = 3
    num_mics = 3
    radius = 1
    x_mics, y_mics = initialize_circular_array(num_mics, radius)

    # Frequency of interests
    f0 = 100

    sources = [
        SoundSource(
            distance=10,
            angle=0.3 * a,
            frequency=f0,
            amplitude=2,
            phase=0.001 * a,
        )
        for a in range(no_sources)
    ]
    sources, _ = s_sparse_sources(s_sparse, sources)

    sampling_rate = 10 * f0
    duration = max(1 / source.frequency for source in sources)
    N = int(sampling_rate * duration)

    num_sources = len(sources)

    t, composite_waveforms, individual_waveforms, delays_dict = (
        simulate_waveforms_multiple_sources(
            x_mics, y_mics, sources, sampling_rate=sampling_rate, duration=duration
        )
    )

    # 1. TIME DOMAIN
    x = np.array(
        [
            source.amplitude * np.sin(2 * np.pi * source.frequency * (t + source.phase))
            for source in sources
        ]
    )

    # These will have the summed amplitude of all the waves emitted.
    y = composite_waveforms

    mic_0_contributors = np.array(
        [individual_waveforms[i][0] for i in range(no_sources)]
    )

    # 2. FREQUENCY DOMAIN
    X = fft(x, axis=1)
    freqs = fftfreq(N, d=1 / sampling_rate)
    # f0 is constant
    C = np.zeros((num_mics, num_sources, N), dtype=complex)

    # C for every frequency
    for i in range(num_mics):
        for j, source in enumerate(sources):
            C[i, j] = np.exp(-2j * np.pi * freqs * delays_dict[j][i])

    Y_pred = np.zeros((num_mics, N), dtype=complex)
    for idf in range(N):
        Y_pred[:, idf] = C[:, :, idf] @ X[:, idf]

    # Recreating X
    Y = fft(y)
    X_pred = np.zeros((num_sources, N), dtype=complex)
    for idf in range(N):
        C_f_pinv = np.linalg.pinv(C[:, :, idf])
        X_pred[:, idf] = C_f_pinv @ Y[:, idf]

    # 3. PLOTTING
    if plot:
        plot_overview(
            x_mics, y_mics, sources, t, composite_waveforms, individual_waveforms
        )

        # Validating mixing model
        # plot_equation(
        #     Y,
        #     Y_pred,
        #     np.array([[1]]),
        #     titles=["Y_fft", "Y_pred=C*X", ""],
        #     ratios=[1, 1, 0],
        # )

        # Reconstructing X
        plot_equation(
            X_pred, X, np.array([[1]]), titles=("X_pred", "X", ""), ratios=(1, 1, 0)
        )

        # plot_signals_side_by_side(t, ifft(X_pred), x, sources, labels=("x_pred", "x"))

        selected_frequency = 1
        plot_C_pinv(C, selected_frequency)


if __name__ == "__main__":
    experiment_7(True)
