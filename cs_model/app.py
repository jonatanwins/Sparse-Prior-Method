import random
import sys
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq, fftshift, ifft, rfftfreq

from cs_model.DFT import DFT, DFT_matrix
from cs_model.plotting import (
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
    def __init__(
        self, distance, angle, frequency, amplitude=1.0, phase=0.0, function=np.sin
    ):
        self.distance = distance
        self.angle = angle
        self.frequency = frequency
        self.amplitude = amplitude
        self.phase = phase  # in radians
        self.function = function

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


def simulate_waveform_for_source(x_positions, y_positions, src, t):
    """
    Compute the waveform at each microphone due to one sound source.

    Returns:
        waveforms: A 2D array (mic index x time samples) for the source.
        delays: Delay for each microphone.
    """

    delays = calculate_delays(x_positions, y_positions, src)

    # at the microphone, hence the shift
    # BUG SQUASH: the shift is negative as we are at an earlier stage of the signal
    waveforms = np.array(
        [
            src.amplitude
            * src.function(2 * np.pi * src.frequency * (t - delay) + src.phase)
            for delay in delays
        ]
    )

    return waveforms, delays


def simulate_waveforms_multiple_sources(
    x_positions, y_positions, sources, sampling_rate, duration=None
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
            function=src.function,
        )
        for src in sources
    ]
    random.seed(seed)
    indicies = random.sample(range(len(sources)), s)
    for i in indicies:
        sparse_sources[i] = sources[i]
    return sparse_sources, indicies


def recover_x(X_2s, len_x, ONLY_FOR_PLOT):
    """ "
    Theorem 2.15 - Pronys method
    Reconstructs s-sparse X from 2s Y measurements
    """
    s = len(X_2s) // 2
    A = np.zeros((s, s), dtype=complex)

    for k in range(s):
        A[:, s - k - 1] = X_2s[k : k + s]  # Sliding X_2s

    b = -X_2s[s : 2 * s]
    w = np.linalg.solve(A, b)  # p hat values from 1 to s

    # plot_equation(A, w, b, titles=("A", "w", "-X_2s[s : 2s]"))

    q_hat = np.concatenate(([1], w, np.zeros(len_x - (s + 1))))
    # plot_equation(
    #     q_hat,
    #     w,
    #     np.zeros(len_x - (s + 1)),
    #     titles=(r"$\hat q$", "w", "np.zeros(len_x- (s+1))"),
    # )

    q = ifft(q_hat)
    plot_equation(
        q_hat,
        10_000 * q,
        ONLY_FOR_PLOT,
        titles=(r"$\hat q$", r"$10 000 \cdot q$", "x_true"),
    )
    inds = np.where(np.abs(q) <= 1e-5)[0]

    exp_matrix = np.exp(-2j * np.pi * np.outer(np.arange(s), inds) / len_x)

    x_S = np.linalg.solve(exp_matrix, X_2s[:s])
    # plot_equation(
    #     exp_matrix,
    #     x_S,
    #     X_2s[:s],
    #     titles=(r"$ \mathcal{F}$", r"$x_S$", r"$X_{2s}[:s]$"),
    # )

    x = np.zeros(len_x, dtype=complex)
    x[inds] = x_S

    return x


def run_simulation(
    array_type="circular",
    num_mics=5,
    spacing=0.1,
    no_sources=1,
    s_sparse=None,
    f0=100,
    distance=1.5,
    amplitude=2,
    phase_step=0.3,
    angle_base=np.pi / 4,
    sampling_rate_factor=10,
    simulation_duration=None,
):

    # 1. Initialize microphone array.
    if array_type.lower() == "linear":
        x_mics, y_mics = initialize_linear_array(num_mics, spacing)
    elif array_type.lower() == "circular":
        x_mics, y_mics = initialize_circular_array(num_mics, spacing)
    else:
        raise ValueError("array_type must be 'linear' or 'circular'")

    # 2. Create sound sources.
    sources = [
        SoundSource(
            distance=distance,
            angle=angle_base + phase_step * a,
            frequency=f0,
            amplitude=amplitude,
            phase=phase_step * a,
            function=np.sin,
        )
        for a in range(no_sources)
    ]
    if s_sparse is not None and s_sparse < no_sources:
        sources, active_indices = s_sparse_sources(s_sparse, sources)
    else:
        active_indices = list(range(no_sources))

    # 3. Define sampling parameters.
    sampling_rate = sampling_rate_factor * f0
    if simulation_duration is None:
        simulation_duration = max(1 / src.frequency for src in sources)
    N = int(sampling_rate * simulation_duration)

    # 4. Simulate the waveforms at the microphones.
    t, composite_waveforms, individual_waveforms, delays_dict = (
        simulate_waveforms_multiple_sources(
            x_mics, y_mics, sources, sampling_rate, simulation_duration
        )
    )

    # 5. TIME DOMAIN SIGNALS.
    x_time = np.array(
        [
            src.amplitude * src.function(2 * np.pi * src.frequency * t + src.phase)
            for src in sources
        ]
    )
    y_time = composite_waveforms

    # 6. FREQUENCY DOMAIN PROCESSING.
    X = fft(x_time, axis=1)
    freqs = fftfreq(N, d=1 / sampling_rate)

    # Build the mixing matrix C: dimensions [num_mics x num_sources x N].
    C = np.zeros((num_mics, len(sources), N), dtype=complex)
    for i in range(num_mics):
        for j, src in enumerate(sources):
            C[i, j, :] = np.exp(-2j * np.pi * freqs * delays_dict[j][i])

    # Predicted observations in the frequency domain.
    Y_pred = np.zeros((num_mics, N), dtype=complex)
    for idf in range(N):
        Y_pred[:, idf] = C[:, :, idf] @ X[:, idf]

    # Reconstruct the source spectrum.
    Y_fft = fft(y_time, axis=1)
    num_sources = len(sources)
    X_pred = np.zeros((num_sources, N), dtype=complex)
    C_pinv = np.zeros((num_sources, num_mics, N), dtype=complex)
    for idf in range(N):
        C_f = C[:, :, idf]
        C_f_pinv = np.linalg.pinv(C_f)
        C_pinv[:, :, idf] = C_f_pinv
        X_pred[:, idf] = C_f_pinv @ Y_fft[:, idf]

    # Inverse FFT to recover time-domain source signals.
    x_pred = ifft(X_pred, axis=1)

    sim_result = SimpleNamespace(
        t=t,
        composite_waveforms=composite_waveforms,
        individual_waveforms=individual_waveforms,
        delays_dict=delays_dict,
        x_time=x_time,
        y_time=y_time,
        X=X,
        Y=Y_fft,
        freqs=freqs,
        C=C,
        C_pinv=C_pinv,
        Y_pred=Y_pred,
        X_pred=X_pred,
        x_pred=x_pred,
        sources=sources,
        x_mics=x_mics,
        y_mics=y_mics,
        sampling_rate=sampling_rate,
        duration=simulation_duration,
        N=N,
        active_indices=active_indices,
    )
    return sim_result


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
            phase=0.3 * a,
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
            src.amplitude * src.function(2 * np.pi * src.frequency * (t) + src.phase)
            for src in sources
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
        plot_equation(Y[:, 1], C[:, :, 1], X[:, 1], ratios=(1, 8, 1), font_size=12)

        # Reconstructing X
        plot_equation(
            X_pred, X, np.array([[1]]), titles=("X_pred", "X", ""), ratios=(1, 1, 0)
        )

        # plot_signals_side_by_side(t, ifft(X_pred), x, sources, labels=("x_pred", "x"))

        selected_frequency = 1
        plot_C_pinv(C, selected_frequency)


def experiment_8():
    """
    Dummy example testing Pronys method
    """
    N = 32
    x = np.zeros(N, dtype=complex)
    non_zero_inds = [3, 5, 9, 12]  # known as S in Foucart MICS
    x[non_zero_inds] = [1, 4, 3, 2]
    s = len(non_zero_inds)

    X = fft(x)
    x_recovered = recover_x(X[: 2 * s], N, x)
    # plot_equation(x_recovered, x, X, titles=("x_recovered", "x", "X"))
    # plot_equation(x, X, X[: 2 * s], titles=("x", "X", r"$X_{2s}$"), font_size=10)


if __name__ == "__main__":
    experiment_7(True)
