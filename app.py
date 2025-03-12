import numpy as np
import matplotlib.pyplot as plt
import sys

from plotting import (
    plot_overview,
    plot_Y_comparison,
    plot_complex_matrix,
    plot_equation,
    plot_time_and_frequency,
    plot_matrix_3D,
    plot_C_pinv,
)
from scipy.fft import fft, ifft, fftfreq, fftshift, rfftfreq
from DFT import DFT, DFT_matrix


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


def simulate_waveform_for_source(x_positions, y_positions, source, t, cosine=False):
    """
    Compute the waveform at each microphone due to one sound source.

    Returns:
        waveforms: A 2D array (mic index x time samples) for the source.
        delays: Delay for each microphone.
    """

    delays = calculate_delays(x_positions, y_positions, source)
    # BUG

    f = np.cos if cosine else np.sin

    # at the microphone, hence the shift
    # BUG SQUASH: the shift is negative as we are at an earlier stage of the signal
    waveforms = np.array(
        [
            source.amplitude * f(2 * np.pi * source.frequency * (t - delay))
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
        x_mics, y_mics = initialize_linear_array(array_size, microphone_spacing)

    sources = [
        SoundSource(distance=2.0, angle=np.pi / 4, frequency=100, amplitude=1.0),
        SoundSource(distance=3.0, angle=np.pi / 3, frequency=150, amplitude=0.8),
        SoundSource(distance=3.0, angle=np.pi / 5, frequency=300, amplitude=0.8),
    ]

    t, composite_waveforms, individual_waveforms, delays_dict = (
        simulate_waveforms_multiple_sources(x_mics, y_mics, sources)
    )

    plot_overview(x_mics, y_mics, sources, t, composite_waveforms, individual_waveforms)


def experiment_2():
    array_size = 8
    microphone_spacing = 0.1
    x_mics, y_mics = initialize_linear_array(array_size, microphone_spacing)

    sources = [
        SoundSource(distance=2.0, angle=np.pi / 4, frequency=10, amplitude=5.0),
        SoundSource(distance=2.0, angle=np.pi / 5, frequency=50, amplitude=1.0),
        SoundSource(distance=2.0, angle=np.pi / 6, frequency=300, amplitude=0.3),
    ]

    t, composite_waveforms, individual_waveforms, delays_dict = (
        simulate_waveforms_multiple_sources(x_mics, y_mics, sources)
    )

    plot_overview(x_mics, y_mics, sources, t, composite_waveforms, individual_waveforms)

    # Fourier Transform of the composite waveforms
    N = len(t)
    # multiple waveforms, hence axis=1
    frequency_spectrum = fft(composite_waveforms, axis=1)
    fft_frequencies = fftfreq(N, d=t[1] - t[0])

    # Plot the frequency spectrum for each microphone
    plt.figure(figsize=(12, 6))
    for i in range(composite_waveforms.shape[0]):
        plt.scatter(fft_frequencies, frequency_spectrum[i], label=f"Mic {i+1}")

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title("Frequency Spectrum of Composite Waveforms (Positive Frequencies)")
    plt.legend()
    plt.xlim(-400, 400)
    plt.grid()
    plt.show()


def experiment_3(plot=False, cosine=False):
    array_size = 8
    microphone_spacing = 0.1
    x_mics, y_mics = initialize_linear_array(array_size, microphone_spacing)

    # Frequency of interest
    f0 = 1_000 / 4

    sources = [
        SoundSource(distance=2.0, angle=np.pi / 4, frequency=f0, amplitude=1.0),
        SoundSource(distance=2.0, angle=np.pi / 3, frequency=f0, amplitude=0.8),
    ]

    t, composite_waveforms, individual_waveforms, delays_dict = (
        simulate_waveforms_multiple_sources(x_mics, y_mics, sources, cosine=cosine)
    )

    if plot:
        plot_overview(
            x_mics, y_mics, sources, t, composite_waveforms, individual_waveforms
        )

    # Fourier Transform of the composite waveforms
    N = len(t)
    frequency_spectrum = fft(composite_waveforms, axis=1)  # i.e. Y(omega)
    fft_frequencies = fftfreq(N, d=t[1] - t[0])

    omega = 2 * np.pi * f0

    # C(omega) matrix for a specific angle frequency
    num_mics = array_size
    num_sources = len(sources)
    # Bear in mind this is C_omega_0, C would be a cube
    C = np.zeros((num_mics, num_sources), dtype=complex)

    for i in range(num_mics):
        for j, source in enumerate(sources):
            C[i, j] = np.exp(-1j * omega * delays_dict[j + 1][i])
            # TODO add attenuation as 1/d_{ij}

    # the time signal at the source, before any phaseshifting
    f = np.cos if cosine else np.sin
    x = np.array([source.amplitude * f(omega * t) for source in sources])
    X = fft(x, axis=1)

    idx = np.argmin(np.abs(fft_frequencies - f0))

    Y_f0 = frequency_spectrum[:, idx]
    X_f0 = X[:, idx]

    Y_pred_f0 = C.dot(X_f0)

    # print(f"{Y_pred_f0=}")
    # print(f"{Y_f0=}")
    print(f"{C.shape=}")

    plot_complex_matrix(C)
    # plot_complex_matrix(Y_f0)
    plot_equation(Y_pred_f0, C, X_f0)

    # TODO Compressed sensing algoritme
    # TODO invertere


def experiment_4(plot=False):
    # really simple dft

    array_size = 1
    microphone_spacing = 0.1
    x_mics, y_mics = initialize_linear_array(array_size, microphone_spacing)

    # Frequency of interest
    f0 = 100
    sampling_rate = 10 * f0
    duration = 2 / f0
    N = int(sampling_rate * duration)

    sources = [
        SoundSource(distance=20, angle=0, frequency=f0, amplitude=1.0),
    ]

    t, composite_waveforms, individual_waveforms, delays_dict = (
        simulate_waveforms_multiple_sources(
            x_mics, y_mics, sources, sampling_rate=sampling_rate, duration=duration
        )
    )

    # MODIFY from here on for 6
    tau = delays_dict[0][0]

    # 1. Time-domain delay
    x = np.sin(2 * np.pi * f0 * t)
    y = composite_waveforms[0]

    # 2. frequency domain approach
    X = fft(x)
    freqs = fftfreq(N, d=1 / sampling_rate)
    C = np.exp(-2j * np.pi * freqs * tau)
    Y_pred = C * X
    y_pred = ifft(Y_pred)

    # 3. Time-delay approach
    Y_fft = fft(y)
    freq_fft = fftfreq(len(y), d=1 / sampling_rate)

    if plot:
        plot_overview(
            x_mics, y_mics, sources, t, composite_waveforms, individual_waveforms
        )
        # plot_equation(
        #     Y_dft,
        #     F,
        #     y,
        #     titles=("Y", "F", "y"),
        #     polar=True,
        #     ratios=[1, 10, 1],
        # )

        plot_equation(Y_fft, Y_pred, X, ["Y_fft", "Y_pred ", "X"], polar=True)
        y_pred = ifft(Y_pred)
        plot_time_and_frequency(y_pred, Y_pred, t, freq_fft, "Y_pred")
        plot_time_and_frequency(y, Y_fft, t, freq_fft, "Y from tau")

        # Plot time domain comparison
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.plot(t, y, label="y (time delay)", linewidth=2)
        plt.plot(t, y_pred.real, "--", label="y_pred (ifft(C * X))", linewidth=2)
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.title("Time Domain Comparison")
        plt.legend()


def experiment_5():
    # Parameters
    sampling_rate = 100  # Sampling frequency in Hz
    duration = 1.0  # Duration in seconds
    N = int(sampling_rate * duration)
    t = np.linspace(0, duration, N, endpoint=False)
    f0 = 10  # 0  # hz

    tau = np.float64(0.05830903790087463)  # time delay in seconds (e.g., 50 ms)

    # 1. Time-domain delay:
    # Original signal x(t) = sin(2πt)
    x = np.sin(2 * np.pi * f0 * t)
    # y(t) = sin(2π(t - tau))
    y = np.sin(2 * np.pi * f0 * (t - tau))

    # 2. Frequency domain approach:
    # Compute Fourier transform of x
    X = fft(x)
    # Frequency axis (Hz)
    freqs = fftfreq(N, d=1 / sampling_rate)
    # Compute the phase shift for each frequency: C(f) = exp(-2πi * f * tau)
    C = np.exp(-2j * np.pi * freqs * tau)
    # Apply the delay in the frequency domain: Y_pred = C * X
    Y_pred = C * X
    # Inverse FFT to recover the time-domain signal from Y_pred
    y_pred = ifft(Y_pred)

    # Also compute Y from the time-delayed y for comparison
    Y = fft(y)

    # Plot time domain comparison
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(t, y, label="y (time delay)", linewidth=2)
    plt.plot(t, y_pred.real, "--", label="y_pred (ifft(C * X))", linewidth=2)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Time Domain Comparison")
    plt.legend()

    # Plot frequency domain comparison (magnitude spectra)
    plt.subplot(2, 1, 2)
    plt.plot(freqs, np.abs(Y), label="|Y| (fft of y)", linewidth=2)
    plt.plot(freqs, np.abs(Y_pred), "--", label="|Y_pred| (C * X)", linewidth=2)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.title("Frequency Domain Comparison")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Report maximum absolute errors between the two methods
    time_error = np.max(np.abs(y - y_pred.real))
    freq_error = np.max(np.abs(Y - Y_pred))
    print("Max absolute error in time domain:", time_error)
    print("Max absolute error in frequency domain:", freq_error)


def experiment_6(plot=False):
    # TODO multiple sources and add noise

    num_mics = 10
    radius = 1
    x_mics, y_mics = initialize_circular_array(num_mics, radius)

    # Frequency of interest
    f0 = 100

    sources = [
        SoundSource(distance=10, angle=a, frequency=f0, amplitude=1.0)
        for a in [i * 0.2 for i in range(10)]
    ]
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
    x = np.array([np.sin(2 * np.pi * source.frequency * t) for source in sources])
    y = composite_waveforms

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

    selected_frequency = 1
    plot_C_pinv(C, selected_frequency)

    if plot:
        plot_overview(
            x_mics, y_mics, sources, t, composite_waveforms, individual_waveforms
        )
        # plot_matrix_3D(C)
        plot_equation(
            Y_pred[:, 1],
            C[:, :, 1],
            X[:, 1],
            titles=["Y_pred_f0 ", "C_f0 ", "X_f0"],
            polar=True,
            show_values=True,
            ratios=(1, 10, 1),
        )
        plot_equation(
            Y,
            Y_pred,
            np.array([[1]]),
            titles=["Y_fft", "Y_pred=C*X", ""],
            ratios=[1, 1, 0],
        )

        # Reconstructing X
        plot_equation(X_pred, X, X, titles=("X_pred", "X", ""), ratios=(1, 1, 0))


if __name__ == "__main__":
    # experiment_4(True)
    experiment_6(False)
