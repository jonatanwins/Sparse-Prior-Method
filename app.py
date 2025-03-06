import numpy as np
import matplotlib.pyplot as plt
from plotting import (
    plot_overview,
    plot_Y_comparison,
    plot_complex_matrix,
    plot_equation,
    plot_time_and_frequency,
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
        individual_waveforms[idx + 1] = waveform
        delays_dict[idx + 1] = delays

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
    f0 = 1
    sampling_rate = 10

    sources = [
        SoundSource(distance=speed_of_sound / 5, angle=0, frequency=f0, amplitude=1.0),
    ]

    t, composite_waveforms, individual_waveforms, delays_dict = (
        simulate_waveforms_multiple_sources(
            x_mics, y_mics, sources, sampling_rate=sampling_rate
        )
    )

    # Our time signal
    y = composite_waveforms[0]
    # There is only one composite waveform and it needs to be unpacked
    dft_Y, dft_freq = DFT(composite_waveforms[0])
    # It is crucial that you retrieve the fftfreqs with 1/sampling_rate, otherwise 1hz is assumed
    fft_Y, fft_freq = fft(y), fftfreq(len(y), d=1 / sampling_rate)

    F = DFT_matrix(y)
    if plot:
        plot_overview(
            x_mics, y_mics, sources, t, composite_waveforms, individual_waveforms
        )
        plot_equation(
            dft_Y, F, y, titles=("Y", "F", "y"), show_values=True, ratios=[1, 10, 1]
        )
        plot_time_and_frequency(y, dft_Y, t, dft_freq)
        plot_time_and_frequency(y, fft_Y, t, fft_freq)

    idx = np.argmin(np.abs(dft_freq - f0))

    # TODO do the fourier mixing model calculations and visualize,
    # this should let you compare the direct approach (fft of Y) with the computed approach CX
    # Step 1: y -> Y and x -> CX
    # Constructing C_omega_0 for the time delay
    tau = delays_dict[1]
    C = np.exp(-2j * np.pi * tau)
    x = np.sin(2 * np.pi * t)
    X, spectrum_x = DFT(x, sampling_rate=sampling_rate)

    Y_pred = C * X
    # BUG y_pred is complex
    y_pred = ifft(Y_pred)

    print(f"{abs(y_pred[0])=}")
    print(f"{y[0]=}")
    # plot_time_and_frequency(
    #     y,
    #     y_pred,
    #     time_axis=t,
    #     frequency_axis=t,
    #     title="y top and y ifft from CX bottom",
    #     absolute=False,
    # )

    # plot_time_and_frequency(
    #     y_pred, Y_pred, t, spectrum_x, title="Inverse fourier Y from CX"
    # )

    # plot_time_and_frequency(x, abs(X), time_axis=t, frequency_axis=spectrum_x)

    # plot_equation(
    #     X, DFT_matrix(x), x, titles=["X", "F", "x"], ratios=[1, 10, 1], polar=True
    # )
    # plot_equation(fft_Y, C, X, polar=True)
    # plot_equation(C * X, dft_Y, fft_Y, titles=["CX", " DFT_Y ", " FFT_Y "], polar=True)
    # plot_equation(C * X, C, X, titles=["C*X", "C", "X"], polar=True)


if __name__ == "__main__":
    B = np.array(
        [
            [-10, 3 + 3j, 10],
            [1j, 1 - 1j, 5 + 5j],
            [2j + 8, 8j - 2, 6 + 4j],
        ]
    )
    # plot_complex_matrix_hsv(B, show_values=True)

    experiment_4(plot=False)
