import numpy as np
import matplotlib.pyplot as plt
from plotting import plot_all
from scipy.fft import fft, ifft, fftfreq, fftshift


# Constants
speed_of_sound = 343  # m/s


def initialize_linear_array(array_size, microphone_spacing):
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
    )  # angle frequency times respective delaya

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
        individual_waveforms[idx + 1] = waveform
        # TODO maybe cut the Source in the name
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

    plot_all(x_mics, y_mics, sources, t, composite_waveforms, individual_waveforms)


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

    plot_all(x_mics, y_mics, sources, t, composite_waveforms, individual_waveforms)

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


def experiment_3():
    array_size = 8
    microphone_spacing = 0.1
    x_mics, y_mics = initialize_linear_array(array_size, microphone_spacing)

    frequency = 100

    sources = [
        SoundSource(distance=2.0, angle=np.pi / 4, frequency=frequency, amplitude=1.0),
        SoundSource(distance=2.0, angle=np.pi / 3, frequency=frequency, amplitude=0.8),
    ]

    t, composite_waveforms, individual_waveforms, delays_dict = (
        simulate_waveforms_multiple_sources(x_mics, y_mics, sources)
    )

    # plot_all(x_mics, y_mics, sources, t, composite_waveforms, individual_waveforms)

    # Fourier Transform of the composite waveforms
    N = len(t)
    frequency_spectrum = fft(composite_waveforms, axis=1)
    fft_frequencies = fftfreq(N, d=t[1] - t[0])

    # Frequency of interest
    f0 = 100
    omega = 2 * np.pi * f0

    # C(omega) matrix
    num_mics = array_size
    num_sources = len(sources)
    # print(f'{x_mics=}, {x_mics.shape=}')
    C = np.zeros((num_mics, num_sources), dtype=complex)

    print(delays_dict[1])

    # TODO imaginary unit
    # Define imaginary unit
    j_imag = 1j

    for i in range(num_mics):
        for j, source in enumerate(sources):
            C[i, j] = np.exp(-j_imag * source.frequency * delays_dict[j + 1][i])
            # TODO add attenuation as 1/d_{ij}

    # Plot the frequency spectrum for each microphone
    plt.figure(figsize=(12, 6))
    for i in range(composite_waveforms.shape[0]):
        plt.plot(fft_frequencies, frequency_spectrum[i], label=f"Mic {i+1}")

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title("Frequency Spectrum of Composite Waveforms")
    plt.legend()
    plt.xlim(-400, 400)
    plt.grid()
    plt.show()


if __name__ == "__main__":
    experiment_3()
