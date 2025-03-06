import numpy as np
import matplotlib.pyplot as plt
from plotting import plot_complex_matrix, plot_equation
from scipy.fft import fft, fftfreq

# ------------------------
# Making the DFT from scratch
# ------------------------


def DFT(x, sampling_rate=False):
    N = len(x)
    if not sampling_rate:
        sampling_rate = N

    omega = np.exp(-1j * 2 * np.pi / N)
    F = np.zeros((N, N), dtype=complex)
    for i in range(N):
        for j in range(N):
            F[i, j] = omega ** (i * j)

    X = F @ x

    # This depends on the spacing which is 1/sampling_rate
    frequencies = np.array([k for k in range(N)]) * sampling_rate / N

    return X, frequencies


def DFT_matrix(x):
    N = len(x)
    k = np.arange(N)
    F = np.exp(-2j * np.pi * np.outer(k, k) / N)
    return F


def experiment_0():
    sampling_rate = 20
    frequency = 1
    duration = 1
    amplitude = 1
    t = np.linspace(0, duration, int(sampling_rate * duration))
    x = amplitude * np.sin(2 * np.pi * frequency * t)
    smooth_t = np.linspace(0, duration, 100 * int(sampling_rate * duration))
    smooth_f = amplitude * np.sin(2 * np.pi * frequency * smooth_t)

    fig, (time_ax, freq_ax) = plt.subplots(2, 1)
    time_ax.plot(smooth_t, smooth_f, color="gold")
    time_ax.scatter(t, x)

    comp_X, comp_freq = DFT(x)
    X, freq = fft(x), fftfreq(len(x))

    print(f"{comp_freq=}")
    print(f"{freq=}")

    plot_equation(X, comp_X, x, titles=("X", "F", "x"), show_values=False)


if __name__ == "__main__":
    # TODO lage IDFT og reversere signal, både som invers og ikke
    experiment_0()
    # plt.show()
