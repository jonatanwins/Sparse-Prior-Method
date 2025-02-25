import numpy as np
import matplotlib.pyplot as plt
from plotting import plot_complex_matrix, plot_equation

# ------------------------
# Making the DFT from scratch
# ------------------------


def DFT(x):
    N = len(x)
    omega = np.exp(-1j * 2 * np.pi / N)
    F = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            F[i, j] = omega ** (i * j)
    # plot_complex_matrix(F)

    X = F @ x
    plot_equation(X, F, x, titles=("X", "F", "x"))


def experiment_0():
    sampling_rate = 5
    frequency = 1
    duration = 1
    amplitude = 1
    t = np.linspace(0, duration, int(sampling_rate * duration))
    f = amplitude * np.sin(2 * np.pi * frequency * t)
    smooth_t = np.linspace(0, duration, 100 * int(sampling_rate * duration))
    smooth_f = amplitude * np.sin(2 * np.pi * frequency * smooth_t)

    fig, (time_ax, freq_ax) = plt.subplots(2, 1)
    time_ax.plot(smooth_t, smooth_f, color="gold")
    time_ax.scatter(t, f)

    dft = DFT(f)


if __name__ == "__main__":
    experiment_0()
    plt.show()
