import numpy as np
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt
from typeguard import typechecked


def complex_to_rgb(matrix):
    """
    Convert a complex matrix to an RGB image using an HSV mapping:
      - Hue: phase (normalized from -pi to pi → 0 to 1)
      - Saturation: fixed at 1.
      - Value: magnitude (normalized to 0-1), but set to 0 (black) for small magnitudes regardless of angle.
    """
    # Ensure matrix is two-dimensional.
    mat = np.atleast_2d(matrix)

    # Compute phase and magnitude.
    phase = np.angle(mat)  # Range: [-pi, pi]
    magnitude = np.abs(mat)

    # Normalize phase to [0,1].
    norm_phase = (phase + np.pi) / (2 * np.pi)
    # Normalize magnitude to [0,1] using its maximum.
    max_mag = magnitude.max() if magnitude.max() != 0 else 1
    norm_mag = magnitude / max_mag

    # Build an HSV image.
    hsv = np.zeros(mat.shape + (3,))
    hsv[..., 0] = norm_phase  # Hue from phase.
    hsv[..., 1] = 1  # Full saturation.
    hsv[..., 2] = norm_mag  # Value from normalized magnitude.

    # Set values with small magnitude to black (Value = 0) regardless of angle.
    # Use a threshold of 0.0001 for magnitude, matching the display logic elsewhere.
    threshold = 0.0001
    small_mask = magnitude < threshold
    hsv[small_mask, 2] = 0  # Set Value to 0 for small magnitudes.

    # Convert HSV to RGB.
    rgb = hsv_to_rgb(hsv)
    return rgb


def plot_complex_matrix_on_ax(
    ax, matrix, title="", show_values=True, polar=False, font_size=8
):
    """
    Plot a complex matrix on the provided Axes object using an HSV mapping.

    The cell color is determined by:
      - Hue: phase of the complex number.
      - Value: magnitude.

    Optionally, the numerical complex value is overlaid.
    """
    # Expand if single column vector
    # not sure if this causes problems
    # essential that it is before complex to rgb
    if np.ndim(matrix) < 2:
        matrix = np.expand_dims(matrix, axis=1)

    # Convert the complex matrix to an RGB image.
    rgb = complex_to_rgb(matrix)
    ax.imshow(rgb, interpolation="none", aspect="auto")
    ax.set_title(title)
    # set the title size
    ax.title.set_fontsize(font_size + 20)
    ax.axis("off")

    if show_values:
        n, m = matrix.shape
        for i in range(n):
            for j in range(m):
                if abs(matrix[i, j]) < 0.01:
                    value_str = "0"
                elif polar:
                    # Compute magnitude and phase (in degrees)
                    r = np.abs(matrix[i, j])
                    if polar == "absolute":
                        value_str = f"{r:.1f}"
                    else:
                        theta = np.angle(matrix[i, j], deg=True)
                        if theta == 180:
                            value_str = f"{-r:.1f}"
                        elif theta == 0:
                            value_str = f"{r:.1f}"
                        else:
                            value_str = f"{r:.1f}∠{theta:.0f}°"
                else:
                    value_str = f"{matrix[i, j].real:.1f}{matrix[i, j].imag:+.1f}j"
                ax.text(
                    j,
                    i,
                    value_str,
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=font_size,
                )


@typechecked
def plot_matrices(
    matrices: list, titles: list[str] | None = None, polar=False, font_size=8
):
    """
    Plot multiple complex matrices in an automatically arranged grid.

    Args:
        matrices: list of complex matrices to plot
        titles: list of titles for each subplot
        polar: whether to display values in polar form
        font_size: font size for overlaid text
    """
    num_matrices = len(matrices)

    if titles is None:
        titles = [""] * num_matrices

    # Calculate grid dimensions for a roughly square layout
    if num_matrices == 1:
        nrows, ncols = 1, 1
    elif num_matrices == 2:
        nrows, ncols = 1, 2
    else:
        ncols = int(np.ceil(np.sqrt(num_matrices)))
        nrows = int(np.ceil(num_matrices / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))

    # Flatten axes array for easier indexing
    if num_matrices == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()

    # Plot each matrix
    for i in range(num_matrices):
        plot_complex_matrix_on_ax(
            axes[i], matrices[i], title=titles[i], polar=polar, font_size=font_size
        )

    # Hide unused subplots
    for i in range(num_matrices, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
