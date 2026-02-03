import numpy as np

# Convert to complex vectors and matrices to augmented real form for optimization
def to_real_augmented(x_complex: np.ndarray) -> np.ndarray:
    # X is a row vector, either (N,) or (1,N), or a column vector (N, 1)
    if x_complex.ndim == 1 or (x_complex.ndim == 2 and x_complex.shape[1] == 1):
        x_real = np.array([x_complex.real, x_complex.imag]).reshape(-1, 1)
    else:
        x_real = np.block(
            [[x_complex.real, -x_complex.imag], [x_complex.imag, x_complex.real]]
        )
    return x_real


def from_real_augmented(x_real: np.ndarray) -> np.ndarray:
    n = x_real.shape[0] // 2
    return (x_real[:n] + 1j * x_real[n:]).reshape(-1, 1)