import numpy as np

# ------- X,Y matrix/vector representations -------


def _as_column_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector)
    if vector.ndim == 1:  # (S,) or (M,)
        return vector.reshape(-1, 1)
    if vector.ndim == 2 and vector.shape[1] == 1:  # (S, 1) or (M, 1)
        return vector
    raise ValueError(f"Expected a vector or column vector, got shape {vector.shape}")


def matrix_to_frequency_major_vector(matrix: np.ndarray) -> np.ndarray:
    """X (S, F) -> [ X_1[0], X_2[0], ..., X_S[F-1] ] (SF, 1)"""
    matrix = np.asarray(matrix)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {matrix.shape}")
    # F = Fortran order, which is column-major
    return matrix.reshape(-1, order="F").reshape(-1, 1)


def frequency_major_vector_to_matrix(
    vector: np.ndarray, num_rows: int, num_freqs: int
) -> np.ndarray:
    """[ X_1[0], X_2[0], ..., X_S[F-1] ] (SF, 1) -> X (S, F) matrix"""
    vector = _as_column_vector(vector)
    expected_size = num_rows * num_freqs
    if vector.shape[0] != expected_size:
        raise ValueError(
            f"Expected vector of length {expected_size}, got {vector.shape[0]}"
        )
    # (SF, 1) -> (SF,) -> (S, F)
    return vector.ravel().reshape((num_rows, num_freqs), order="F")


def complex_matrix_to_augmented_real_vector(matrix: np.ndarray) -> np.ndarray:
    """(S, F) -> (2SF, 1)"""
    vector = matrix_to_frequency_major_vector(matrix)
    return np.vstack([vector.real, vector.imag])


def augmented_real_vector_to_complex_matrix(
    vector_real: np.ndarray, num_rows: int, num_freqs: int
) -> np.ndarray:
    """(2SF, 1) -> (S, F)"""
    vector_real = _as_column_vector(vector_real)
    num_complex = num_rows * num_freqs
    if vector_real.shape[0] != 2 * num_complex:
        raise ValueError(
            f"Expected augmented-real vector of length {2 * num_complex}, "
            f"got {vector_real.shape[0]}"
        )
    vector_complex = vector_real[:num_complex] + 1j * vector_real[num_complex:]
    return frequency_major_vector_to_matrix(vector_complex, num_rows, num_freqs)


# ------- A, B block matrices -------


def mixing_tensor_to_frequency_major_matrix(tensor: np.ndarray) -> np.ndarray:
    """A (M, S, F) -> A_big (MF, SF), block-diagonal in frequency-major form."""
    tensor = np.asarray(tensor)
    if tensor.ndim != 3:
        raise ValueError(f"Expected a 3D tensor, got shape {tensor.shape}")

    num_mics, num_sources, num_freqs = tensor.shape
    block_matrix = np.zeros(
        (num_mics * num_freqs, num_sources * num_freqs),
        dtype=tensor.dtype,
    )

    for f in range(num_freqs):
        row_slice = slice(f * num_mics, (f + 1) * num_mics)
        col_slice = slice(f * num_sources, (f + 1) * num_sources)
        block_matrix[row_slice, col_slice] = tensor[:, :, f]

    return block_matrix


def complex_matrix_to_augmented_real_matrix(matrix: np.ndarray) -> np.ndarray:
    """Complex 2D operator M -> [[Re(M), -Im(M)], [Im(M), Re(M)]]."""
    matrix = np.asarray(matrix)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {matrix.shape}")
    return np.block([[matrix.real, -matrix.imag], [matrix.imag, matrix.real]])


# ------ Normalization of input shapes -------


def normalize_frequency_system(
    A: np.ndarray,
    Y: np.ndarray,
    X: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, bool]:
    """
    Normalize to the canonical complex multi-frequency shapes:
        A -> (M, S, F)
        Y -> (M, F)
        X -> (S, F) if provided

    Returns:
        A_norm, Y_norm, X_norm, was_single_frequency
    """
    A = np.asarray(A)
    Y = np.asarray(Y)
    X_norm = None if X is None else np.asarray(X)

    if A.ndim == 2:
        A = A[:, :, None]
        was_single_frequency = True
    elif A.ndim == 3:
        was_single_frequency = A.shape[2] == 1
    else:
        raise ValueError(f"Expected A with ndim 2 or 3, got shape {A.shape}")

    num_mics, num_sources, num_freqs = A.shape

    if Y.ndim == 1:
        if num_freqs != 1:
            raise ValueError(
                f"1D Y is only valid for single-frequency systems, got A.shape={A.shape}"
            )
        if Y.shape[0] != num_mics:
            raise ValueError(f"Expected Y of length {num_mics}, got shape {Y.shape}")
        Y = Y.reshape(num_mics, 1)
    elif Y.ndim == 2:
        if Y.shape != (num_mics, num_freqs):
            raise ValueError(
                f"Expected Y with shape {(num_mics, num_freqs)}, got {Y.shape}"
            )
    else:
        raise ValueError(f"Expected Y with ndim 1 or 2, got shape {Y.shape}")

    if X_norm is not None:
        if X_norm.ndim == 1:
            if num_freqs != 1:
                raise ValueError(
                    f"1D X is only valid for single-frequency systems, got A.shape={A.shape}"
                )
            if X_norm.shape[0] != num_sources:
                raise ValueError(
                    f"Expected X of length {num_sources}, got shape {X_norm.shape}"
                )
            X_norm = X_norm.reshape(num_sources, 1)
        elif X_norm.ndim == 2:
            if X_norm.shape != (num_sources, num_freqs):
                raise ValueError(
                    f"Expected X with shape {(num_sources, num_freqs)}, got {X_norm.shape}"
                )
        else:
            raise ValueError(f"Expected X with ndim 1 or 2, got shape {X_norm.shape}")

    return A, Y, X_norm, was_single_frequency


if __name__ == "__main__":
    # X (S=2, F=3), X_j[k] = jk + i*jk
    X = np.array(
        [[11 + 11j, 12 + 12j, 13 + 13j], [21 + 21j, 22 + 22j, 23 + 23j]], dtype=complex
    )

    A = np.zeros((2, 3, 2), dtype=int)  # (M=2, S=3, F=2)
    for i in range(2):
        for j in range(3):
            for k in range(2):
                A[i, j, k] = 100 * (i + 1) + 10 * (j + 1) + (k + 1)

    assert np.allclose(
        matrix_to_frequency_major_vector(X),
        np.array([11 + 11j, 21 + 21j, 12 + 12j, 22 + 22j, 13 + 13j, 23 + 23j]).reshape(
            -1, 1
        ),  # (6, 1)
    )

    print("")
    assert np.allclose(
        complex_matrix_to_augmented_real_vector(X),
        np.array([11, 21, 12, 22, 13, 23, 11, 21, 12, 22, 13, 23]).reshape(
            -1, 1
        ),  # (12, 1)
    )

    assert np.allclose(
        mixing_tensor_to_frequency_major_matrix(A),
        np.array(
            [
                [111, 121, 131, 0, 0, 0],
                [211, 221, 231, 0, 0, 0],
                [0, 0, 0, 112, 122, 132],
                [0, 0, 0, 212, 222, 232],
            ]
        ),
    )

    # Test algebra
    X = np.array(
        [
            [11, 12],
            [21, 22],
            [31, 32],
        ],
        dtype=int,
    )
    Y = np.einsum("msf,sf->mf", A, X)  # (M, F)
    Y_direct = matrix_to_frequency_major_vector(Y)
    A_big = mixing_tensor_to_frequency_major_matrix(A)
    Y_from_blocks = A_big @ matrix_to_frequency_major_vector(X)

    assert np.allclose(Y_from_blocks, Y_direct)

    for matrix in (X, Y):
        vector = matrix_to_frequency_major_vector(matrix)
        vector_real = complex_matrix_to_augmented_real_vector(matrix)
        assert np.allclose(
            frequency_major_vector_to_matrix(vector, *matrix.shape), matrix
        )
        assert np.allclose(
            augmented_real_vector_to_complex_matrix(vector_real, *matrix.shape),
            matrix,
        )

    print("representations.py round-trip tests passed")
