import numpy as np
from scipy.fft import fft, fftfreq, ifft
from ..geometry.arrays import circular_array, linear_array, arc_array
from ..geometry.SoundSource import SoundSource
from ..constants import SPEED_OF_SOUND
from .Simulation import Simulation

# ── 1. Geometry ──────────────────────────────────────────────────────────────


def construct_geometry(
    array_type: str,
    num_mics: int,
    num_sources: int,
    mic_radius: float | None = None,
    mic_angle_start: float = 0.0,
    mic_angle_span: float = 2 * np.pi,
    source_distance: float | None = None,
    source_angle_start: float = 0.0,
    source_angle_span: float = 2 * np.pi,
):
    """Create mic array (M x 2) and source list (S,)."""
    array_type = array_type.lower()

    if mic_radius is None or source_distance is None:
        raise ValueError("mic_radius and source_distance must be provided")

    # 1. Microphones
    if array_type == "linear":
        mics = linear_array(num_mics, mic_radius)
    elif np.isclose(mic_angle_span, 2 * np.pi):
        mics = circular_array(num_mics, mic_radius)
    else:
        mics = arc_array(num_mics, mic_radius, mic_angle_start, mic_angle_span)

    # 2. Sources
    if num_sources == 1:
        source_angles = np.array([source_angle_start])
    # full circle
    elif np.isclose(source_angle_span, 2 * np.pi):
        source_angles = source_angle_start + (
            source_angle_span / num_sources
        ) * np.arange(num_sources)
    # sector
    else:
        source_angles = np.linspace(
            source_angle_start,
            source_angle_start + source_angle_span,
            num_sources,
            endpoint=True,
        )

    # convert angles to SoundSource objects
    sources = [
        SoundSource(
            distance=source_distance,
            angle=float(angle),
            time_series=None,
        )
        for angle in source_angles
    ]
    return mics, sources


# ── 2. Select active (non-mute) sources ─────────────────────────────────────


def select_active_sources(
    sources: list[SoundSource],
    num_active_sources: int,
    rng: np.random.Generator | None = None,
) -> list[int]:
    """Randomly pick the defined number of active source indices."""
    if rng is None:
        rng = np.random.default_rng()
    return rng.choice(len(sources), size=num_active_sources, replace=False).tolist()


# ── 3. Generate time-domain signals ─────────────────────────────────────────


def generate_signals(
    sources: list[SoundSource],
    active_indices: list[int],
    sampling_rate: float,
    duration: float,
    mode: str = "sine",
    frequencies: list[list[float]] | None = None,
    phases: list[float] | None = None,
    rng: np.random.Generator | None = None,
    amplitude: float = 1.0,
):
    """
    Assign time_series to each source. Muted sources get zeros.

    Args:
        mode: "sine" — sum-of-sines per source (requires frequencies, phases).
              "noise" — white Gaussian noise.
        frequencies: (S,) list of freq-lists, e.g. [[100, 200], [150]].
        phases: (S,) list of starting phases.
    """
    N = int(sampling_rate * duration)
    t = np.linspace(0, duration, N, endpoint=False)

    if rng is None:
        rng = np.random.default_rng()

    for idx, src in enumerate(sources):
        if idx not in active_indices:
            src.time_series = np.zeros(N)
            continue

        if mode == "sine":
            if frequencies is None or phases is None:
                raise ValueError("sine mode requires frequencies and phases")
            src.time_series = np.sum(
                [np.sin(2 * np.pi * f * t + phases[idx]) for f in frequencies[idx]],
                axis=0,
            )
        elif mode == "noise":
            src.time_series = rng.standard_normal(N)
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'sine' or 'noise'.")

        # Normalize each source to the same RMS amplitude (before noise is added).
        rms = np.sqrt(np.mean(src.time_series**2))
        if rms > 0:
            src.time_series = amplitude * src.time_series / rms

    return sources, t


# -- 3b. Generate frequency-domain signals ─────────────────────────────────


def generate_frequency_domain_signals(
    sources: list[SoundSource],
    active_indices: list[int],
    sampling_rate: float,
    duration: float,
    component_amplitude: float = 1.0,
    magnitude_jitter: float = 0.0,
    rng: np.random.Generator | None = None,
):
    """
    Assign artificial source spectra directly in the frequency domain.

    Active sources get nearly flat positive-frequency magnitudes with random
    phase. The spectrum is mirrored to be conjugate-symmetric so that the IFFT
    is real-valued. Inactive sources get zero spectrum.

    Args:
        component_amplitude: Magnitude of each active positive-frequency bin.
        magnitude_jitter: Relative perturbation in [0, 1). Example: 0.1 gives
            magnitudes in [0.9, 1.1] * component_amplitude.
    """
    if not 0.0 <= magnitude_jitter < 1.0:
        raise ValueError("magnitude_jitter must satisfy 0 <= magnitude_jitter < 1")

    N = int(sampling_rate * duration)
    if N < 2:
        raise ValueError("sampling_rate * duration must give at least 2 samples")

    if rng is None:
        rng = np.random.default_rng()

    X = np.zeros((len(sources), N), dtype=complex)

    # FFT ordering:
    # even N: [0, 1, ..., N/2-1, -N/2, ..., -1]
    # odd N:  [0, 1, ..., (N-1)/2, -(N-1)/2, ..., -1]
    if N % 2 == 0:
        positive_bins = np.arange(1, N // 2)
        nyquist_bin = N // 2
    else:
        positive_bins = np.arange(1, (N + 1) // 2)
        nyquist_bin = None

    for idx, src in enumerate(sources):
        if idx not in active_indices:
            src.time_series = np.zeros(N)
            continue

        magnitudes = np.full(len(positive_bins), component_amplitude, dtype=float)
        if magnitude_jitter > 0.0:
            magnitudes *= 1.0 + rng.uniform(
                -magnitude_jitter, magnitude_jitter, size=len(positive_bins)
            )

        phases = rng.uniform(0.0, 2 * np.pi, size=len(positive_bins))
        X[idx, positive_bins] = magnitudes * np.exp(1j * phases)
        X[idx, -positive_bins] = np.conj(X[idx, positive_bins])

        # Nyquist must be real when N is even.
        if nyquist_bin is not None:
            nyquist_magnitude = component_amplitude
            if magnitude_jitter > 0.0:
                nyquist_magnitude *= 1.0 + rng.uniform(
                    -magnitude_jitter, magnitude_jitter
                )
            X[idx, nyquist_bin] = nyquist_magnitude * rng.choice([-1.0, 1.0])

        src.time_series = ifft(X[idx]).real

    return sources, X


# ── 4. Delay matrix ─────────────────────────────────────────────────────────


def compute_delay_matrix(mics: np.ndarray, sources: list[SoundSource]):
    """
    Returns:
        delays: (M x S) propagation delay from each source to each mic.
    """
    src_pos = np.array([s.get_position() for s in sources])  # (S x 2)
    # broadcasting: (M,1,2) - (1,S,2) -> (M,S)
    # This is very efficient but pretty confusing.
    # With np.newaxis we create an axis such that broadcasting can later
    # interpret the subtraction as (M,S,2) - (M,S,2), but the data is
    # not actually duplicated in memory.
    # We can then compute the distances directly by taking the 2-norm
    # over the last axis (the coordinates)
    diff = mics[:, np.newaxis, :] - src_pos[np.newaxis, :, :]  # TODO
    distances = np.linalg.norm(diff, axis=2)
    return distances / SPEED_OF_SOUND


# ── 5. Mixing matrix A ──────────────────────────────────────────────────────


def compute_mixing_matrix(delays: np.ndarray, freqs: np.ndarray):
    """
    Args:
        delays: (M x S)
        freqs:  (N,) frequency bins from fftfreq

    Returns:
        A: (M x S x N) complex mixing matrix, A[m,s,n] = exp(-i*2pi*f_n*tau_{ms})
    """
    # Combination of mics and sources for all frequencies
    # (M, S, 1) * (1, 1, N) -> (M, S, N)
    return np.exp(
        -2j * np.pi * delays[:, :, np.newaxis] * freqs[np.newaxis, np.newaxis, :]
    )


# -- 6. pseudoinverses ---
def moore_penrose_inverse(A: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Per-frequency Moore-Penrose recovery.

    Args:
        A: (M, S, F) complex mixing matrix
        Y: (M, F) complex observations

    Returns:
        X_mp: (S, F) complex estimate
    """
    M, S, F = A.shape
    if Y.shape != (M, F):
        raise ValueError(f"Expected Y shape {(M, F)}, got {Y.shape}")

    X_mp = np.zeros((S, F), dtype=complex)
    for k in range(F):
        # same as X_mp[:, k] = np.linalg.pinv(A[:, :, k]) @ Y[:, k]
        X_mp[:, k] = np.linalg.lstsq(A[:, :, k], Y[:, k], rcond=None)[0]
    return X_mp


def ridge_inverse(
    A: np.ndarray,
    Y: np.ndarray,
    noise_power: float,
) -> np.ndarray:
    """
    Per-frequency ridge / Tikhonov recovery with one shared regularization level.

    Args:
        A: (M, S, F) complex mixing matrix
        Y: (M, F) complex observations
        noise_power: scalar observation-space noise power used as lambda

    Returns:
        X_ridge: (S, F) complex estimate
    """
    M, S, F = A.shape
    if Y.shape != (M, F):
        raise ValueError(f"Expected Y shape {(M, F)}, got {Y.shape}")
    if noise_power < 0:
        raise ValueError("noise_power must be nonnegative")

    X_ridge = np.zeros((S, F), dtype=complex)
    I = np.eye(S, dtype=complex)

    # X_ridge = (A^H A + noise_power * I)^(-1) A^H Y  per frequency bin
    for k in range(F):
        Ak = A[:, :, k]
        # Solves (A^H A + noise_power * I) X = A^H Y for X, which is more stable than explicitly computing the inverse.
        X_ridge[:, k] = np.linalg.solve(
            Ak.conj().T @ Ak + noise_power * I,
            Ak.conj().T @ Y[:, k],
        )
    return X_ridge


def _simulate_from_spectrum(
    mics: np.ndarray,
    sources: list[SoundSource],
    active_indices: list[int],
    X: np.ndarray,
    sampling_rate: float,
    duration: float,
    freq_selector=None,
    seed: int = 0,
    sensor_snr_db: float | None = None,
    model_snr_db: float | None = None,
    inverse_method: str = "mp",
) -> Simulation:
    """
    Shared simulation core starting from source spectra X.

    Args:
        X: (S, N) complex source spectra over the full FFT grid.
    """
    rng = np.random.default_rng(seed)
    sensor_noise_seed = int(rng.integers(0, 2**31))
    model_noise_seed = int(rng.integers(0, 2**31))

    S = len(sources)
    M = len(mics)
    N = int(sampling_rate * duration)

    X = np.asarray(X)
    if X.shape != (S, N):
        raise ValueError(f"Expected X shape {(S, N)}, got {X.shape}")

    x_complex = ifft(X, axis=1)
    if not np.allclose(x_complex.imag, 0.0, atol=1e-10):
        raise ValueError(
            "Expected X to be conjugate-symmetric so that the time-domain signals are real."
        )
    x = x_complex.real

    for idx, src in enumerate(sources):
        src.time_series = x[idx]

    freqs = fftfreq(N, d=1.0 / sampling_rate)

    delays = compute_delay_matrix(mics, sources)
    A = compute_mixing_matrix(delays, freqs)

    # Y = A @ X  per frequency bin, vectorised over all bins:
    # A: (M, S, N), X: (S, N) -> Y_i = \sum_j A_{ij} * X_{j}
    # Same as Y[:, n] = A[:,:,n] @ X[:, n]
    Y_clean = np.einsum("msn,sn->mn", A, X)

    if freq_selector is not None:
        freq_mask = freq_selector(freqs)
        X = X[:, freq_mask]
        A = A[:, :, freq_mask]
        Y_clean = Y_clean[:, freq_mask]
        freqs = freqs[freq_mask]

    eta = np.zeros_like(Y_clean, dtype=complex)
    delta = np.zeros_like(X, dtype=complex)

    if sensor_snr_db is not None:
        sensor_noise_rng = np.random.default_rng(sensor_noise_seed)
        P_Y = np.mean(np.abs(Y_clean) ** 2)
        P_eta = P_Y * (10 ** (-sensor_snr_db / 10))
        eta = np.sqrt(P_eta / 2) * (
            sensor_noise_rng.standard_normal(Y_clean.shape)
            + 1j * sensor_noise_rng.standard_normal(Y_clean.shape)
        )

    if model_snr_db is not None:
        model_noise_rng = np.random.default_rng(model_noise_seed)
        P_X = np.mean(np.abs(X) ** 2)
        P_delta = P_X * (10 ** (-model_snr_db / 10))
        delta = np.sqrt(P_delta / 2) * (
            model_noise_rng.standard_normal(X.shape)
            + 1j * model_noise_rng.standard_normal(X.shape)
        )

    Y = Y_clean + np.einsum("msn,sn->mn", A, delta) + eta

    if inverse_method == "mp":
        X_pinv = moore_penrose_inverse(A, Y)
    elif inverse_method == "ridge":
        if model_snr_db is None and sensor_snr_db is not None:
            noise_power = P_eta
            X_pinv = ridge_inverse(A, Y, noise_power)
        else:
            raise NotImplementedError(
                "Ridge inverse not implemented for model noise or no sensor noise. "
                "i.e. only works if model_snr_db is None and sensor_snr_db is not None"
            )
    else:
        raise ValueError(
            f"Unknown inverse_method '{inverse_method}', choose 'mp' or 'ridge'."
        )

    return Simulation(
        Y=Y,
        A=A,
        X=X,
        X_pinv=X_pinv,
        x=x,
        Y_clean=Y_clean,
        eta=eta,
        delta=delta,
        freqs=freqs,
        sources=sources,
        mics=mics,
        active_indices=active_indices,
        sampling_rate=sampling_rate,
        duration=duration,
    )


# ── 6. Simulate: full pipeline ──────────────────────────────────────────────


def simulate_from_time_domain(
    mics: np.ndarray,
    sources: list[SoundSource],
    active_indices: list[int],
    sampling_rate: float,
    duration: float,
    freq_selector=None,
    seed: int = 0,
    sensor_snr_db: float | None = None,
    model_snr_db: float | None = None,
    inverse_method: str = "mp",
) -> Simulation:
    """
    Time-domain simulation path: build x first, then FFT to X.

    Frequency-domain simulation: Y = A @ X.

    Dimensions (M=mics, S=sources, N=time samples):
        x  : (S x N)  time-domain source signals
        X  : (S x N)  FFT of x
        A  : (M x S x N) mixing matrix per freq bin
        Y  : (M x N)  observed spectra
    """
    x = np.array([src.time_series for src in sources])  # (S, N)
    X = np.array(fft(x, axis=1))  # (S, N)

    return _simulate_from_spectrum(
        mics=mics,
        sources=sources,
        active_indices=active_indices,
        X=X,
        sampling_rate=sampling_rate,
        duration=duration,
        freq_selector=freq_selector,
        seed=seed,
        sensor_snr_db=sensor_snr_db,
        model_snr_db=model_snr_db,
        inverse_method=inverse_method,
    )


def simulate_from_frequency_domain(
    mics: np.ndarray,
    sources: list[SoundSource],
    active_indices: list[int],
    X: np.ndarray,
    sampling_rate: float,
    duration: float,
    freq_selector=None,
    seed: int = 0,
    sensor_snr_db: float | None = None,
    model_snr_db: float | None = None,
    inverse_method: str = "mp",
) -> Simulation:
    """
    Frequency-domain simulation path: start from a pre-built source spectrum X.
    """
    return _simulate_from_spectrum(
        mics=mics,
        sources=sources,
        active_indices=active_indices,
        X=X,
        sampling_rate=sampling_rate,
        duration=duration,
        freq_selector=freq_selector,
        seed=seed,
        sensor_snr_db=sensor_snr_db,
        model_snr_db=model_snr_db,
        inverse_method=inverse_method,
    )


# ── 7. Quick simulation ─────────────────────────────────────────────────────


def _random_sine_params(
    num_sources, sampling_rate, rng, nyquist_factor=0.8, min_freq=100
):
    """Auto-generate frequencies and phases for sine mode."""
    nyquist = sampling_rate / 2
    frequencies = []
    phases = []
    for _ in range(num_sources):
        n_tones = rng.integers(1, 4)  # 1–3 tones per source
        freqs = sorted(
            rng.uniform(min_freq, nyquist * nyquist_factor, size=n_tones).tolist()
        )
        frequencies.append(freqs)
        phases.append(float(rng.uniform(0, 2 * np.pi)))
    return frequencies, phases


def make_frequency_selector(
    min_freq_hz: float | None = None,
    single_frequency_hz: float | None = None,
):
    if min_freq_hz is not None and single_frequency_hz is not None:
        raise ValueError("Use either min_freq_hz or single_frequency_hz, not both.")

    def freq_selector(freqs):
        if single_frequency_hz is not None:
            return np.isclose(freqs, single_frequency_hz)
        if min_freq_hz is not None:
            return freqs >= min_freq_hz
        return np.ones_like(freqs, dtype=bool)

    return freq_selector


def quick_sim(
    num_mics: int,
    num_sources: int,
    num_active: int,
    seed: int = 0,
    sampling_rate: float = 2000.0,
    duration: float = 0.05,
    source_distance: float = 1.5,
    mic_radius: float = 0.3,
    array_type: str = "circular",
    mic_angle_start: float = 0.0,
    mic_angle_span: float = 2 * np.pi,
    source_angle_start: float = 0.0,
    source_angle_span: float = 2 * np.pi,
    mode: str = "noise",
    amplitude: float = 1.0,
    min_freq_hz: float | None = None,
    single_frequency_hz: float | None = None,
    sensor_snr_db: float | None = None,
    model_snr_db: float | None = None,
    inverse_method: str = "mp",
) -> Simulation:
    """
    One-liner convenience: sources placed uniformly over the configured angle span.

    Args:
        num_mics: Number of microphones (M).
        num_sources: Number of candidate sources (S).
        num_active: How many sources emit signal.
        seed: Master seed — split internally so that changing num_active
              does not alter the noise waveforms of already-selected sources.
        sampling_rate: Sampling rate in Hz.
        duration: Signal duration in seconds.
        source_distance: Distance of the source ring from the origin (m).
        mic_radius: Radius of the mic array, or adjacent spacing for linear arrays.
        array_type: "circular", "arc", or "linear".
        mic_angle_start: Starting angle for arc microphone placement (rad).
        mic_angle_span: Angular span for arc microphone placement (rad).
        source_angle_start: Starting angle for source placement (rad).
        source_angle_span: Angular span for source placement (rad).
        mode: "noise" (white Gaussian) or "sine" (random sum-of-sines).
        amplitude: Amplitude of the signals.
        single_frequency_hz: If provided, keep only this exact FFT frequency bin.

    Returns:
        Simulation dataclass with Y, A, X, x, etc.
    """
    # Seeds for source randomness, the noise is governed in simulate()
    rng = np.random.default_rng(seed)
    selection_rng = np.random.default_rng(int(rng.integers(0, 2**31)))
    signal_rng = np.random.default_rng(int(rng.integers(0, 2**31)))
    simulation_seed = int(rng.integers(0, 2**31))

    mics, sources = construct_geometry(
        array_type=array_type,
        num_mics=num_mics,
        num_sources=num_sources,
        mic_radius=mic_radius,
        mic_angle_start=mic_angle_start,
        mic_angle_span=mic_angle_span,
        source_distance=source_distance,
        source_angle_start=source_angle_start,
        source_angle_span=source_angle_span,
    )

    active = select_active_sources(sources, num_active, rng=selection_rng)

    if mode == "sine":

        frequencies, phases = _random_sine_params(
            num_sources, sampling_rate, signal_rng
        )
        sources, _ = generate_signals(
            sources,
            active_indices=active,
            sampling_rate=sampling_rate,
            duration=duration,
            mode="sine",
            frequencies=frequencies,
            phases=phases,
            rng=signal_rng,
            amplitude=amplitude,
        )
    else:
        sources, _ = generate_signals(
            sources,
            active_indices=active,
            sampling_rate=sampling_rate,
            duration=duration,
            mode="noise",
            rng=signal_rng,
            amplitude=amplitude,
        )

    freq_selector = make_frequency_selector(min_freq_hz, single_frequency_hz)

    sim = simulate_from_time_domain(
        mics=mics,
        sources=sources,
        active_indices=active,
        sampling_rate=sampling_rate,
        duration=duration,
        freq_selector=freq_selector,
        seed=simulation_seed,
        sensor_snr_db=sensor_snr_db,
        model_snr_db=model_snr_db,
        inverse_method=inverse_method,
    )

    return sim


def quick_frequency_sim(
    num_mics: int,
    num_sources: int,
    num_active: int,
    seed: int = 0,
    sampling_rate: float = 2000.0,
    duration: float = 0.05,
    source_distance: float = 1.5,
    mic_radius: float = 0.3,
    array_type: str = "circular",
    mic_angle_start: float = 0.0,
    mic_angle_span: float = 2 * np.pi,
    source_angle_start: float = 0.0,
    source_angle_span: float = 2 * np.pi,
    component_amplitude: float = 1.0,
    magnitude_jitter: float = 0.0,
    min_freq_hz: float | None = None,
    single_frequency_hz: float | None = None,
    sensor_snr_db: float | None = None,
    model_snr_db: float | None = None,
    inverse_method: str = "mp",
) -> Simulation:
    """
    Artificial frequency-domain simulation with near-flat active spectra.

    Each active source gets approximately equal-magnitude frequency components
    with random phase. This is intended as a solver sanity check rather than a
    physically realistic source model.

    If single_frequency_hz is provided, only that exact FFT frequency bin is kept.
    """
    rng = np.random.default_rng(seed)
    selection_rng = np.random.default_rng(int(rng.integers(0, 2**31)))
    spectrum_rng = np.random.default_rng(int(rng.integers(0, 2**31)))
    simulation_seed = int(rng.integers(0, 2**31))

    mics, sources = construct_geometry(
        array_type=array_type,
        num_mics=num_mics,
        num_sources=num_sources,
        mic_radius=mic_radius,
        mic_angle_start=mic_angle_start,
        mic_angle_span=mic_angle_span,
        source_distance=source_distance,
        source_angle_start=source_angle_start,
        source_angle_span=source_angle_span,
    )

    active = select_active_sources(sources, num_active, rng=selection_rng)

    sources, X = generate_frequency_domain_signals(
        sources,
        active_indices=active,
        sampling_rate=sampling_rate,
        duration=duration,
        component_amplitude=component_amplitude,
        magnitude_jitter=magnitude_jitter,
        rng=spectrum_rng,
    )

    freq_selector = make_frequency_selector(min_freq_hz, single_frequency_hz)

    return simulate_from_frequency_domain(
        mics=mics,
        sources=sources,
        active_indices=active,
        X=X,
        sampling_rate=sampling_rate,
        duration=duration,
        freq_selector=freq_selector,
        seed=simulation_seed,
        sensor_snr_db=sensor_snr_db,
        model_snr_db=model_snr_db,
        inverse_method=inverse_method,
    )
