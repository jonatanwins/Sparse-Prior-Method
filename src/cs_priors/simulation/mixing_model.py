import numpy as np
from scipy.fft import fft, fftfreq
from ..geometry.arrays import circular_array, linear_array, arc_array
from ..geometry.SoundSource import SoundSource
from ..constants import SPEED_OF_SOUND
from .Simulation import Simulation


# ── 1. Geometry ──────────────────────────────────────────────────────────────


def construct_geometry(
    num_mics: int,
    array_type: str,
    num_sources: int,
    source_distance: float,
    mic_radius: float | None = None,
    angle_start: float = 0.0,
    angle_span: float = 2 * np.pi,
    mic_spacing: float | None = None,
    angle_base: float | None = None,
    angle_step: float | None = None,
):
    """Create mic array (M x 2) and source list (S,)."""
    array_type = array_type.lower()

    if mic_spacing is not None:
        mic_radius = mic_spacing
    if mic_radius is None:
        raise ValueError("mic_radius must be provided")

    if angle_base is not None or angle_step is not None:
        if angle_base is None or angle_step is None:
            raise ValueError("angle_base and angle_step must be provided together")
        source_angles = angle_base + angle_step * np.arange(num_sources)
    elif num_sources == 1:
        if array_type == "arc":
            source_angles = [angle_start + angle_span / 2]
        else:
            source_angles = [angle_start]
    elif array_type == "arc":
        source_angles = np.linspace(
            angle_start, angle_start + angle_span, num_sources, endpoint=True
        )
    else:
        angle_step = angle_span / num_sources
        source_angles = angle_start + angle_step * np.arange(num_sources)

    if array_type == "linear":
        mics = linear_array(num_mics, mic_radius)
    elif array_type == "circular":
        mics = circular_array(num_mics, mic_radius)
    elif array_type == "arc":
        mics = arc_array(num_mics, mic_radius, angle_start, angle_span)
    else:
        raise ValueError("array_type must be 'linear', 'circular', or 'arc'")

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

    return sources, t


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


# ── 6. Simulate: full pipeline ──────────────────────────────────────────────


def simulate(
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
    Frequency-domain simulation: Y = A @ X.

    Dimensions (M=mics, S=sources, N=time samples):
        x  : (S x N)  time-domain source signals
        X  : (S x N)  FFT of x
        A  : (M x S x N) mixing matrix per freq bin
        Y  : (M x N)  observed spectra
    """
    # Noise seeds
    rng = np.random.default_rng(seed)
    sensor_noise_seed = int(rng.integers(0, 2**31))
    model_noise_seed = int(rng.integers(0, 2**31))

    S = len(sources)
    M = len(mics)
    N = int(sampling_rate * duration)

    # Stack time series -> x (S x N)
    x = np.array([src.time_series for src in sources])  # (S, N)

    # FFT of source signals -> X (S x N)
    X = np.array(fft(x, axis=1))

    # Frequency bins
    freqs = fftfreq(N, d=1.0 / sampling_rate)

    # Delay matrix (M x S) and mixing matrix A (M x S x N)
    delays = compute_delay_matrix(mics, sources)
    A = compute_mixing_matrix(delays, freqs)

    # Y = A @ X  per frequency bin, vectorised over all bins:
    # A: (M, S, N), X: (S, N) -> Y_i = \sum_j A_{ij} * X_{j}
    # Same as Y[:, n] = A[:,:,n] @ X[:, n]
    Y_clean = np.einsum("msn,sn->mn", A, X)

    # Select only some of the frequencies F < N
    if freq_selector is not None:
        freq_mask = freq_selector(freqs)
        F = np.sum(freq_mask)  # number of frequencies after masking
        X = X[:, freq_mask]
        A = A[:, :, freq_mask]
        Y_clean = Y_clean[:, freq_mask]
        freqs = freqs[freq_mask]
    else:
        F = N

    # Add noise before computing the pseudoinverse
    eta = np.zeros_like(Y_clean, dtype=complex)
    delta = np.zeros_like(X, dtype=complex)

    if sensor_snr_db is not None:
        # snr_db = 10 * log10(P_signal / P_noise)
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

    # Pseudoinverse recovery: X_pinv[:, k] = pinv(A[:,:,k]) @ Y[:, k]
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
    angle_start: float = 0.0,
    angle_span: float = 2 * np.pi,
    mode: str = "noise",
    min_freq_hz: float | None = None,
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
        mic_radius: Radius of the mic array (m).
        array_type: "circular", "arc", or "linear".
        angle_start: Starting angle for source placement and arc arrays (rad).
        angle_span: Angular span for source placement and arc arrays (rad).
        mode: "noise" (white Gaussian) or "sine" (random sum-of-sines).

    Returns:
        Simulation dataclass with Y, A, X, x, etc.
    """
    # Seeds for source randomness, the noise is governed in simulate()
    rng = np.random.default_rng(seed)
    selection_rng = np.random.default_rng(int(rng.integers(0, 2**31)))
    signal_rng = np.random.default_rng(int(rng.integers(0, 2**31)))
    simulation_seed = int(rng.integers(0, 2**31))

    mics, sources = construct_geometry(
        num_mics=num_mics,
        array_type=array_type,
        num_sources=num_sources,
        source_distance=source_distance,
        mic_radius=mic_radius,
        angle_start=angle_start,
        angle_span=angle_span,
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
        )
    else:
        sources, _ = generate_signals(
            sources,
            active_indices=active,
            sampling_rate=sampling_rate,
            duration=duration,
            mode="noise",
            rng=signal_rng,
        )

    def freq_selector(freqs):
        if min_freq_hz is None:
            return np.ones_like(freqs, dtype=bool)
        return freqs >= min_freq_hz

    sim = simulate(
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


# ── 8. Sector simulation ────────────────────────────────────────────────────


def quick_sector_sim(
    num_mics: int,
    num_sources: int,
    num_active: int,
    seed: int = 0,
    sampling_rate: float = 2000.0,
    duration: float = 0.05,
    source_distance: float = 1.5,
    mic_radius: float = 0.3,
    angle_start: float = 0.0,
    angle_span: float = np.pi / 2,
    mode: str = "noise",
    min_freq_hz: float | None = None,
    sensor_snr_db: float | None = None,
    model_snr_db: float | None = None,
    inverse_method: str = "mp",
) -> Simulation:
    """
    Sector convenience sim: mics on an arc, sources in the same angular sector.

    Both microphones and sources are placed within
    [angle_start, angle_start + angle_span].

    Args:
        num_mics:        Number of microphones (M).
        num_sources:     Number of candidate sources (S).
        num_active:      How many sources emit signal.
        seed:            Master seed.
        sampling_rate:   Sampling rate in Hz.
        duration:        Signal duration in seconds.
        source_distance: Radius of the source ring (m).
        mic_radius:      Radius of the mic arc (m).
        angle_start:     Starting angle of the sector (rad).
        angle_span:      Angular width of the sector (rad).
        mode:            "noise" (white Gaussian) or "sine" (random sum-of-sines).
        min_freq_hz:     Minimum frequency to include in the simulation (Hz).
        sensor_snr_db:   SNR of the sensor noise (dB).
        model_snr_db:    SNR of the model noise (dB).
    Returns:
        Simulation dataclass.
    """
    return quick_sim(
        num_mics=num_mics,
        num_sources=num_sources,
        num_active=num_active,
        seed=seed,
        sampling_rate=sampling_rate,
        duration=duration,
        source_distance=source_distance,
        mic_radius=mic_radius,
        array_type="arc",
        angle_start=angle_start,
        angle_span=angle_span,
        mode=mode,
        min_freq_hz=min_freq_hz,
        sensor_snr_db=sensor_snr_db,
        model_snr_db=model_snr_db,
        inverse_method=inverse_method,
    )
