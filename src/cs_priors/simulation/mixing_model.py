import numpy as np
import random
from scipy.fft import fft, fftfreq
from ..geometry.arrays import circular_array, linear_array, arc_array
from ..geometry.SoundSource import SoundSource
from ..constants import SPEED_OF_SOUND
from .Simulation import Simulation


# ── 1. Geometry ──────────────────────────────────────────────────────────────


def construct_geometry(
    num_mics: int,
    array_type: str,
    mic_spacing: float,
    num_sources: int,
    source_distance: float,
    angle_base: float,
    angle_step: float,
):
    """Create mic array (M x 2) and source list (S,)."""
    if array_type.lower() == "linear":
        mics = linear_array(num_mics, mic_spacing)
    elif array_type.lower() == "circular":
        mics = circular_array(num_mics, mic_spacing)
    elif array_type.lower() == "arc":
        raise ValueError("Use arc_array() directly or quick_sector_sim() for arc mics.")
    else:
        raise ValueError("array_type must be 'linear' or 'circular'")

    sources = [
        SoundSource(
            distance=source_distance,
            angle=angle_base + angle_step * s,
            time_series=None,
        )
        for s in range(num_sources)
    ]
    return mics, sources


# ── 2. Select active (non-mute) sources ─────────────────────────────────────


def select_active_sources(sources, num_active_sources, seed=None):
    """Randomly pick the defined number of active source indices."""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    return random.sample(range(len(sources)), num_active_sources)


# ── 3. Generate time-domain signals ─────────────────────────────────────────


def generate_signals(
    sources: list[SoundSource],
    active_indices: list[int],
    sampling_rate: float,
    duration: float,
    mode: str = "sine",
    frequencies: list[list[float]] | None = None,
    phases: list[float] | None = None,
    seed: int | None = None,
):
    """
    Assign time_series to each source.  Muted sources get zeros.

    Args:
        mode: "sine" — sum-of-sines per source (requires frequencies, phases).
              "noise" — white Gaussian noise.
        frequencies: (S,) list of freq-lists, e.g. [[100, 200], [150]].
        phases: (S,) list of starting phases.
    """
    N = int(sampling_rate * duration)
    t = np.linspace(0, duration, N, endpoint=False)

    if seed is not None:
        np.random.seed(seed)

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
            src.time_series = np.random.randn(N)
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


# ── 6. Simulate: full pipeline ──────────────────────────────────────────────


def simulate(
    mics: np.ndarray,
    sources: list[SoundSource],
    active_indices: list[int],
    sampling_rate: float,
    duration: float,
) -> Simulation:
    """
    Frequency-domain simulation: Y = A @ X.

    Dimensions (M=mics, S=sources, N=time samples):
        x  : (S x N)  time-domain source signals
        X  : (S x N)  FFT of x
        A  : (M x S x N) mixing matrix per freq bin
        Y  : (M x N)  observed spectra
    """
    S = len(sources)
    M = len(mics)
    N = int(sampling_rate * duration)

    # Stack time series -> x (S x N)
    x = np.array([src.time_series for src in sources])  # (S, N)

    # FFT of source signals -> X (S x N)
    X = fft(x, axis=1)

    # Frequency bins
    freqs = fftfreq(N, d=1.0 / sampling_rate)

    # Delay matrix (M x S) and mixing matrix A (M x S x N)
    delays = compute_delay_matrix(mics, sources)
    A = compute_mixing_matrix(delays, freqs)

    # Y = A @ X  per frequency bin, vectorised over all bins:
    # A: (M, S, N), X: (S, N) -> Y_i = \sum_j A_{ij} * X_{j}
    # Same as Y[:, n] = A[:,:,n] @ X[:, n]
    Y = np.einsum("msn,sn->mn", A, X)

    # Pseudoinverse recovery: X_pinv[:, n] = pinv(A[:,:,n]) @ Y[:, n]
    X_pinv = np.zeros((S, N), dtype=complex)
    for n in range(N):
        X_pinv[:, n] = np.linalg.pinv(A[:, :, n]) @ Y[:, n]

    return Simulation(
        Y=Y,
        A=A,
        X=X,
        X_pinv=X_pinv,
        x=x,
        freqs=freqs,
        sources=sources,
        mics=mics,
        active_indices=active_indices,
        sampling_rate=sampling_rate,
        duration=duration,
    )


# ── 7. Quick simulation ─────────────────────────────────────────────────────


def _random_sine_params(num_sources, sampling_rate, rng):
    """Auto-generate frequencies and phases for sine mode."""
    nyquist = sampling_rate / 2
    frequencies = []
    phases = []
    for _ in range(num_sources):
        n_tones = rng.integers(1, 4)  # 1–3 tones per source
        freqs = sorted(rng.uniform(20, nyquist * 0.2, size=n_tones).tolist())
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
    mic_spacing: float = 0.3,
    array_type: str = "circular",
    mode: str = "noise",
    snr_db: float | None = None,
) -> Simulation:
    """
    One-liner convenience: sources on a uniform ring.

    Args:
        num_mics: Number of microphones (M).
        num_sources: Number of candidate sources (S).
        num_active: How many sources emit signal.
        seed: Master seed — split internally so that changing num_active
              does not alter the noise waveforms of already-selected sources.
        sampling_rate: Sampling rate in Hz.
        duration: Signal duration in seconds.
        source_distance: Distance of the source ring from the origin (m).
        mic_spacing: Spacing parameter for the mic array (m).
        array_type: "circular" or "linear".
        mode: "noise" (white Gaussian) or "sine" (random sum-of-sines).
        snr_db: If given, additive white Gaussian noise is added to Y
                at this SNR (dB).  None means noiseless.

    Returns:
        Simulation dataclass with Y, A, X, x, etc.
    """
    rng = np.random.default_rng(seed)
    selection_seed = int(rng.integers(0, 2**31))
    signal_seed = int(rng.integers(0, 2**31))

    # Uniform ring: sources equally spaced around 2π
    angle_step = 2 * np.pi / num_sources

    mics, sources = construct_geometry(
        num_mics=num_mics,
        array_type=array_type,
        mic_spacing=mic_spacing,
        num_sources=num_sources,
        source_distance=source_distance,
        angle_base=0.0,
        angle_step=angle_step,
    )

    active = select_active_sources(sources, num_active, seed=selection_seed)

    if mode == "sine":
        sine_rng = np.random.default_rng(signal_seed)
        frequencies, phases = _random_sine_params(num_sources, sampling_rate, sine_rng)
        sources, _ = generate_signals(
            sources,
            active,
            sampling_rate,
            duration,
            mode="sine",
            frequencies=frequencies,
            phases=phases,
        )
    else:
        sources, _ = generate_signals(
            sources,
            active,
            sampling_rate,
            duration,
            mode="noise",
            seed=signal_seed,
        )

    sim = simulate(mics, sources, active, sampling_rate, duration)

    # Optional measurement noise
    if snr_db is not None:
        noise_rng = np.random.default_rng(seed + 1)
        signal_power = np.mean(np.abs(sim.Y) ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.sqrt(noise_power / 2) * (
            noise_rng.standard_normal(sim.Y.shape)
            + 1j * noise_rng.standard_normal(sim.Y.shape)
        )
        sim.Y = sim.Y + noise

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
    snr_db: float | None = None,
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
        snr_db:          Optional measurement noise level (dB).

    Returns:
        Simulation dataclass.
    """
    rng = np.random.default_rng(seed)
    selection_seed = int(rng.integers(0, 2**31))
    signal_seed = int(rng.integers(0, 2**31))

    # Mics on an arc
    mics = arc_array(num_mics, mic_radius, angle_start, angle_span)

    # Sources evenly spaced in the same sector
    if num_sources == 1:
        source_angles = [angle_start + angle_span / 2]
    else:
        source_angles = np.linspace(
            angle_start, angle_start + angle_span, num_sources, endpoint=True
        )

    sources = [
        SoundSource(distance=source_distance, angle=float(a), time_series=None)
        for a in source_angles
    ]

    active = select_active_sources(sources, num_active, seed=selection_seed)

    if mode == "sine":
        sine_rng = np.random.default_rng(signal_seed)
        frequencies, phases = _random_sine_params(num_sources, sampling_rate, sine_rng)
        sources, _ = generate_signals(
            sources,
            active,
            sampling_rate,
            duration,
            mode="sine",
            frequencies=frequencies,
            phases=phases,
        )
    else:
        sources, _ = generate_signals(
            sources,
            active,
            sampling_rate,
            duration,
            mode="noise",
            seed=signal_seed,
        )

    sim = simulate(mics, sources, active, sampling_rate, duration)

    # Optional measurement noise
    if snr_db is not None:
        noise_rng = np.random.default_rng(seed + 1)
        signal_power = np.mean(np.abs(sim.Y) ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.sqrt(noise_power / 2) * (
            noise_rng.standard_normal(sim.Y.shape)
            + 1j * noise_rng.standard_normal(sim.Y.shape)
        )
        sim.Y = sim.Y + noise

    return sim
