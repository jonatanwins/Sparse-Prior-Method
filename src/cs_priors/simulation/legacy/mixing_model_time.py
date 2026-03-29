import numpy as np
import random
from scipy.fft import fft, ifft, fftfreq
from ...geometry.arrays import circular_array, linear_array
from ...geometry import SoundSource
from ...constants import SPEED_OF_SOUND
from ..Simulation import Simulation


def calculate_delays(mics, source: SoundSource):
    source_x, source_y = source.get_position()
    distances = np.sqrt((mics[:, 0] - source_x) ** 2 + (mics[:, 1] - source_y) ** 2)
    delays = distances / SPEED_OF_SOUND
    return delays


def single_waveform_at_all_mics(mics, src, t):
    """
    Compute the waveform at each microphone due to one sound source.

    Returns:
        waveforms: A 2D array (mic index x time samples) for the source.
        delays: Delay for each microphone.
    """

    delays = calculate_delays(mics, src)

    # at the microphone, hence the shift
    # BUG SQUASH: the shift is negative as we are at an earlier stage of the signal

    if isinstance(src.frequency, (list, np.ndarray)):
        # Sum multiple frequency components
        waveforms = np.zeros((len(mics), len(t)))
        for freq in src.frequency:
            waveforms += np.array(
                [
                    src.amplitude
                    * src.function(2 * np.pi * freq * (t - delay) + src.phase)
                    for delay in delays
                ]
            )
    else:
        waveforms = np.array(
            [
                src.amplitude
                * src.function(2 * np.pi * src.frequency * (t - delay) + src.phase)
                for delay in delays
            ]
        )

    return waveforms, delays


def waveforms_at_mics(mics, sources, sampling_rate, duration=None):
    """
    TODO
    composite waveforms, uses same t for all
    """
    if duration is None:
        min_freq = min(source.frequency for source in sources)
        duration = 1 / min_freq

    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    composite_waveforms = np.zeros((len(mics), len(t)))
    individual_waveforms = {}
    delays_dict = {}

    for idx, source in enumerate(sources):
        waveform, delays = single_waveform_at_all_mics(mics, source, t)  # same t
        composite_waveforms += waveform
        individual_waveforms[idx] = waveform
        delays_dict[idx] = delays

    return t, composite_waveforms, individual_waveforms, delays_dict


def s_sparse_sources(s, sources, seed=None):
    sparse_sources = [
        SoundSource(
            distance=src.distance,
            angle=src.angle,
            frequency=src.frequency,
            amplitude=0,
            phase=src.phase,
            function=src.function,
        )
        for src in sources
    ]
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    indicies = random.sample(range(len(sources)), s)
    for i in indicies:
        sparse_sources[i] = sources[i]
    return sparse_sources, indicies


def run_simulation(
    array_type="circular",
    num_mics=5,
    spacing=0.1,
    num_sources=1,
    s_sparse=None,
    f0=100,
    frequencies=None,
    distance=1.5,
    amplitude=2,
    amplitude_step=0,
    phase_step=0.3,
    angle_base=np.pi / 4,
    angle_step=None,
    sampling_rate_factor=10,
    simulation_duration=None,
    walls=[],
    seed=None,
):
    if angle_step is None:
        angle_step = 2 * np.pi / num_sources

    # 1. Initialize microphone array.
    if array_type.lower() == "linear":
        mics = linear_array(num_mics, spacing)
    elif array_type.lower() == "circular":
        mics = circular_array(num_mics, spacing)
    else:
        raise ValueError("array_type must be 'linear' or 'circular'")

    # 2. Create sound sources.
    # Broadband sources, e.g. frequencies = [[100, 200], [150, 250], ...]
    if frequencies is not None:
        sources = [
            SoundSource(
                distance=distance,
                angle=angle_base + angle_step * a,
                frequency=frequencies[a],
                amplitude=amplitude,
                phase=phase_step * a,
                function=np.sin,
            )
            for a in range(num_sources)
        ]
        # list of the form [[100, 150, 200], [300]] get the max frequency
        f_min = min(min(freq_list) for freq_list in frequencies)
        f_max = max(max(freq_list) for freq_list in frequencies)
        simulation_duration = 1 / f_min
        SAMPLING_RATE = sampling_rate_factor * f_max

    # if there is only one frequency, all sources have the same frequency
    elif isinstance(f0, (int, float)):
        sources = [
            SoundSource(
                distance=distance,
                angle=angle_base + angle_step * a,
                frequency=f0,
                amplitude=amplitude,
                phase=phase_step * a,
                function=np.sin,
            )
            for a in range(num_sources)
        ]
        simulation_duration = 1 / f0
        SAMPLING_RATE = sampling_rate_factor * f0

    # Each source has its own frequency
    elif isinstance(f0, (list, np.ndarray)):
        if len(f0) != num_sources:
            raise ValueError(
                "If f0 is a list or array, its length must match num_sources"
            )
        sources = [
            SoundSource(
                distance=distance,
                angle=angle_base + angle_step * a,
                frequency=f0[a],
                amplitude=amplitude + amplitude_step * a,
                phase=phase_step * a,
                function=np.sin,
            )
            for a in range(num_sources)
        ]
        f_min = min(f0)
        f_max = max(f0)
        simulation_duration = 1 / f_min
        SAMPLING_RATE = sampling_rate_factor * f_max
    else:
        raise ValueError(
            "f0 must be a number, list, or array or frequencies argument must be provided"
        )

    # Apply sparsity if specified.
    if s_sparse is not None and s_sparse < num_sources:
        sources, active_indices = s_sparse_sources(s_sparse, sources, seed=seed)
    else:
        active_indices = list(range(num_sources))

    # 3. Define sampling parameters.
    N = int(SAMPLING_RATE * simulation_duration)

    # 4. Simulate the waveforms at the microphones.
    t, composite_waveforms, individual_waveforms, delays_dict = waveforms_at_mics(
        mics, sources, SAMPLING_RATE, simulation_duration
    )

    # 5. TIME DOMAIN SIGNALS.
    if frequencies is not None:
        x_time = np.array(
            [
                sum(
                    src.amplitude * src.function(2 * np.pi * freq * t + src.phase)
                    for freq in src.frequency
                )
                for src in sources
            ]
        )
    else:
        # print("Assuming single frequency per source")
        x_time = np.array(
            [
                src.amplitude * src.function(2 * np.pi * src.frequency * t + src.phase)
                for src in sources
            ]
        )
    y_time = composite_waveforms

    # 6. FREQUENCY DOMAIN PROCESSING.
    X = fft(x_time, axis=1)
    freqs = fftfreq(N, d=1 / SAMPLING_RATE)

    # Build the mixing matrix C: dimensions [num_mics x num_sources x N].
    A = np.zeros((num_mics, len(sources), N), dtype=complex)
    for i in range(num_mics):
        for j, src in enumerate(sources):
            A[i, j, :] = np.exp(-2j * np.pi * freqs * delays_dict[j][i])

    # Predicted observations in the frequency domain.
    Y_pred = np.zeros((num_mics, N), dtype=complex)
    for idf in range(N):
        Y_pred[:, idf] = A[:, :, idf] @ X[:, idf]

    # Reconstruct the source spectrum.
    Y_fft = fft(y_time, axis=1)
    num_sources = len(sources)
    X_pred = np.zeros((num_sources, N), dtype=complex)
    A_pinv = np.zeros((num_sources, num_mics, N), dtype=complex)
    for idf in range(N):
        # TODO Change this to C[f] = matrix for that frequency, i.e. block matrices
        A_f = A[:, :, idf]
        A_f_pinv = np.linalg.pinv(A_f)
        A_pinv[:, :, idf] = A_f_pinv
        X_pred[:, idf] = A_f_pinv @ Y_fft[:, idf]

    # Inverse FFT to recover time-domain source signals.
    x_pred = ifft(X_pred, axis=1)

    return Simulation(
        t=t,
        composite_waveforms=composite_waveforms,
        individual_waveforms=individual_waveforms,
        delays_dict=delays_dict,
        x_time=x_time,
        y_time=y_time,
        X=X,
        Y=Y_fft,
        freqs=freqs,
        A=A,
        A_pinv=A_pinv,
        Y_pred=Y_pred,
        X_pred=X_pred,
        x_pred=x_pred,
        sources=sources,
        mics=mics,
        walls=walls,
        sampling_rate=SAMPLING_RATE,
        duration=simulation_duration,
        N=N,
        active_indices=active_indices,
    )


def just_YAX_from_simulation(
    num_mics=3,
    num_sources=5,
    s_sparse=2,
    freq_index=1,
    angle_step=0.3,
    angle_base=np.pi / 4,
    phase_step=0.3,
    seed=None,
):
    """
    Helper function to run a simulation and extract Y, A, X0, and X_TRUE for a specific frequency index.
    Returns:
        Y: Measurements at microphones for a specific frequency index
        A: Mixing matrix at that frequency index
        X0: Initial guess for source signals (pseudoinverse solution)
        X_TRUE: True source signals from the simulation
    """
    sim = run_simulation(
        num_mics=num_mics,
        num_sources=num_sources,
        s_sparse=s_sparse,
        angle_step=angle_step,
        angle_base=angle_base,
        phase_step=phase_step,
        seed=seed,
    )
    Y = sim.Y[:, freq_index].reshape(-1, 1)  # Measurements
    A = sim.A[:, :, freq_index]  # Mixing matrix
    X0 = np.linalg.pinv(A) @ Y  # initial guess for X
    X_TRUE = sim.X[:, freq_index]  # True source signals
    return Y, A, X0, X_TRUE
