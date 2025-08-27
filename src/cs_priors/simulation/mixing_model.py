import numpy as np
from random import random
from scipy.fft import fft, ifft, fftfreq
from ..geometry.arrays import circular_array, linear_array
from ..geometry.sources import SoundSource
from ..constants import SPEED_OF_SOUND
from .Simulation import Simulation


def calculate_delays(mics, source: SoundSource):
    source_x, source_y = source.get_position()
    distances = np.sqrt((mics.x - source_x) ** 2 + (mics.y - source_y) ** 2)
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
    composite_waveforms = np.zeros((len(mics.x), len(t)))
    individual_waveforms = {}
    delays_dict = {}

    for idx, source in enumerate(sources):
        waveform, delays = single_waveform_at_all_mics(mics, source, t)  # same t
        composite_waveforms += waveform
        individual_waveforms[idx] = waveform
        delays_dict[idx] = delays

    return t, composite_waveforms, individual_waveforms, delays_dict


def s_sparse_sources(s, sources, seed=2025):
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
    random.seed(seed)
    indicies = random.sample(range(len(sources)), s)
    for i in indicies:
        sparse_sources[i] = sources[i]
    return sparse_sources, indicies


def run_simulation(
    array_type="circular",
    num_mics=5,
    spacing=0.1,
    no_sources=1,
    s_sparse=None,
    f0=100,
    distance=1.5,
    amplitude=2,
    phase_step=0.3,
    angle_base=np.pi / 4,
    sampling_rate_factor=10,
    simulation_duration=None,
):

    # 1. Initialize microphone array.
    if array_type.lower() == "linear":
        mics = linear_array(num_mics, spacing)
    elif array_type.lower() == "circular":
        mics = circular_array(num_mics, spacing)
    else:
        raise ValueError("array_type must be 'linear' or 'circular'")

    # 2. Create sound sources.
    sources = [
        SoundSource(
            distance=distance,
            angle=angle_base + phase_step * a,
            frequency=f0,
            amplitude=amplitude,
            phase=phase_step * a,
            function=np.sin,
        )
        for a in range(no_sources)
    ]
    if s_sparse is not None and s_sparse < no_sources:
        sources, active_indices = s_sparse_sources(s_sparse, sources)
    else:
        active_indices = list(range(no_sources))

    # 3. Define sampling parameters.
    SAMPLING_RATE = sampling_rate_factor * f0
    if simulation_duration is None:
        simulation_duration = max(1 / src.frequency for src in sources)
    N = int(SAMPLING_RATE * simulation_duration)

    # 4. Simulate the waveforms at the microphones.
    t, composite_waveforms, individual_waveforms, delays_dict = waveforms_at_mics(
        mics, sources, SAMPLING_RATE, simulation_duration
    )

    # 5. TIME DOMAIN SIGNALS.
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
    C = np.zeros((num_mics, len(sources), N), dtype=complex)
    for i in range(num_mics):
        for j, src in enumerate(sources):
            C[i, j, :] = np.exp(-2j * np.pi * freqs * delays_dict[j][i])

    # Predicted observations in the frequency domain.
    Y_pred = np.zeros((num_mics, N), dtype=complex)
    for idf in range(N):
        Y_pred[:, idf] = C[:, :, idf] @ X[:, idf]

    # Reconstruct the source spectrum.
    Y_fft = fft(y_time, axis=1)
    num_sources = len(sources)
    X_pred = np.zeros((num_sources, N), dtype=complex)
    C_pinv = np.zeros((num_sources, num_mics, N), dtype=complex)
    for idf in range(N):
        C_f = C[:, :, idf]
        C_f_pinv = np.linalg.pinv(C_f)
        C_pinv[:, :, idf] = C_f_pinv
        X_pred[:, idf] = C_f_pinv @ Y_fft[:, idf]

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
        C=C,
        C_pinv=C_pinv,
        Y_pred=Y_pred,
        X_pred=X_pred,
        x_pred=x_pred,
        sources=sources,
        mics=mics,
        sampling_rate=SAMPLING_RATE,
        duration=simulation_duration,
        N=N,
        active_indices=active_indices,
    )
