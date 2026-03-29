"""
Tests for the frequency-domain simulation pipeline.

Assumes mixing_model.py exposes:
- moore_penrose_inverse
- ridge_inverse
- simulate(..., inverse_method="mp" | "ridge")
"""

import numpy as np
from numpy.testing import assert_allclose
from scipy.fft import ifft

from ..simulation.Simulation import Simulation
from ..simulation.mixing_model import (
    construct_geometry,
    select_active_sources,
    generate_signals,
    quick_sim,
    simulate,
    moore_penrose_inverse,
    ridge_inverse,
)


def _manual_mp(A: np.ndarray, Y: np.ndarray) -> np.ndarray:
    M, S, F = A.shape
    X_mp = np.zeros((S, F), dtype=complex)
    for k in range(F):
        X_mp[:, k] = np.linalg.pinv(A[:, :, k]) @ Y[:, k]
    return X_mp


def _manual_ridge(A: np.ndarray, Y: np.ndarray, noise_power: float) -> np.ndarray:
    M, S, F = A.shape
    X_ridge = np.zeros((S, F), dtype=complex)
    I = np.eye(S, dtype=complex)
    for k in range(F):
        Ak = A[:, :, k]
        X_ridge[:, k] = (
            np.linalg.inv(Ak.conj().T @ Ak + noise_power * I) @ Ak.conj().T @ Y[:, k]
        )
    return X_ridge


def _run(
    num_mics: int = 6,
    num_sources: int = 4,
    num_active: int = 2,
    mode: str = "sine",
    seed: int = 0,
    sampling_rate: float = 1000.0,
    duration: float = 0.1,
    sensor_snr_db: float | None = None,
    model_snr_db: float | None = None,
    min_freq_hz: float | None = None,
    **simulate_kwargs,
) -> Simulation:
    master = np.random.default_rng(seed)
    selection_rng = np.random.default_rng(int(master.integers(0, 2**31)))
    signal_rng = np.random.default_rng(int(master.integers(0, 2**31)))
    simulation_seed = int(master.integers(0, 2**31))

    mics, sources = construct_geometry(
        num_mics=num_mics,
        array_type="circular",
        num_sources=num_sources,
        source_distance=0.5,
        mic_radius=0.1,
    )
    active = select_active_sources(sources, num_active, rng=selection_rng)

    if mode == "sine":
        frequencies = [[50.0 * (i + 1)] for i in range(num_sources)]
        phases = [0.3 * i for i in range(num_sources)]
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

    return simulate(
        mics=mics,
        sources=sources,
        active_indices=active,
        sampling_rate=sampling_rate,
        duration=duration,
        freq_selector=freq_selector,
        seed=simulation_seed,
        sensor_snr_db=sensor_snr_db,
        model_snr_db=model_snr_db,
        **simulate_kwargs,
    )


def _run_square_exact() -> Simulation:
    num_mics = num_sources = 3
    sampling_rate = 1000.0
    duration = 0.1

    mics, sources = construct_geometry(
        num_mics=num_mics,
        array_type="circular",
        num_sources=num_sources,
        source_distance=0.5,
        mic_radius=0.1,
    )
    active = [0, 1, 2]

    # Zero-mean tones aligned with FFT bins so the DC bin is effectively zero.
    frequencies = [[100.0], [200.0], [300.0]]
    phases = [0.1, 0.2, 0.3]
    sources, _ = generate_signals(
        sources,
        active_indices=active,
        sampling_rate=sampling_rate,
        duration=duration,
        mode="sine",
        frequencies=frequencies,
        phases=phases,
        rng=np.random.default_rng(0),
    )

    return simulate(
        mics=mics,
        sources=sources,
        active_indices=active,
        sampling_rate=sampling_rate,
        duration=duration,
        seed=0,
        inverse_method="mp",
    )


def test_simulate_returns_expected_shapes():
    sim = _run()
    N = int(sim.sampling_rate * sim.duration)
    M = sim.mics.shape[0]
    S = len(sim.sources)

    assert isinstance(sim, Simulation)
    assert sim.x.shape == (S, N)
    assert sim.X.shape == (S, N)
    assert sim.A.shape == (M, S, N)
    assert sim.Y_clean.shape == (M, N)
    assert sim.Y.shape == (M, N)
    assert sim.eta.shape == (M, N)
    assert sim.delta.shape == (S, N)
    assert sim.X_pinv.shape == (S, N)
    assert sim.freqs.shape == (N,)
    assert sim.mics.shape == (M, 2)


def test_clean_forward_model_matches_einsum():
    sim = _run(sensor_snr_db=None, model_snr_db=None)
    expected = np.einsum("msf,sf->mf", sim.A, sim.X)
    assert_allclose(sim.Y_clean, expected, atol=1e-12)
    assert_allclose(sim.Y, sim.Y_clean, atol=1e-12)


def test_noisy_observation_decomposes_as_Y_clean_plus_A_delta_plus_eta():
    sim = _run(sensor_snr_db=20.0, model_snr_db=25.0)
    expected = sim.Y_clean + np.einsum("msf,sf->mf", sim.A, sim.delta) + sim.eta
    assert_allclose(sim.Y, expected, atol=1e-12)


def test_ifft_of_X_recovers_x():
    sim = _run()
    x_recovered = ifft(sim.X, axis=1)
    assert_allclose(x_recovered.real, sim.x, atol=1e-12)
    assert_allclose(x_recovered.imag, 0.0, atol=1e-12)


def test_inactive_sources_have_zero_time_signal():
    sim = _run(num_sources=5, num_active=2)
    for i in range(len(sim.sources)):
        if i not in sim.active_indices:
            assert_allclose(sim.x[i], 0.0, atol=1e-15)


def test_seed_reproducibility_for_full_simulation():
    sim1 = quick_sim(
        num_mics=4,
        num_sources=3,
        num_active=2,
        seed=99,
        mode="noise",
        sensor_snr_db=20.0,
        model_snr_db=25.0,
    )
    sim2 = quick_sim(
        num_mics=4,
        num_sources=3,
        num_active=2,
        seed=99,
        mode="noise",
        sensor_snr_db=20.0,
        model_snr_db=25.0,
    )

    assert sim1.active_indices == sim2.active_indices
    assert_allclose(sim1.x, sim2.x)
    assert_allclose(sim1.Y_clean, sim2.Y_clean)
    assert_allclose(sim1.eta, sim2.eta)
    assert_allclose(sim1.delta, sim2.delta)
    assert_allclose(sim1.Y, sim2.Y)
    assert_allclose(sim1.X_pinv, sim2.X_pinv)


def test_sensor_and_model_noise_streams_are_independent_under_toggles():
    common = dict(
        num_mics=4,
        num_sources=3,
        num_active=2,
        seed=7,
        mode="noise",
        sampling_rate=1000.0,
        duration=0.05,
    )

    sim_sensor = quick_sim(**common, sensor_snr_db=20.0, model_snr_db=None)
    sim_model = quick_sim(**common, sensor_snr_db=None, model_snr_db=20.0)
    sim_both = quick_sim(**common, sensor_snr_db=20.0, model_snr_db=20.0)

    assert_allclose(sim_sensor.delta, 0.0, atol=0.0)
    assert_allclose(sim_model.eta, 0.0, atol=0.0)
    assert_allclose(sim_sensor.eta, sim_both.eta)
    assert_allclose(sim_model.delta, sim_both.delta)


def test_moore_penrose_helper_matches_manual_pinv():
    sim = _run(sensor_snr_db=20.0, model_snr_db=25.0)
    expected = _manual_mp(sim.A, sim.Y)
    actual = moore_penrose_inverse(sim.A, sim.Y)
    assert_allclose(actual, expected, atol=1e-12)


def test_simulate_mp_solution_matches_manual_pinv():
    sim = _run(sensor_snr_db=20.0, model_snr_db=25.0, inverse_method="mp")
    expected = _manual_mp(sim.A, sim.Y)
    assert_allclose(sim.X_pinv, expected, atol=1e-12)


def test_ridge_helper_matches_direct_ridge_formula():
    sim = _run(sensor_snr_db=20.0, model_snr_db=None)
    noise_power = np.mean(np.abs(sim.eta) ** 2)
    expected = _manual_ridge(sim.A, sim.Y, noise_power)
    actual = ridge_inverse(sim.A, sim.Y, noise_power)
    assert_allclose(actual, expected, atol=1e-12)


def test_simulate_ridge_solution_matches_direct_ridge_formula():
    sensor_snr_db = 20.0
    sim = _run(sensor_snr_db=sensor_snr_db, model_snr_db=None, inverse_method="ridge")
    P_Y = np.mean(np.abs(sim.Y_clean) ** 2)
    noise_power = P_Y * (10 ** (-sensor_snr_db / 10))
    expected = _manual_ridge(sim.A, sim.Y, noise_power)
    assert_allclose(sim.X_pinv, expected, atol=1e-12)


def test_square_noiseless_all_frequencies_recovers_true_X():
    sim = _run_square_exact()

    nonzero_bins = np.where(np.linalg.norm(sim.X, axis=0) > 1e-8)[0]
    for k in nonzero_bins:
        assert np.linalg.matrix_rank(sim.A[:, :, k]) == sim.A.shape[1]

    assert_allclose(sim.X_pinv, sim.X, atol=1e-10)
