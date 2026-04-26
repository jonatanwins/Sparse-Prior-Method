import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.fft import fft, ifft

from ..simulation.mixing_model import (
    construct_geometry,
    generate_signals,
    quick_frequency_sim,
    simulate_from_time_domain,
    simulate_from_frequency_domain,
)


def test_quick_frequency_sim_builds_flat_active_positive_frequency_magnitudes():
    component_amplitude = 2.0

    sim = quick_frequency_sim(
        num_mics=4,
        num_sources=6,
        num_active=2,
        seed=0,
        sampling_rate=1000.0,
        duration=0.1,
        component_amplitude=component_amplitude,
        magnitude_jitter=0.0,
        min_freq_hz=None,
        sensor_snr_db=None,
        model_snr_db=None,
        inverse_method="mp",
    )

    positive_mask = sim.freqs > 0
    inactive_indices = [i for i in range(sim.X.shape[0]) if i not in sim.active_indices]

    for idx in sim.active_indices:
        assert_allclose(
            np.abs(sim.X[idx, positive_mask]),
            component_amplitude,
            atol=1e-12,
        )

    if inactive_indices:
        assert_allclose(sim.X[inactive_indices], 0.0, atol=1e-12)

    x_recovered = ifft(sim.X, axis=1)
    assert_allclose(x_recovered.real, sim.x, atol=1e-12)
    assert_allclose(x_recovered.imag, 0.0, atol=1e-12)


def test_quick_frequency_sim_respects_magnitude_jitter_bounds():
    component_amplitude = 2.0
    magnitude_jitter = 0.1

    sim = quick_frequency_sim(
        num_mics=4,
        num_sources=6,
        num_active=2,
        seed=1,
        sampling_rate=1000.0,
        duration=0.1,
        component_amplitude=component_amplitude,
        magnitude_jitter=magnitude_jitter,
        min_freq_hz=None,
        sensor_snr_db=None,
        model_snr_db=None,
        inverse_method="mp",
    )

    positive_mask = sim.freqs > 0
    magnitudes = np.abs(sim.X[sim.active_indices][:, positive_mask])

    lower = component_amplitude * (1.0 - magnitude_jitter)
    upper = component_amplitude * (1.0 + magnitude_jitter)

    assert np.all(magnitudes >= lower - 1e-12)
    assert np.all(magnitudes <= upper + 1e-12)


def test_simulate_from_frequency_domain_matches_time_domain_path():
    sampling_rate = 1000.0
    duration = 0.05
    seed = 123

    mics, sources_td = construct_geometry(
        num_mics=3,
        array_type="circular",
        num_sources=4,
        source_distance=0.5,
        mic_radius=0.1,
    )
    _, sources_fd = construct_geometry(
        num_mics=3,
        array_type="circular",
        num_sources=4,
        source_distance=0.5,
        mic_radius=0.1,
    )

    active = [0, 2]
    signal_rng = np.random.default_rng(0)

    sources_td, _ = generate_signals(
        sources_td,
        active_indices=active,
        sampling_rate=sampling_rate,
        duration=duration,
        mode="noise",
        rng=signal_rng,
        amplitude=1.0,
    )

    x = np.array([src.time_series for src in sources_td])
    X = np.array(fft(x, axis=1))

    sim_td = simulate_from_time_domain(
        mics=mics,
        sources=sources_td,
        active_indices=active,
        sampling_rate=sampling_rate,
        duration=duration,
        seed=seed,
        sensor_snr_db=20.0,
        model_snr_db=25.0,
        inverse_method="mp",
    )

    sim_fd = simulate_from_frequency_domain(
        mics=mics,
        sources=sources_fd,
        active_indices=active,
        X=X,
        sampling_rate=sampling_rate,
        duration=duration,
        seed=seed,
        sensor_snr_db=20.0,
        model_snr_db=25.0,
        inverse_method="mp",
    )

    assert_allclose(sim_fd.X, sim_td.X, atol=1e-12)
    assert_allclose(sim_fd.A, sim_td.A, atol=1e-12)
    assert_allclose(sim_fd.Y_clean, sim_td.Y_clean, atol=1e-12)
    assert_allclose(sim_fd.eta, sim_td.eta, atol=1e-12)
    assert_allclose(sim_fd.delta, sim_td.delta, atol=1e-12)
    assert_allclose(sim_fd.Y, sim_td.Y, atol=1e-12)
    assert_allclose(sim_fd.X_pinv, sim_td.X_pinv, atol=1e-12)
    assert_allclose(sim_fd.x, sim_td.x, atol=1e-12)
    assert_allclose(sim_fd.freqs, sim_td.freqs, atol=1e-12)
    assert sim_fd.active_indices == sim_td.active_indices


def test_simulate_from_frequency_domain_rejects_non_conjugate_symmetric_spectrum():
    sampling_rate = 1000.0
    duration = 0.05

    mics, sources = construct_geometry(
        num_mics=3,
        array_type="circular",
        num_sources=4,
        source_distance=0.5,
        mic_radius=0.1,
    )

    N = int(sampling_rate * duration)
    X = np.zeros((len(sources), N), dtype=complex)
    X[0, 1] = 1.0 + 2.0j

    with pytest.raises(ValueError, match="conjugate-symmetric"):
        simulate_from_frequency_domain(
            mics=mics,
            sources=sources,
            active_indices=[0],
            X=X,
            sampling_rate=sampling_rate,
            duration=duration,
            seed=0,
            inverse_method="mp",
        )
