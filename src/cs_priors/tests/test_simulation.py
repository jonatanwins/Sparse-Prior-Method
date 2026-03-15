"""
Tests for the frequency-domain simulation pipeline.

Run with:  pytest src/cs_priors/tests/test_simulation.py -v
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.fft import ifft

from ..constants import SPEED_OF_SOUND
from ..geometry.SoundSource import SoundSource
from ..simulation.mixing_model import (
    construct_geometry,
    select_active_sources,
    generate_signals,
    compute_delay_matrix,
    compute_mixing_matrix,
    simulate,
)
from ..simulation.Simulation import Simulation


# ---------------------------------------------------------------------------
# Helper: run a full simulation with sensible defaults
# ---------------------------------------------------------------------------


def _run(num_mics=6, num_sources=4, num_active=2, mode="sine", seed=42):
    """Convenience wrapper that runs the full pipeline."""
    mics, sources = construct_geometry(
        num_mics=num_mics,
        array_type="circular",
        mic_spacing=0.1,
        num_sources=num_sources,
        source_distance=1.5,
        angle_base=np.pi / 4,
        angle_step=0.3,
    )
    active = select_active_sources(sources, num_active, seed=seed)

    sampling_rate = 1000.0
    duration = 0.01  # 10 ms -> N = 10

    if mode == "sine":
        freqs = [[100.0]] * num_sources
        phases = [0.3 * i for i in range(num_sources)]
        generate_signals(
            sources,
            active,
            sampling_rate,
            duration,
            mode="sine",
            frequencies=freqs,
            phases=phases,
        )
    else:
        generate_signals(
            sources,
            active,
            sampling_rate,
            duration,
            mode="noise",
            seed=seed,
        )

    return simulate(mics, sources, active, sampling_rate, duration)


# ---------------------------------------------------------------------------
# 1. Shapes
# ---------------------------------------------------------------------------


class TestShapes:
    def test_simulation_dataclass(self):
        sim = _run()
        assert isinstance(sim, Simulation)

    def test_all_shapes(self):
        M, S = 6, 4
        sim = _run(num_mics=M, num_sources=S)
        N = int(sim.sampling_rate * sim.duration)
        assert sim.x.shape == (S, N)
        assert sim.X.shape == (S, N)
        assert sim.A.shape == (M, S, N)
        assert sim.Y.shape == (M, N)
        assert sim.freqs.shape == (N,)
        assert sim.mics.shape == (M, 2)


# ---------------------------------------------------------------------------
# 2. Forward model:  Y[:, n] == A[:, :, n] @ X[:, n]  for all n
# ---------------------------------------------------------------------------


class TestForwardModel:
    def test_Y_equals_A_times_X(self):
        sim = _run()
        N = sim.X.shape[1]
        for n in range(N):
            expected = sim.A[:, :, n] @ sim.X[:, n]
            assert_allclose(sim.Y[:, n], expected, atol=1e-10)


# ---------------------------------------------------------------------------
# 3. IFFT(X) recovers x  (Parseval / FFT round-trip)
# ---------------------------------------------------------------------------


class TestFFTRoundTrip:
    def test_ifft_of_X_equals_x(self):
        sim = _run()
        x_recovered = ifft(sim.X, axis=1)
        assert_allclose(x_recovered.real, sim.x, atol=1e-12)
        assert_allclose(x_recovered.imag, 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# 4. Pseudoinverse recovery: pinv(A) @ Y ≈ X  when M > S
# ---------------------------------------------------------------------------


class TestPseudoinverseRecovery:
    def test_overdetermined_recovers_X(self):
        """With more mics than sources and no noise, pinv(A) @ Y ≈ X."""
        sim = _run(num_mics=8, num_sources=3, num_active=3)
        N = sim.X.shape[1]
        X_recovered = np.zeros_like(sim.X)
        for n in range(N):
            X_recovered[:, n] = np.linalg.pinv(sim.A[:, :, n]) @ sim.Y[:, n]
        assert_allclose(X_recovered, sim.X, atol=1e-8)

    def test_overdetermined_recovers_x_time(self):
        """Time-domain recovery via IFFT of pinv solution matches x."""
        sim = _run(num_mics=8, num_sources=3, num_active=3)
        N = sim.X.shape[1]
        X_recovered = np.zeros_like(sim.X)
        for n in range(N):
            X_recovered[:, n] = np.linalg.pinv(sim.A[:, :, n]) @ sim.Y[:, n]
        x_recovered = ifft(X_recovered, axis=1)
        assert_allclose(x_recovered.real, sim.x, atol=1e-8)


# ---------------------------------------------------------------------------
# 5. Sparsity: muted sources have zero signals
# ---------------------------------------------------------------------------


class TestSparsity:
    def test_muted_sources_are_zero(self):
        sim = _run(num_sources=5, num_active=2)
        for i in range(len(sim.sources)):
            if i not in sim.active_indices:
                assert_allclose(sim.x[i], 0.0, atol=1e-15)

    def test_active_count(self):
        sim = _run(num_sources=5, num_active=2)
        assert len(sim.active_indices) == 2

    def test_seed_deterministic(self):
        sim1 = _run(seed=99)
        sim2 = _run(seed=99)
        assert sim1.active_indices == sim2.active_indices
        assert_allclose(sim1.x, sim2.x)


# ---------------------------------------------------------------------------
# 6. Delay matrix correctness
# ---------------------------------------------------------------------------


class TestDelays:
    def test_known_delay(self):
        """A source at distance d from a mic at the origin -> delay = distance / speed_of_sound."""
        mic = np.array([[0.0, 0.0]])
        src = SoundSource(distance=3.0, angle=0.0, time_series=None)
        delays = compute_delay_matrix(mic, [src])
        assert_allclose(delays[0, 0], 3.0 / SPEED_OF_SOUND, atol=1e-12)

    def test_delays_positive(self):
        sim = _run()
        delays = compute_delay_matrix(sim.mics, sim.sources)
        assert np.all(delays > 0)


# ---------------------------------------------------------------------------
# 7. Noise mode works
# ---------------------------------------------------------------------------


class TestNoiseMode:
    def test_noise_simulation_runs(self):
        sim = _run(mode="noise")
        assert isinstance(sim, Simulation)
        # active sources should have non-zero signals
        for i in sim.active_indices:
            assert np.any(sim.x[i] != 0)


# ---------------------------------------------------------------------------
# 8. Parseval's theorem: energy in time ≈ energy in freq / N
# ---------------------------------------------------------------------------


class TestEnergy:
    def test_parseval(self):
        sim = _run()
        N = sim.X.shape[1]
        for i in range(sim.x.shape[0]):
            time_energy = np.sum(np.abs(sim.x[i]) ** 2)
            freq_energy = np.sum(np.abs(sim.X[i]) ** 2) / N
            assert_allclose(time_energy, freq_energy, rtol=1e-10)
