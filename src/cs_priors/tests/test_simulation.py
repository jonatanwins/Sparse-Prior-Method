"""
Regression / correctness tests for the simulation pipeline.

Run with:  pytest src/cs_priors/testing/test_simulation.py -v
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.fft import fft, fftfreq

from ..constants import SPEED_OF_SOUND
from ..geometry.arrays import circular_array, linear_array
from ..geometry.SoundSource import SoundSource
from ..simulation.mixing_model import (
    calculate_delays,
    just_YAX_from_simulation,
    run_simulation,
    s_sparse_sources,
    single_waveform_at_all_mics,
    waveforms_at_mics,
)
from ..simulation.Simulation import Simulation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_sim():
    """Overdetermined single-frequency simulation: 5 mics, 3 sources, 2 active."""
    return run_simulation(
        array_type="circular",
        num_mics=5,
        spacing=0.1,
        num_sources=3,
        s_sparse=2,
        f0=100,
        distance=1.5,
        amplitude=2,
        phase_step=0.3,
        seed=42,
    )


@pytest.fixture
def full_sim():
    """All sources active, no sparsity, 6 mics, 4 sources."""
    return run_simulation(
        array_type="circular",
        num_mics=6,
        spacing=0.1,
        num_sources=4,
        f0=200,
        distance=1.0,
        amplitude=1.0,
        phase_step=0.0,
        seed=0,
    )


# ---------------------------------------------------------------------------
# 1. Return type and shape checks
# ---------------------------------------------------------------------------

class TestSimulationShapes:
    def test_returns_simulation_dataclass(self, simple_sim):
        assert isinstance(simple_sim, Simulation)

    def test_time_vector_length(self, simple_sim):
        assert len(simple_sim.t) == simple_sim.N

    def test_x_time_shape(self, simple_sim):
        num_sources = len(simple_sim.sources)
        assert simple_sim.x_time.shape == (num_sources, simple_sim.N)

    def test_y_time_shape(self, simple_sim):
        num_mics = len(simple_sim.mics)
        assert simple_sim.y_time.shape == (num_mics, simple_sim.N)

    def test_X_shape(self, simple_sim):
        num_sources = len(simple_sim.sources)
        assert simple_sim.X.shape == (num_sources, simple_sim.N)

    def test_Y_shape(self, simple_sim):
        num_mics = len(simple_sim.mics)
        assert simple_sim.Y.shape == (num_mics, simple_sim.N)

    def test_mixing_matrix_A_shape(self, simple_sim):
        num_mics = len(simple_sim.mics)
        num_sources = len(simple_sim.sources)
        assert simple_sim.A.shape == (num_mics, num_sources, simple_sim.N)

    def test_A_pinv_shape(self, simple_sim):
        num_mics = len(simple_sim.mics)
        num_sources = len(simple_sim.sources)
        assert simple_sim.A_pinv.shape == (num_sources, num_mics, simple_sim.N)

    def test_Y_pred_shape(self, simple_sim):
        num_mics = len(simple_sim.mics)
        assert simple_sim.Y_pred.shape == (num_mics, simple_sim.N)

    def test_X_pred_shape(self, simple_sim):
        num_sources = len(simple_sim.sources)
        assert simple_sim.X_pred.shape == (num_sources, simple_sim.N)

    def test_x_pred_shape(self, simple_sim):
        num_sources = len(simple_sim.sources)
        assert simple_sim.x_pred.shape == (num_sources, simple_sim.N)

    def test_freqs_shape(self, simple_sim):
        assert simple_sim.freqs.shape == (simple_sim.N,)

    def test_mics_shape(self, simple_sim):
        assert simple_sim.mics.shape == (5, 2)


# ---------------------------------------------------------------------------
# 2. Forward model self-consistency:  Y_pred[:, f] == C[:,:,f] @ X[:, f]
# ---------------------------------------------------------------------------

class TestForwardModel:
    def test_Y_pred_equals_A_times_X(self, simple_sim):
        """Y_pred should equal the product A @ X at every frequency bin."""
        A, X, Y_pred = simple_sim.A, simple_sim.X, simple_sim.Y_pred
        N = simple_sim.N
        for f_idx in range(N):
            expected = A[:, :, f_idx] @ X[:, f_idx]
            assert_allclose(
                Y_pred[:, f_idx], expected, atol=1e-10,
                err_msg=f"Mismatch at frequency index {f_idx}",
            )

    def test_Y_pred_matches_Y_fft(self, full_sim):
        """When all sources are active the predicted Y should match the measured Y."""
        # Y is the FFT of y_time (microphone recordings)
        # Y_pred is C @ X.  They should agree if the model is self-consistent.
        assert_allclose(full_sim.Y_pred, full_sim.Y, atol=1e-8)


# ---------------------------------------------------------------------------
# 3. Pseudoinverse recovery (overdetermined: mics > sources)
# ---------------------------------------------------------------------------

class TestPseudoinverseRecovery:
    def test_X_pred_recovers_X_overdetermined(self):
        """With more mics than sources and no noise, X_pred ≈ X."""
        sim = run_simulation(
            num_mics=8,
            num_sources=3,
            f0=100,
            amplitude=2,
            phase_step=0.5,
            seed=7,
        )
        assert_allclose(sim.X_pred, sim.X, atol=1e-8)

    def test_x_pred_recovers_x_time_overdetermined(self):
        """Time-domain recovery should match original signals."""
        sim = run_simulation(
            num_mics=8,
            num_sources=3,
            f0=100,
            amplitude=2,
            phase_step=0.5,
            seed=7,
        )
        assert_allclose(sim.x_pred.real, sim.x_time, atol=1e-8)


# ---------------------------------------------------------------------------
# 4. Sparsity
# ---------------------------------------------------------------------------

class TestSparsity:
    def test_sparsity_zeroes_correct_count(self, simple_sim):
        """With s_sparse=2 out of 3 sources, exactly 1 source should be silent."""
        amplitudes = np.array([src.amplitude for src in simple_sim.sources])
        assert np.sum(amplitudes == 0) == 1
        assert np.sum(amplitudes != 0) == 2

    def test_active_indices_match_nonzero_sources(self, simple_sim):
        for idx in simple_sim.active_indices:
            assert simple_sim.sources[idx].amplitude != 0

    def test_active_indices_length(self, simple_sim):
        assert len(simple_sim.active_indices) == 2

    def test_no_sparsity_all_active(self, full_sim):
        assert len(full_sim.active_indices) == 4
        for src in full_sim.sources:
            assert src.amplitude != 0

    def test_sparsity_seed_deterministic(self):
        sim1 = run_simulation(num_sources=5, s_sparse=2, seed=99)
        sim2 = run_simulation(num_sources=5, s_sparse=2, seed=99)
        assert sim1.active_indices == sim2.active_indices
        assert_allclose(sim1.x_time, sim2.x_time)

    def test_silent_sources_have_zero_signal(self, simple_sim):
        """Zeroed-out sources should produce zero time-domain signals."""
        for i, src in enumerate(simple_sim.sources):
            if src.amplitude == 0:
                assert_allclose(
                    simple_sim.x_time[i], 0.0, atol=1e-15,
                    err_msg=f"Source {i} should be silent",
                )


# ---------------------------------------------------------------------------
# 5. Delay correctness
# ---------------------------------------------------------------------------

class TestDelays:
    def test_delay_equals_distance_over_c(self):
        mic = np.array([[0.0, 0.0]])
        src = SoundSource(distance=3.0, angle=0.0, frequency=100)
        delays = calculate_delays(mic, src)
        expected = 3.0 / SPEED_OF_SOUND
        assert_allclose(delays[0], expected, atol=1e-12)

    def test_delays_are_positive(self, simple_sim):
        for delays in simple_sim.delays_dict.values():
            assert np.all(delays > 0)

    def test_delay_varies_across_mics(self, simple_sim):
        """Different mic positions should generally yield different delays."""
        for delays in simple_sim.delays_dict.values():
            assert not np.allclose(delays, delays[0]), (
                "All delays identical — mics may be co-located"
            )

    def test_mixing_matrix_phase_from_delays(self, simple_sim):
        """A[i, j, f] should equal exp(-2j*pi*freq*delay)."""
        A = simple_sim.A
        freqs = simple_sim.freqs
        for j, delays in simple_sim.delays_dict.items():
            for i, delay in enumerate(delays):
                expected = np.exp(-2j * np.pi * freqs * delay)
                assert_allclose(A[i, j, :], expected, atol=1e-12)


# ---------------------------------------------------------------------------
# 6. Single source / single mic — simplest sanity check
# ---------------------------------------------------------------------------

class TestSingleSourceSingleMic:
    def test_single_source_waveform(self):
        freq = 200
        sim = run_simulation(
            num_mics=1,
            num_sources=1,
            f0=freq,
            amplitude=1.0,
            phase_step=0.0,
            spacing=0.0,     # mic at origin would be at (0,0) for linear
            array_type="linear",
        )
        # x_time should be a pure sinusoid at the given frequency
        t = sim.t
        expected = np.sin(2 * np.pi * freq * t)
        assert_allclose(sim.x_time[0], expected, atol=1e-12)

    def test_single_source_recovery(self):
        sim = run_simulation(
            num_mics=1,
            num_sources=1,
            f0=150,
            amplitude=3.0,
            phase_step=0.0,
            array_type="circular",
        )
        assert_allclose(sim.X_pred, sim.X, atol=1e-8)


# ---------------------------------------------------------------------------
# 7. Array type variants
# ---------------------------------------------------------------------------

class TestArrayTypes:
    def test_linear_array_runs(self):
        sim = run_simulation(
            array_type="linear", num_mics=4, spacing=0.05, num_sources=2, f0=100
        )
        assert isinstance(sim, Simulation)

    def test_circular_array_runs(self):
        sim = run_simulation(
            array_type="circular", num_mics=4, spacing=0.1, num_sources=2, f0=100
        )
        assert isinstance(sim, Simulation)

    def test_invalid_array_type_raises(self):
        with pytest.raises(ValueError, match="array_type"):
            run_simulation(array_type="hexagonal")


# ---------------------------------------------------------------------------
# 8. Multi-frequency / broadband sources
# ---------------------------------------------------------------------------

class TestBroadband:
    def test_broadband_shapes(self):
        sim = run_simulation(
            num_mics=4,
            num_sources=2,
            frequencies=[[100, 200], [150, 250]],
            amplitude=1.0,
        )
        assert sim.x_time.shape[0] == 2
        assert sim.Y.shape[0] == 4

    def test_broadband_forward_model(self):
        sim = run_simulation(
            num_mics=4,
            num_sources=2,
            frequencies=[[100, 200], [150, 250]],
            amplitude=1.0,
        )
        A, X, Y_pred = sim.A, sim.X, sim.Y_pred
        for f_idx in range(sim.N):
            expected = A[:, :, f_idx] @ X[:, f_idx]
            assert_allclose(Y_pred[:, f_idx], expected, atol=1e-10)


# ---------------------------------------------------------------------------
# 9. Per-source frequency list (f0 as list)
# ---------------------------------------------------------------------------

class TestPerSourceFrequency:
    def test_f0_list(self):
        sim = run_simulation(
            num_mics=4,
            num_sources=3,
            f0=[100, 150, 200],
            amplitude=1.0,
        )
        assert len(sim.sources) == 3
        assert sim.sources[0].frequency == 100
        assert sim.sources[2].frequency == 200

    def test_f0_list_wrong_length_raises(self):
        with pytest.raises(ValueError, match="length must match"):
            run_simulation(num_sources=3, f0=[100, 200])


# ---------------------------------------------------------------------------
# 10. Sampling parameters
# ---------------------------------------------------------------------------

class TestSamplingParameters:
    def test_N_equals_rate_times_duration(self, simple_sim):
        expected_N = int(simple_sim.sampling_rate * simple_sim.duration)
        assert simple_sim.N == expected_N

    def test_sampling_rate_is_factor_times_freq(self):
        factor = 15
        f0 = 100
        sim = run_simulation(
            f0=f0, sampling_rate_factor=factor, num_sources=1
        )
        assert sim.sampling_rate == factor * f0


# ---------------------------------------------------------------------------
# 11. just_YAX_from_simulation helper
# ---------------------------------------------------------------------------

class TestJustYAX:
    def test_returns_correct_shapes(self):
        Y, A, X0, X_TRUE = just_YAX_from_simulation(
            num_mics=4, num_sources=6, s_sparse=2, freq_index=1, seed=10,
        )
        assert Y.shape == (4, 1)
        assert A.shape == (4, 6)
        assert X0.shape == (6, 1)
        assert X_TRUE.shape == (6,)

    def test_Y_equals_A_times_X_TRUE(self):
        Y, A, X0, X_TRUE = just_YAX_from_simulation(
            num_mics=4, num_sources=6, s_sparse=3, freq_index=2, seed=5,
        )
        expected_Y = (A @ X_TRUE).reshape(-1, 1)
        assert_allclose(Y, expected_Y, atol=1e-10)

    def test_seed_deterministic(self):
        r1 = just_YAX_from_simulation(num_mics=3, num_sources=5, s_sparse=2, seed=42)
        r2 = just_YAX_from_simulation(num_mics=3, num_sources=5, s_sparse=2, seed=42)
        for a, b in zip(r1, r2):
            assert_allclose(a, b)


# ---------------------------------------------------------------------------
# 12. s_sparse_sources utility
# ---------------------------------------------------------------------------

class TestSSparseSources:
    def test_correct_number_active(self):
        sources = [
            SoundSource(distance=1.0, angle=i * 0.5, frequency=100, amplitude=1.0)
            for i in range(6)
        ]
        sparse, indices = s_sparse_sources(3, sources, seed=0)
        assert len(indices) == 3
        nonzero = sum(1 for s in sparse if s.amplitude != 0)
        assert nonzero == 3

    def test_inactive_have_zero_amplitude(self):
        sources = [
            SoundSource(distance=1.0, angle=i * 0.5, frequency=100, amplitude=2.0)
            for i in range(5)
        ]
        sparse, indices = s_sparse_sources(2, sources, seed=1)
        for i, s in enumerate(sparse):
            if i not in indices:
                assert s.amplitude == 0


# ---------------------------------------------------------------------------
# 13. Energy / Parseval's theorem sanity check
# ---------------------------------------------------------------------------

class TestEnergyConsistency:
    def test_parseval_x_time_vs_X(self, full_sim):
        """Energy in time domain ≈ energy in frequency domain (up to N scaling)."""
        for i in range(len(full_sim.sources)):
            time_energy = np.sum(np.abs(full_sim.x_time[i]) ** 2)
            freq_energy = np.sum(np.abs(full_sim.X[i]) ** 2) / full_sim.N
            assert_allclose(time_energy, freq_energy, rtol=1e-10)

    def test_parseval_y_time_vs_Y(self, full_sim):
        for i in range(len(full_sim.mics)):
            time_energy = np.sum(np.abs(full_sim.y_time[i]) ** 2)
            freq_energy = np.sum(np.abs(full_sim.Y[i]) ** 2) / full_sim.N
            assert_allclose(time_energy, freq_energy, rtol=1e-10)
