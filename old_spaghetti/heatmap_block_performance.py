import sys
from pathlib import Path
import seaborn as sns
from typeguard import typechecked
from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt


# Add the src directory to the Python path
# This allows us to import modules from cs_priors
script_path = Path(__file__).resolve()
project_root = script_path.parents[1]  # Go up one level from script to project root
src_path = project_root / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from cs_priors.plotting.plot_complex import plot_matrices
from cs_priors.plotting.plotting import plot_equation, wrapper_plot_geometry
from cs_priors.solvers.moe_group_lasso import (
    tensor_to_block_matrix,
    matrix_to_block_vector,
    block_vector_to_matrix,
)

from cs_priors.solvers.vectorized_sparse_prior import (
    sparse_prior_solution,
)
from cs_priors.solvers.complex_lasso import complex_lasso
from cs_priors.simulation.mixing_model import (
    run_simulation,
)


@typechecked
def noise_threshold(X_matrix: np.ndarray, tolerance_factor: float = 0.1) -> float:
    """
    Compute detection threshold based on maximum signal energy.

    Args:
        X_matrix: Complex source matrix (N, F)
        tolerance_factor: Fraction of max energy to use as threshold

    Returns:
        Detection threshold
    """
    max_absolute = np.abs(X_matrix).max()
    threshold = tolerance_factor * max_absolute
    return threshold


@typechecked
def which_sources_active(X_matrix: np.ndarray, tol: float) -> np.ndarray:
    """
    Determine which sources are active based on their energy across frequencies.

    Args:
        X_matrix: Complex matrix of shape (N, F) where N=sources, F=frequencies
        tol: Threshold for considering a source active

    Returns:
        Indices of active sources (0 to N-1)
    """
    # Check max absolute value across all frequencies for each source
    source_max_energy = np.max(np.abs(X_matrix), axis=1)
    active_indices = np.where(source_max_energy > tol)[0]
    return active_indices


@typechecked
def count_wrong_sources(
    X_true: np.ndarray,
    X_estimated: np.ndarray,
    tol: float,
) -> int:
    """
    Count sources that are wrongly predicted as active or inactive.

    Args:
        X_true: True source matrix (N, F)
        X_estimated: Estimated source matrix (N, F)
        tol: Detection threshold

    Returns:
        Number of incorrectly detected sources
    """
    true_active = set(which_sources_active(X_true, tol))
    estimated_active = set(which_sources_active(X_estimated, tol))
    wrong_active = (true_active - estimated_active).union(
        estimated_active - true_active
    )
    return len(wrong_active)


@typechecked
def get_wrong_source_indices(
    X_true: np.ndarray,
    X_estimated: np.ndarray,
    tol: float,
) -> set[int]:
    """
    Get indices of sources with incorrect detection status.

    Args:
        X_true: True source matrix (N, F)
        X_estimated: Estimated source matrix (N, F)
        tol: Detection threshold

    Returns:
        Set of source indices that are incorrectly detected
    """
    true_active = set(which_sources_active(X_true, tol))
    estimated_active = set(which_sources_active(X_estimated, tol))
    wrong_indices = (true_active - estimated_active).union(
        estimated_active - true_active
    )
    return {int(idx) for idx in wrong_indices}


def _compute_single_config(
    method,
    method_name: str,
    mics: int,
    sources: int,
    sparsity: int,
    seeds: int,
    max_freqs_per_source: int,
    freq_interval: tuple[int, int],
    seed_counter_start: int,
    sampling_rate_factor: float = 10.0,
    debug: bool = False,
):
    """
    Compute detection errors for a single (mics, sources, sparsity) configuration.
    This function is designed to be parallelized.
    """
    sum_wrong_X0 = 0
    sum_wrong_method = 0
    seed_counter = seed_counter_start

    for _ in range(seeds):
        all_frequencies = []
        np.random.seed(seed_counter)
        for s in range(sources):
            count = np.random.randint(1, max_freqs_per_source + 1)
            frequencies = []
            for _ in range(count):
                frequencies.append(
                    np.random.randint(freq_interval[0], freq_interval[1])
                )
            all_frequencies.append(frequencies)

        sim = run_simulation(
            num_mics=mics,
            num_sources=sources,
            s_sparse=sparsity,
            frequencies=all_frequencies,
            seed=seed_counter,
            angle_step=2 * np.pi / sources,
            sampling_rate_factor=sampling_rate_factor,
        )
        num_sources_local, num_frequencies = sim.X.shape

        # Compute baseline (pseudoinverse)
        A_block = tensor_to_block_matrix(sim.A)
        Y_block = matrix_to_block_vector(sim.Y)
        X0_matrix = block_vector_to_matrix(
            np.linalg.pinv(A_block) @ Y_block, sources, num_frequencies
        )

        # Compute method estimate
        X_method = method(Y_matrix=sim.Y, A_tensor=sim.A)

        # Evaluate performance
        tol = noise_threshold(X0_matrix, tolerance_factor=0.1)
        wrong_X0 = count_wrong_sources(sim.X, X0_matrix, tol)
        wrong_method = count_wrong_sources(sim.X, X_method, tol)

        sum_wrong_X0 += wrong_X0
        sum_wrong_method += wrong_method

        seed_counter += 1

        if debug:
            if (
                wrong_method > 1.1 * wrong_X0 + 2
                or wrong_method > num_sources_local * 0.8
            ):
                wrong_indices_method = get_wrong_source_indices(sim.X, X_method, tol)
                wrong_indices_X0 = get_wrong_source_indices(sim.X, X0_matrix, tol)

                # Create indicator matrices for visualization
                wrong_mask_method = np.zeros_like(sim.X, dtype=float)
                wrong_mask_X0 = np.zeros_like(sim.X, dtype=float)
                for idx in wrong_indices_method:
                    max_col = int(np.argmax(np.abs(sim.X[idx, :])))
                    wrong_mask_method[idx, max_col] = -1.0
                for idx in wrong_indices_X0:
                    max_col = int(np.argmax(np.abs(sim.X[idx, :])))
                    wrong_mask_X0[idx, max_col] = -1.0

                plot_matrices(
                    [
                        X0_matrix,
                        wrong_mask_X0,
                        sim.X,
                        X_method,
                        wrong_mask_method,
                        sim.X,
                    ],
                    titles=[
                        "X0 (Pseudoinverse)",
                        f"{wrong_X0} wrong in X0",
                        "X_true",
                        method_name,
                        f"{wrong_method} wrong in {method_name}",
                        "X_true",
                    ],
                    show_values=True,
                )

    avg_wrong_X0 = sum_wrong_X0 / seeds
    avg_wrong_method = sum_wrong_method / seeds

    return {
        "config": (mics, sources, sparsity),
        "avg_wrong_X0": avg_wrong_X0,
        "avg_wrong_method": avg_wrong_method,
    }


@typechecked
def compute_source_detection_errors(
    method,
    method_name: str,
    mic_range: list[int],
    source_range: list[int],
    sparsity_range: list[int],
    seeds: int,
    max_freqs_per_source: int,
    freq_interval: tuple[int, int],
    sampling_rate_factor: float = 10.0,
    debug=False,
    n_jobs=-1,
) -> dict[tuple[int, int, int], dict[str, float]]:
    """
    Compute source detection errors using parallel execution.

    Args:
        method: Solver function
        method_name: Name of method
        mic_range: List of microphone counts to test
        source_range: List of source counts to test
        sparsity_range: List of sparsity levels to test
        seeds: Number of random seeds per configuration
        max_freqs_per_source: Max frequencies per source
        freq_interval: Frequency range (min, max)
        debug: Enable debug plotting
        n_jobs: Number of parallel jobs (-1 = all cores)

    Returns:
        Dictionary of results indexed by (mics, sources, sparsity)
    """
    # Build list of configurations
    configs = []
    seed_counter = seeds
    for mics in mic_range:
        for sources in source_range:
            for sparsity in sparsity_range:
                configs.append(
                    {
                        "mics": mics,
                        "sources": sources,
                        "sparsity": sparsity,
                        "seed_counter_start": seed_counter,
                    }
                )
                seed_counter += seeds  # Each config gets its own seed block

    # Run computations in parallel
    results_list = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(_compute_single_config)(
            method=method,
            method_name=method_name,
            mics=cfg["mics"],
            sources=cfg["sources"],
            sparsity=cfg["sparsity"],
            seeds=seeds,
            max_freqs_per_source=max_freqs_per_source,
            freq_interval=freq_interval,
            seed_counter_start=cfg["seed_counter_start"],
            sampling_rate_factor=sampling_rate_factor,
            debug=debug,
        )
        for cfg in configs
    )

    # Aggregate results
    results = {}
    for res in results_list:
        mics, sources, sparsity = res["config"]
        results[(mics, sources, sparsity)] = {
            "avg_wrong_X0": res["avg_wrong_X0"],
            method_name: res["avg_wrong_method"],
        }
        print(
            f"  → Avg wrong: X0={res['avg_wrong_X0']:.2f}, {method_name}={res['avg_wrong_method']:.2f} for mics={mics}, sources={sources}, sparsity={sparsity}"
        )

    return results


def heatmap_average_wrong_sources(
    method: object,
    mic_range: list[int],
    sources: int,
    sparsity_range: list[int],
    method_name: str | None = None,
    freq_interval: tuple[int, int] = (100, 500),
    max_freqs_per_source=5,
    seeds=10,
    vmin=None,
    vmax=None,
    debug=False,
):
    """
    For a set number of sources, plot a heatmap of the average number of wrong sources detected by the method across different numbers of microphones and sparsity levels.
    The method should be a function that takes Y and A as input and returns an estimate of X.
    The heatmap will have microphones on one axis and sparsity on the other, with the color representing the average number of wrong sources detected.
    """
    if method_name is None:
        name = method.__name__
    else:
        name = method_name

    results = compute_source_detection_errors(
        method=method,
        method_name=name,
        mic_range=mic_range,
        source_range=[sources],
        sparsity_range=sparsity_range,
        seeds=seeds,
        max_freqs_per_source=max_freqs_per_source,
        freq_interval=freq_interval,
        debug=debug,
    )

    heatmap_data = np.array(
        [
            [results[(mics, sources, sparsity)][name] for sparsity in sparsity_range]
            for mics in mic_range
        ]
    )
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        heatmap_data,
        xticklabels=sparsity_range,
        yticklabels=mic_range,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    plt.title(f"Average Wrong Sources Detected by {name} (Sources={sources})")
    plt.xlabel("Sparsity Level")
    plt.ylabel("Number of Microphones")
    plt.show()


def run_basic_detection_tests() -> None:
    """
    Simple sanity checks for noise_threshold, which_sources_active,
    count_wrong_sources, and get_wrong_source_indices.
    """
    # Build a tiny synthetic example (N=3 sources, F=4 frequencies)
    X_true = np.array(
        [
            [1 + 0j, 0.5 + 0j, 0.2 + 0j, 0.0 + 0j],
            [0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j],
            [0.9 + 0j, 0.8 + 0j, 0.0 + 0j, 0.0 + 0j],
        ]
    )
    X_est = np.array(
        [
            [0.9 + 0j, 0.1 + 0j, 0.0 + 0j, 0.0 + 0j],
            [0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j],
            [0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j],
        ]
    )

    tol = noise_threshold(X_true, tolerance_factor=0.2)
    active_true = which_sources_active(X_true, tol)
    active_est = which_sources_active(X_est, tol)
    wrong_count = count_wrong_sources(X_true, X_est, tol)
    wrong_indices = get_wrong_source_indices(X_true, X_est, tol)

    print("[Basic Tests]")
    print(f"  tol = {tol:.3f}")
    print(f"  active_true = {active_true}")
    print(f"  active_est = {active_est}")
    print(f"  wrong_count = {wrong_count}")
    print(f"  wrong_indices = {sorted(wrong_indices)}")


if __name__ == "__main__":

    def pseudo_inverse_solution(Y_matrix, A_tensor):
        A_block = tensor_to_block_matrix(A_tensor)
        Y_block = matrix_to_block_vector(Y_matrix)
        X_pinv = block_vector_to_matrix(
            np.linalg.pinv(A_block) @ Y_block, A_tensor.shape[1], A_tensor.shape[2]
        )
        return X_pinv

    def complex_pair_solution(Y_matrix, A_tensor):
        from group_lasso import GroupLasso
        from cs_priors.solvers.real_augmented import (
            from_real_augmented,
            to_real_augmented,
        )

        A = tensor_to_block_matrix(A_tensor)
        Y = matrix_to_block_vector(Y_matrix)
        # Group real and imaginary parts together for each frequency bin
        num_src_x_freq = A.shape[1]
        groups = np.concatenate(
            [np.arange(num_src_x_freq), np.arange(num_src_x_freq)]
        )  # group real and imag parts of the same frequency together
        alpha = 1e-4
        group_reg = A.shape[0] * alpha  # samples are mics x frequencies
        max_iter = 20000

        model = GroupLasso(
            groups=groups, group_reg=group_reg, n_iter=max_iter, supress_warning=True
        )
        X_cpair = block_vector_to_matrix(
            from_real_augmented(
                model.fit(to_real_augmented(A), to_real_augmented(Y)).coef_.reshape(
                    -1, 1
                )
            ),
            A_tensor.shape[1],
            A_tensor.shape[2],
        )
        return X_cpair

    mic_range = [3, 5, 8]
    source_range = [10]
    sparsity_range = [1, 2, 3]
    seeds = 3
    max_freqs_per_source = 3
    freq_interval = (100, 500)

    heatmap_average_wrong_sources(
        method=complex_pair_solution,
        mic_range=mic_range,
        sources=source_range[0],
        sparsity_range=sparsity_range,
        seeds=seeds,
        max_freqs_per_source=max_freqs_per_source,
        freq_interval=freq_interval,
        debug=True,
    )
