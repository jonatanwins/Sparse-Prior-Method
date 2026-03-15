import sys
from pathlib import Path
import seaborn as sns
from typeguard import typechecked
from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt
from cs_priors.plotting.plot_complex import plot_matrices
from cs_priors.plotting.plotting import plot_equation, wrapper_plot_geometry
from cs_priors.solvers.moe_group_lasso import (
    tensor_to_block_matrix,
    matrix_to_block_vector,
)


# Add the src directory to the Python path
# This allows us to import modules from cs_priors
script_path = Path(__file__).resolve()
project_root = script_path.parents[1]  # Go up one level from script to project root
src_path = project_root / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from cs_priors.solvers.vectorized_sparse_prior import (
    sparse_prior_solution,
)
from cs_priors.solvers.complex_lasso import complex_lasso
from cs_priors.simulation.mixing_model import (
    run_simulation,
    just_YAX_from_simulation,
)


@typechecked
def noise_threshold(X0: np.ndarray, tolerance_factor: float = 0.1) -> float:
    # X0 is complex valued, we need to set the threshold based on its maximum value in absolute terms
    max_absolute = np.abs(X0).max()
    threshold = np.abs(tolerance_factor * max_absolute)
    return threshold


@typechecked
def which_sources_active(X: np.ndarray, tol: float):
    # X is a numpy array (complex or real)
    # tol is the threshold for considering a source as active
    active_indices = np.where(np.abs(X) > tol)[0]
    return active_indices


@typechecked
def wrong_predictions(X_true: np.ndarray, X_estimated: np.ndarray, tol: float):
    """
    Returns the indicies of sources that are wrongly predicted as active or inactive
    """
    all_indices = set(range(len(X_true)))
    true_active = set(which_sources_active(X_true, tol))
    estimated_active = set(which_sources_active(X_estimated, tol))
    wrong_active = (true_active - estimated_active).union(
        estimated_active - true_active
    )
    return wrong_active


@typechecked
def vector_wrong_predictions(
    X_true: np.ndarray, X_estimated: np.ndarray, tol: float, plot: bool = False
):
    wrong_indices = wrong_predictions(X_true, X_estimated, tol)
    wrong_vector = np.zeros_like(X_true, dtype=float)
    wrong_vector[list(wrong_indices)] = -1.0
    if not plot:
        return wrong_vector
    if plot:
        plot_equation(
            X_true,
            X_estimated,
            wrong_vector,
            titles=(
                "True",
                "Estimated",
                f"{len(wrong_indices)} wrong (tol: {tol:.2f})",
            ),
        )


@typechecked
def tensor_wrong_detected_sources(
    method1,
    name,
    microphones: list[int],
    sources: list[int],
    sparsities: list[int],
    seeds: int = 10,
    debug: bool = False,
    frequency_idx: int | None = 1,
    freq_interval: tuple[int, int] | None = None,
) -> dict:
    """
    Computes the number of wrongly detected sources for a method and the pseudoinverse across different numbers of microphones, sources, and sparsity levels.
    Returns a tensor of shape (len(microphones), len(sources), len(sparsities)) where each entry contains the average number of wrongly detected sources across the specified number of seeds.
    """

    results = {}

    for mic in microphones:
        for source in sources:
            for sparsity in sparsities:
                wrong_counts = []
                for seed in range(seeds):

                    # 1. Block structure using all frequencies
                    if freq_interval is not None:
                        all_frequencies = []
                        np.random.seed(seed)  # Set seed for reproducibility
                        for _ in range(source):
                            count = np.random.randint(1, 10)
                            frequencies = []
                            for _ in range(count):
                                frequencies.append(
                                    np.random.randint(
                                        freq_interval[0], freq_interval[1]
                                    )
                                )
                            all_frequencies.append(frequencies)
                        sim = run_simulation(
                            num_mics=mic,
                            num_sources=source,
                            s_sparse=sparsity,
                            seed=seed,
                            frequencies=all_frequencies,
                            angle_step=2 * np.pi / source,
                        )
                        Y = matrix_to_block_vector(sim.Y)
                        A = tensor_to_block_matrix(sim.A)
                        X0 = np.linalg.pinv(A) @ Y
                        X_TRUE = matrix_to_block_vector(sim.X)

                    # 2. Single frequency index
                    elif frequency_idx is not None:
                        sim = run_simulation(
                            num_mics=mic,
                            num_sources=source,
                            s_sparse=sparsity,
                            seed=seed,
                            angle_step=2 * np.pi / source,
                        )
                        Y = sim.Y[:, frequency_idx]
                        A = sim.A[:, :, frequency_idx]
                        X0 = np.linalg.pinv(A) @ Y
                        X_TRUE = sim.X[:, frequency_idx]

                    # 3. Error if neither frequency_idx nor freq_interval is provided
                    else:
                        raise ValueError(
                            "Either frequency_idx or freq_interval must be provided"
                        )

                    # 4. Compute the single wrong count
                    tol = noise_threshold(X0, tolerance_factor=0.1)
                    X_method1 = method1(Y=Y, A=A)
                    wrong_X0 = len(wrong_predictions(X_TRUE, X0, tol))
                    wrong1 = len(wrong_predictions(X_TRUE, X_method1, tol))
                    wrong_counts.append((wrong_X0, wrong1))

                    # 5. Debugging
                    if debug:
                        # Plot geometry once
                        if (
                            seed == 0
                            and mic == microphones[0]
                            and source == sources[0]
                            and sparsity == sparsities[0]
                        ):
                            wrapper_plot_geometry(sim, skip_labels=True)

                        # when the method is significantly worse than the pseudoinverse, plot the wrong predictions
                        if wrong1 > 1.1 * wrong_X0 + 2:
                            # which indices are wrong
                            vector_wrong1 = vector_wrong_predictions(
                                X_TRUE, X_method1, tol, plot=False
                            )
                            vector_wrong_X0 = vector_wrong_predictions(
                                X_TRUE, X0, tol, plot=False
                            )
                            plot_matrices(
                                [
                                    X0,
                                    vector_wrong_X0,
                                    X_TRUE,
                                    X_method1,
                                    vector_wrong1,
                                    X_TRUE,
                                ],
                                titles=[
                                    "X0",
                                    f"{wrong_X0} wrong in X0",
                                    "X_true",
                                    name,
                                    f"{wrong1} wrong in {name}",
                                    "X_true",
                                ],
                                polar=True,
                                font_size=10,
                            )
                avg_wrong_X0 = np.mean([wc[0] for wc in wrong_counts])
                avg_wrong1 = np.mean([wc[1] for wc in wrong_counts])
                # results[(mic, source, sparsity)] = (
                #     avg_wrong_X0,
                #     avg_wrong1,
                # )
                results[(mic, source, sparsity)] = {
                    "avg_wrong_X0": avg_wrong_X0,
                    name: avg_wrong1,
                }
    return results


def heatmap_average_wrong_sources(
    method1,
    name,
    mic_range,
    source_range,
    sparsity_range,
    seeds=10,
    debug=False,
    vmin=None,
    vmax=None,
    frequency_idx: int | None = 1,
    freq_interval: tuple[int, int] | None = None,
):
    results = tensor_wrong_detected_sources(
        method1=method1,
        name=name,
        microphones=mic_range,
        sources=source_range,
        sparsities=sparsity_range,
        seeds=seeds,
        debug=debug,
        frequency_idx=frequency_idx,
        freq_interval=freq_interval,
    )

    # If sources is fixed (length 1), create heatmap: mics vs sparsity
    if len(source_range) == 1:
        source = source_range[0]
        heatmap_data = np.zeros((len(mic_range), len(sparsity_range)))
        for i, mic in enumerate(mic_range):
            for j, sparsity in enumerate(sparsity_range):
                heatmap_data[i, j] = results[(mic, source, sparsity)][name]

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
        plt.xlabel("Sparsity")
        plt.ylabel("Number of Microphones")
        plt.title(f"{name} - Wrong Predictions (Sources = {source})")
        plt.show()
    else:
        # Create one heatmap per sparsity level
        for sparsity in sparsity_range:
            heatmap_data = np.zeros((len(mic_range), len(source_range)))
            for i, mic in enumerate(mic_range):
                for j, source in enumerate(source_range):
                    heatmap_data[i, j] = results[(mic, source, sparsity)][name]

            plt.figure(figsize=(10, 6))
            sns.heatmap(
                heatmap_data,
                xticklabels=source_range,
                yticklabels=mic_range,
                annot=True,
                fmt=".2f",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )
            plt.xlabel("Number of Sources")
            plt.ylabel("Number of Microphones")
            plt.title(f"{name} - Sparsity = {sparsity}")
            plt.show()


if __name__ == "__main__":

    from cs_priors.simulation.mixing_model import run_simulation

    from cs_priors.solvers.complex_lasso import complex_lasso

    sim = run_simulation(num_sources=10, num_mics=5, s_sparse=3)
    Y = sim.Y[:, 1]
    A = sim.A[:, :, 1]
    X0 = np.linalg.pinv(A) @ Y
    X_TRUE = sim.X[:, 1]

    X_lasso = complex_lasso(Y=Y, A=A, alpha=0.1)
    tol = noise_threshold(X0, tolerance_factor=0.1)
    # plot_wrong_predictions(X_TRUE, X_lasso, tol)
    heatmap_average_wrong_sources(
        method1=lambda Y, A: sparse_prior_solution(Y=Y, A=A),
        name="Sparse Prior method",
        mic_range=[3, 5, 7, 10],
        source_range=[10],
        sparsity_range=[0, 1, 2, 3],
        seeds=50,
        debug=False,
    )
