import sys
from pathlib import Path
import seaborn as sns
from typeguard import typechecked
from joblib import Parallel, delayed
import numpy as np


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
def plot_wrong_predictions(X_true: np.ndarray, X_estimated: np.ndarray, tol: float):
    wrong_indices = wrong_predictions(X_true, X_estimated, tol)
    wrong_vector = np.zeros_like(X_true, dtype=float)
    wrong_vector[list(wrong_indices)] = -1.0
    plot_equation(
        X_true,
        X_estimated,
        wrong_vector,
        titles=("True", "Estimated", f"{len(wrong_indices)} wrong (tol: {tol:.2f})"),
    )


@typechecked
def tensor_wrong_detected_sources(
    method1,
    method2,
    microphones: list[int],
    sources: list[int],
    sparsities: list[int],
    seeds: int = 10,
    alpha=1e-3,
) -> np.ndarray:
    """
    Computes the number of wrongly detected sources for two methods and the pseudoinverse across different numbers of microphones, sources, and sparsity levels.
    Returns a tensor of shape (len(microphones), len(sources), len(sparsities)) where each entry contains the average number of wrongly detected sources across the specified number of seeds.
    """

    results = {}

    for mic in microphones:
        for source in sources:
            for sparsity in sparsities:
                wrong_counts = []
                for _ in range(seeds):
                    Y, A, X0, X_TRUE = just_YAX_from_simulation(
                        num_mics=mic,
                        num_sources=source,
                        s_sparse=sparsity,
                    )
                    tol = noise_threshold(X0, tolerance_factor=0.1)
                    X_method1 = method1(Y=Y, A=A, alpha=alpha)
                    X_method2 = method2(Y=Y, A=A, alpha=alpha)
                    wrong_X0 = len(wrong_predictions(X_TRUE, X0, tol))
                    wrong1 = len(wrong_predictions(X_TRUE, X_method1, tol))
                    wrong2 = len(wrong_predictions(X_TRUE, X_method2, tol))
                    wrong_counts.append((wrong_X0, wrong1, wrong2))
                avg_wrong_X0 = np.mean([wc[0] for wc in wrong_counts])
                avg_wrong1 = np.mean([wc[1] for wc in wrong_counts])
                avg_wrong2 = np.mean([wc[2] for wc in wrong_counts])
                results[(mic, source, sparsity)] = (
                    avg_wrong_X0,
                    avg_wrong1,
                    avg_wrong2,
                )
                results[(mic, source, sparsity)] = {
                    "avg_wrong_X0": avg_wrong_X0,
                    "avg_wrong_method1": avg_wrong1,
                    "avg_wrong_method2": avg_wrong2,
                }
    return results


def heatmap_average_wrong_sources(
    method1,
    method2,
    mic_range,
    source_range,
    sparsity_range,
    seeds=10,
    alpha=1e-3,
    debug=False,
):
    results = tensor_wrong_detected_sources(
        method1=method1,
        method2=method2,
        microphones=mic_range,
        sources=source_range,
        sparsities=sparsity_range,
        seeds=seeds,
        alpha=alpha,
    )
    # Process results into a format suitable for heatmap plotting
    heatmap_data = np.zeros((len(mic_range), len(source_range)))
    for i, mic in enumerate(mic_range):
        for j, source in enumerate(source_range):
            avg_wrong_method1 = np.mean(
                [
                    results[(mic, source, sparsity)]["avg_wrong_method1"]
                    for sparsity in sparsity_range
                ]
            )
            heatmap_data[i, j] = avg_wrong_method1
            if debug:
                print(
                    f"Mic: {mic}, Source: {source}, Avg Wrong Method 1: {avg_wrong_method1}"
                )

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        heatmap_data,
        xticklabels=source_range,
        yticklabels=mic_range,
        annot=True,
        fmt=".2f",
        cmap="viridis",
    )
    plt.xlabel("Number of Sources")
    plt.ylabel("Number of Microphones")
    plt.title("Average Number of Wrongly Detected Sources (Method 1)")
    plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from cs_priors.simulation.mixing_model import run_simulation
    from cs_priors.plotting.plotting import plot_equation, plot_matrices
    from cs_priors.solvers.complex_lasso import complex_lasso

    sim = run_simulation(num_sources=10, num_mics=5, s_sparse=3)
    Y = sim.Y[:, 1]
    A = sim.C[:, :, 1]
    X0 = np.linalg.pinv(A) @ Y
    X_TRUE = sim.X[:, 1]

    X_lasso = complex_lasso(Y=Y, A=A, alpha=0.1)
    tol = noise_threshold(X0, tolerance_factor=0.1)
    # plot_wrong_predictions(X_TRUE, X_lasso, tol)
    heatmap_average_wrong_sources(
        method1=complex_lasso,
        method2=sparse_prior_solution,
        mic_range=[3, 5, 7],
        source_range=[5, 10, 15],
        sparsity_range=[2, 4, 6],
        seeds=5,
        alpha=0.1,
        debug=True,
    )
