import sys
from pathlib import Path
import time
import scipy.optimize as optimize
import seaborn as sns
from sklearn.linear_model import Lasso
from typeguard import typechecked
from joblib import Parallel, delayed


# Add the src directory to the Python path
# This allows us to import modules from cs_priors
script_path = Path(__file__).resolve()
project_root = script_path.parents[1]  # Go up two levels from script to project root
src_path = project_root / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import numpy as np
import random
import matplotlib.pyplot as plt
from cs_priors.simulation.mixing_model import run_simulation
from cs_priors.plotting.plotting import (
    plot_equation,
    plot_two_line_equation,
    plot_overview,
)

from cs_priors.solvers.vectorized_sparse_prior import (
    to_real_augmented,
    from_real_augmented,
    sparse_prior_solution,
)
from cs_priors.solvers.complex_lasso import complex_lasso


def initialize_underdetermined_system(
    num_mics=8,
    num_sources=10,
    sparsity=2,
    amplitude=10,
    complex_valued=False,
    seed=2026,
):

    # set the random seed for reproducibility
    np.random.seed(seed)
    if complex_valued:
        X_true = amplitude * (
            np.random.randn(num_sources) + 1j * np.random.randn(num_sources)
        )
    else:
        X_true = amplitude * np.random.randn(num_sources)

    # choose sparse indicies
    zero_indices = np.random.choice(num_sources, num_sources - sparsity, replace=False)
    X_true[zero_indices] = 0  # make X sparse

    if complex_valued:
        A = np.random.randn(num_mics, num_sources) + 1j * np.random.randn(
            num_mics, num_sources
        )
    else:
        A = np.random.randn(num_mics, num_sources)
    Y = A @ X_true

    return X_true, A, Y


def just_YAX_from_simulation(  # moved to mixing_model.py
    num_mics=3,
    num_sources=5,
    s_sparse=2,
    freq_index=1,
    angle_step=0.3,
    angle_base=np.pi / 4,
    phase_step=0.3,
):
    sim = run_simulation(
        num_mics=num_mics,
        num_sources=num_sources,
        s_sparse=s_sparse,
        angle_step=angle_step,
        angle_base=angle_base,
        phase_step=phase_step,
    )
    Y = sim.Y[:, freq_index]  # Measurements
    A = sim.C[:, :, freq_index]  # Mixing matrix
    X0 = np.linalg.pinv(A) @ Y  # initial guess for X
    X_TRUE = sim.X[:, freq_index]  # True source signals
    return Y, A, X0, X_TRUE


@typechecked
def noise_threshold(X0, tolerance_factor: float = 0.1):
    # X0 is complex valued, we need to set the threshold based on its maximum value in absolute terms
    max_absolute = np.abs(X0).max()
    threshold = np.abs(tolerance_factor * max_absolute)
    return threshold


def count_nonzero(x, tol: float):
    return np.sum(np.abs(x) > tol)


def nonzero_difference(X1, X2, tol: float):
    # X1 and X2 are numpy arrays
    s_X1 = count_nonzero(X1, tol)
    s_X2 = count_nonzero(X2, tol)
    return np.abs(s_X1 - s_X2)


def run_comparison_sparsity(
    mic_range=[], source_range=[], sparsity_range=[], seed=1, alpha=1e-3
):
    np.random.seed(seed)
    results = {}

    for num_mics in mic_range:
        for num_sources in source_range:
            for s_sparse in sparsity_range:
                sim = run_simulation(
                    num_mics=num_mics, num_sources=num_sources, s_sparse=s_sparse
                )
                freq_index = 1
                Y = sim.Y[:, freq_index]  # Measurements
                A = sim.C[:, :, freq_index]  # Mixing matrix
                X0 = np.linalg.pinv(A) @ Y  # initial guess for X
                X_TRUE = sim.X[:, freq_index]  # True source signals

                X_sp, B = sparse_prior_solution(X0, A)
                X_lasso = complex_lasso(A, Y, alpha=alpha)

                threshold = noise_threshold(X0)

                error_sp = nonzero_difference(X_TRUE, X_sp, tol=threshold)
                error_lasso = nonzero_difference(X_TRUE, X_lasso, tol=threshold)

                results[(num_mics, num_sources, s_sparse)] = {
                    "num_mics": num_mics,
                    "num_sources": num_sources,
                    "s_sparse": s_sparse,
                    "X_TRUE": X_TRUE,
                    "X_sp": X_sp,
                    "X_lasso": X_lasso,
                    "error_sp": error_sp,
                    "error_lasso": error_lasso,
                }

    return results


# Legacy
def heatmap_sparsity_comparison(results, s_sparse):
    """
    This function is legacy and only used in LASSO_comparison.ipynb
    It creates heatmaps comparing the error in number of non-zeros for
    Sparse Prior and LASSO methods across different numbers of microphones
    and sources, for a fixed sparsity level s_sparse.

    It does not take the average over multiple seeds, the proceeding functions do that.
    """
    import pandas as pd

    df = pd.DataFrame.from_dict(results, orient='index')

    # Filter to only include rows with the specified s_sparse value
    df = df[df['s_sparse'] == s_sparse]

    pivot_sp = df.pivot_table(
        index="num_sources", columns="num_mics", values="error_sp"
    )
    pivot_lasso = df.pivot_table(
        index="num_sources", columns="num_mics", values="error_lasso"
    )

    # Set common color scale across both heatmaps
    vmin = min(pivot_sp.min().min(), pivot_lasso.min().min())
    vmax = max(pivot_sp.max().max(), pivot_lasso.max().max())

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.heatmap(pivot_sp, annot=True, fmt=".0f", cmap="YlGnBu", vmin=vmin, vmax=vmax)
    plt.title(f"Sparse Prior Error in Non-Zeros ({s_sparse} nonzero sources)")
    plt.xlabel("Number of Microphones")
    plt.ylabel("Number of Sources")

    plt.subplot(1, 2, 2)
    sns.heatmap(pivot_lasso, annot=True, fmt=".0f", cmap="YlGnBu", vmin=vmin, vmax=vmax)
    plt.title(f"LASSO Error in Non-Zeros ({s_sparse} nonzero sources)")
    plt.xlabel("Number of Microphones")
    plt.ylabel("Number of Sources")

    plt.tight_layout()
    plt.show()


def average_sparsity_comparison(
    mic_range: list[int],
    sources: int,
    sparsity_range: list[int],
    num_seeds: int,
    alpha=1e-3,
    debug=False,
):
    def compute_single_case(num_mics, s_sparse, seed):
        np.random.seed(seed)
        random.seed(seed)
        Y, A, X0, X_TRUE = just_YAX_from_simulation(
            num_mics=num_mics,
            num_sources=sources,
            s_sparse=s_sparse,
            angle_step=2 * np.pi / sources,
        )
        X_sp, B = sparse_prior_solution(X0, A)
        X_lasso = complex_lasso(A, Y, alpha=alpha)

        threshold = noise_threshold(X0)
        error_X0 = nonzero_difference(X_TRUE, X0, tol=threshold)
        error_lasso = nonzero_difference(X_TRUE, X_lasso, tol=threshold)
        error_sp = nonzero_difference(X_TRUE, X_sp, tol=threshold)

        return (num_mics, s_sparse, error_X0, error_lasso, error_sp)

    # Generate all parameter combinations
    params = [(m, s, seed)
              for m in mic_range
              for s in sparsity_range
              for seed in range(num_seeds)]

    # Parallel execution (use n_jobs=-1 for all cores)
    all_results = Parallel(n_jobs=-1)(
        delayed(compute_single_case)(m, s, seed) for m, s, seed in params
    )

    # Aggregate results
    results = {}
    for num_mics, s_sparse, e_X0, e_lasso, e_sp in all_results:
        key = (num_mics, s_sparse)
        if key not in results:
            results[key] = {'error_X0': 0, 'error_lasso': 0, 'error_sp': 0, 'count': 0}
        results[key]['error_X0'] += e_X0
        results[key]['error_lasso'] += e_lasso
        results[key]['error_sp'] += e_sp
        results[key]['count'] += 1

    # Average
    for key in results:
        count = results[key].pop('count')
        results[key] = {k: v/count for k, v in results[key].items()}

    return results


def heatmap_average(
    mic_range: list[int],
    sources: int,
    sparsity_range: list[int],
    num_seeds: int,
    alpha=1e-3,
    debug=False,
):
    results = average_sparsity_comparison(
        mic_range=mic_range,
        sources=sources,
        sparsity_range=sparsity_range,
        num_seeds=num_seeds,
        alpha=alpha,
        debug=debug,
    )
    import pandas as pd

    df = pd.DataFrame.from_dict(results, orient='index')

    # Calculate global min and max across all methods for consistent color scale
    all_values = []
    for method in ['error_X0', 'error_lasso', 'error_sp']:
        pivot_table = df.pivot_table(
            index=df.index.get_level_values(0),
            columns=df.index.get_level_values(1),
            values=method,
        )
        all_values.extend(pivot_table.values.flatten())
    vmin = np.nanmin(all_values)
    vmax = np.nanmax(all_values)

    # Plot heatmaps for each method
    for method in ['error_X0', 'error_lasso', 'error_sp']:
        pivot_table = df.pivot_table(
            index=df.index.get_level_values(0),
            columns=df.index.get_level_values(1),
            values=method,
        )
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            pivot_table, annot=True, fmt=".2f", cmap="viridis", vmin=vmin, vmax=vmax
        )
        plt.title(f"Heatmap of {method} (sources={sources})")
        plt.xlabel("Sparsity level (s_sparse)")
        plt.ylabel("Number of microphones (num_mics)")
        plt.show()


def compare_minimal_angle(
    angle_steps, num_mics, num_sources, num_seeds=1, autoset=False
):
    error_lasso = np.zeros(len(angle_steps))
    error_sp = np.zeros(len(angle_steps))
    error_X0 = np.zeros(len(angle_steps))

    for seed in range(num_seeds):
        np.random.seed(seed)
        random.seed(seed)
        for i, angle_step in enumerate(angle_steps):
            angle_base = np.random.uniform(0, 360)
            phase_step = np.random.uniform(0, 2 * np.pi)

            if autoset:
                num_sources = int(2 * np.pi / angle_step)
                num_mics = int(autoset * num_sources)
                print(f"Auto setting {num_sources} sources and {num_mics} mics")

            Y, A, X0, X_TRUE = just_YAX_from_simulation(
                num_mics=num_mics,
                num_sources=num_sources,
                s_sparse=num_sources,
                angle_step=angle_step,
                angle_base=angle_base,
                phase_step=phase_step,
            )
            X_sp, B = sparse_prior_solution(X0, A)
            X_lasso = complex_lasso(A, Y, alpha=1e-3)

            error_lasso[i] = np.linalg.norm(
                X_lasso.reshape(-1, 1) - X_TRUE.reshape(-1, 1)
            ) / np.linalg.norm(X_TRUE)
            error_sp[i] = np.linalg.norm(
                X_sp.reshape(-1, 1) - X_TRUE.reshape(-1, 1)
            ) / np.linalg.norm(X_TRUE)
            error_X0[i] = np.linalg.norm(
                X0.reshape(-1, 1) - X_TRUE.reshape(-1, 1)
            ) / np.linalg.norm(X_TRUE)

        sp_marker_size = 5
        # only set the label once
        if seed == num_seeds - 1:
            plt.plot(
                angle_steps,
                error_X0,
                label="Pseudo-inverse",
                color="gray",
                linestyle="--",
            )
            plt.scatter(angle_steps, error_lasso, label="LASSO", color="orange")
            plt.scatter(
                angle_steps,
                error_sp,
                label="Sparse Prior",
                color="blue",
                marker="x",
                s=30,
            )
        else:
            plt.plot(angle_steps, error_X0, color="gray", linestyle="--")
            plt.scatter(angle_steps, error_lasso, color="orange")
            plt.scatter(
                angle_steps, error_sp, color="blue", marker="o", s=sp_marker_size
            )

    if not autoset:
        plt.xscale("log")
    plt.xlabel("Angle step (degrees)")
    plt.ylabel("Relative error")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.title("Reconstruction error vs angle step")
    plt.show()


if __name__ == "__main__":
    s_sparse = 2
    sim = run_simulation(num_mics=3, num_sources=5, s_sparse=s_sparse)

    # plot_overview(sim)

    freq_index = 1
    Y = sim.Y[:, freq_index].reshape(-1, 1)  # Measurements
    A = sim.C[:, :, freq_index]  # Mixing matrix
    X0 = (np.linalg.pinv(A) @ Y).reshape(-1, 1)  # initial guess for X
    X_TRUE = sim.X[:, freq_index].reshape(-1, 1)  # True source signals

    X_sp, B = sparse_prior_solution(X0, A)
    X_lasso = complex_lasso(A, Y, alpha=0.1)

    threshold = noise_threshold(Y)
    print(f"Noise threshold: {threshold}")
    print(f"Number of non-zeros in True X: {count_nonzero(X_TRUE, tol=threshold)}")
    print(
        f"Number of non-zeros in Sparse Prior X: {count_nonzero(X_sp, tol=threshold)}"
    )
    print(f"Number of non-zeros in LASSO X: {count_nonzero(X_lasso, tol=threshold)}")

    # Compare solutions
    plot_two_line_equation(
        X_TRUE,
        X_sp,
        X_lasso,
        ("True X", "Sparse Prior X", "LASSO X"),
        X0,
        A,
        Y,
        ("Initial X0", "Mixing Matrix A", "Measurements Y"),
    )
