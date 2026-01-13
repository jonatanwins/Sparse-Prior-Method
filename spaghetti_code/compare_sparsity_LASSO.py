import sys
from pathlib import Path
from pyparsing import alphas
import scipy.optimize as optimize
import seaborn as sns
from sklearn.linear_model import Lasso

# Add the src directory to the Python path
# This allows us to import modules from cs_priors
script_path = Path(__file__).resolve()
project_root = script_path.parents[1]  # Go up two levels from script to project root
src_path = project_root / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import numpy as np
import matplotlib.pyplot as plt
from cs_priors.simulation.mixing_model import run_simulation
from cs_priors.plotting.plotting import (
    plot_equation,
    plot_two_line_equation,
    plot_overview,
)
from cs_priors.solvers.sparse_prior import (
    to_real_augmented,
    from_real_augmented,
    sparse_prior_solution,
)


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


def just_YAX_from_simulation(
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


def complex_lasso(A, Y, alpha=0.1):
    """
    Solve the complex LASSO problem by converting to a real-valued problem.
    Minimize ||Y - A X||_2^2 + alpha * ||X||_1
    where A is complex, Y is complex, and X is complex.---_
    """
    # If A is full rank, we do not need LASSO
    X0 = np.linalg.pinv(A) @ Y  # initial guess for X
    mics, sources = A.shape
    if mics >= sources:
        return X0

    X0_real = to_real_augmented(X0)
    A_real = to_real_augmented(A)
    Y_real = to_real_augmented(Y)

    lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000)
    lasso.coef_ = X0_real.flatten()
    lasso.fit(A_real, Y_real)
    X_real = lasso.coef_

    X_complex = from_real_augmented(X_real)
    return X_complex


def noise_threshold(Y, A, tolerance_factor=0.1):
    X0 = np.linalg.pinv(A) @ Y
    max = X0.max()
    threshold = np.abs(tolerance_factor * max)
    return threshold


def count_nonzero(x, tol):
    return np.sum(np.abs(x) > tol)


def nonzero_difference(X1, X2, tol):
    # X1 and X2 are numpy arrays
    s_X1 = count_nonzero(X1, tol)
    s_X2 = count_nonzero(X2, tol)
    return np.abs(s_X1 - s_X2)


def run_comparison_sparsity(
    mic_range=[], source_range=[], sparsity_range=[], alpha=1e-3
):
    results = []
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

                threshold = noise_threshold(Y, A)

                error_sp = nonzero_difference(X_TRUE, X_sp, tol=threshold)
                error_lasso = nonzero_difference(X_TRUE, X_lasso, tol=threshold)

                results.append(
                    {
                        "num_mics": num_mics,
                        "num_sources": num_sources,
                        "s_sparse": s_sparse,
                        "X_TRUE": X_TRUE,
                        "X_sp": X_sp,
                        "X_lasso": X_lasso,
                        "error_sp": error_sp,
                        "error_lasso": error_lasso,
                    }
                )
    return results


def heatmap_sparsity_comparison(results, s_sparse):
    import pandas as pd

    df = pd.DataFrame(results)

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


if __name__ == "__main__":
    s_sparse = 2
    sim = run_simulation(num_mics=3, num_sources=5, s_sparse=s_sparse)

    # plot_overview(sim)

    freq_index = 1
    Y = sim.Y[:, freq_index]  # Measurements
    A = sim.C[:, :, freq_index]  # Mixing matrix
    X0 = np.linalg.pinv(A) @ Y  # initial guess for X
    X_TRUE = sim.X[:, freq_index]  # True source signals

    X_sp, B = sparse_prior_solution(X0, A)
    X_lasso = complex_lasso(A, Y, alpha=0.1)

    threshold = noise_threshold(Y, A)
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
