import sys
from pathlib import Path
from pyparsing import alphas
import scipy.optimize as optimize
import seaborn as sns
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV

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
from cs_priors.plotting.plotting import plot_equation, plot_two_line_equation
from cs_priors.solvers.sparse_prior import (
    to_real_augmented,
    from_real_augmented,
    sparse_prior_solution,
)


def tensor_lasso_runs(
    num_mics=[8],
    num_sources=[10],
    sparsities=[2],
    alphas=[0.1],
    cross_validate=False,
    noise_level=None,
):

    results_lasso = {}
    results_sparse_prior = {}
    results_X0 = {}
    results_Xtrue = {}
    best_alphas = {}
    for n_src in num_sources:
        for n_mic in num_mics:
            for s_sparse in sparsities:
                sim = run_simulation(
                    num_sources=n_src,
                    num_mics=n_mic,
                    s_sparse=s_sparse,
                    angle_step=np.pi / 180 * 10,
                )
                freq_index = 1
                Y = sim.Y[:, freq_index]  # Measurements
                A = sim.A[:, :, freq_index]  # Mixing matrix
                X0 = np.linalg.pinv(A) @ Y  # initial guess for X

                if noise_level is not None:
                    noise_level = noise_level
                    noise = noise_level * np.random.randn(
                        *Y.shape
                    ) + 1j * noise_level * np.random.randn(
                        *Y.shape
                    )  # * unpacks the shape
                    Y = Y + noise

                # LASSO regression
                A_real = np.block([[A.real, -A.imag], [A.imag, A.real]])
                Y_real = to_real_augmented(Y)  # Flatten to 1D for sklearn

                # Cross-validate to find the best alpha
                if cross_validate:
                    print(f"min of 5 and n_mic: {min(5, n_mic)} when mic is {n_mic}")
                    folds = min(5, n_mic) if n_mic > 1 else None
                    print(f"Using {folds} folds for LassoCV with n_mic={n_mic}")
                    lasso = LassoCV(
                        alphas=alphas, cv=folds, random_state=2025, fit_intercept=False
                    )
                    best_alphas[(n_src, n_mic, s_sparse)] = lasso.alpha_
                else:
                    lasso = Lasso(
                        alpha=alphas[0],
                        fit_intercept=False,
                        max_iter=10000,
                        warm_start=True,
                    )
                lasso.coef_ = to_real_augmented(X0).flatten()  # initialize with X0
                lasso.fit(A_real, Y_real.flatten())

                X_lasso_real = lasso.coef_.reshape(-1, 1)
                X_lasso = from_real_augmented(X_lasso_real)

                # compute the sparse prior solution
                X_sparse_prior, B = sparse_prior_solution(X0, A)

                # store results
                results_lasso[(n_src, n_mic, s_sparse)] = X_lasso
                results_sparse_prior[(n_src, n_mic, s_sparse)] = X_sparse_prior
                results_X0[(n_src, n_mic, s_sparse)] = X0
                results_Xtrue[(n_src, n_mic, s_sparse)] = sim.X[:, freq_index]

    return results_lasso, results_sparse_prior, results_X0, results_Xtrue, best_alphas


def get_error_lasso_sparse_prior(
    num_sources=[10],
    num_mics=range(2, 11),
    sparsities=range(1, 11),
    noise_level=None,
):
    # handle int inputs
    if isinstance(num_mics, int):
        num_mics = [num_mics]
    if isinstance(num_sources, int):
        num_sources = [num_sources]
    if isinstance(sparsities, int):
        sparsities = [sparsities]

    # alphas = np.logspace(-7, -1, 100)
    alpha = 5e-4
    results_lasso, results_sparse_prior, results_X0, results_Xtrue, best_alphas = (
        tensor_lasso_runs(
            num_mics=num_mics,
            num_sources=num_sources,
            sparsities=sparsities,
            alphas=[alpha],
            noise_level=noise_level,
        )
    )

    # get errors
    errors_lasso = {}
    errors_sparse_prior = {}
    errors_x0 = {}
    for key in results_lasso.keys():
        X_lasso = results_lasso[key]
        X_sparse_prior = results_sparse_prior[key]
        X0 = results_X0[key]
        X_true = results_Xtrue[key]

        error_lasso = np.linalg.norm(X_true.reshape(-1, 1) - X_lasso.reshape(-1, 1))
        error_sparse_prior = np.linalg.norm(
            X_true.reshape(-1, 1) - X_sparse_prior.reshape(-1, 1)
        )
        error_initial = np.linalg.norm(X_true.reshape(-1, 1) - X0.reshape(-1, 1))

        errors_lasso[key] = error_lasso
        errors_sparse_prior[key] = error_sparse_prior
        errors_x0[key] = error_initial

    return errors_lasso, errors_sparse_prior, errors_x0, best_alphas


def compare_error_fixed_no_sources(
    errors_lasso,
    errors_sparse_prior,
    errors_x0,
    no_sources=10,
    mics_range=range(2, 11),
    sparsity_range=range(1, 11),
    show_individual_error=True,
    show_diff_error=True,
    noise_level=None,
):
    # seaborn heatmap of difference in error between lasso and sparse prior

    # Compute common vmin and vmax for equal scaling
    vmin = min(
        min(errors_lasso.values()),
        min(errors_sparse_prior.values()),
        min(errors_x0.values()),
    )
    vmax = max(
        max(errors_lasso.values()),
        max(errors_sparse_prior.values()),
        max(errors_x0.values()),
    )

    plt.figure(figsize=(6, 5))

    error_matrix_lasso = np.array(
        [
            [errors_lasso[(no_sources, n_mic, s_sparse)] for s_sparse in sparsity_range]
            for n_mic in mics_range
        ]
    )
    error_matrix_sparse_prior = np.array(
        [
            [
                errors_sparse_prior[(no_sources, n_mic, s_sparse)]
                for s_sparse in sparsity_range
            ]
            for n_mic in mics_range
        ]
    )
    error_matrix_x0 = np.array(
        [
            [errors_x0[(no_sources, n_mic, s_sparse)] for s_sparse in sparsity_range]
            for n_mic in mics_range
        ]
    )

    if show_individual_error:
        plt.subplot(1, 3, 1)
        sns.heatmap(
            error_matrix_x0,
            annot=True,
            fmt=".2f",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            xticklabels=list(sparsity_range),
            yticklabels=list(mics_range),
        )
        plt.title(f"Error for Initial X0, Noise level={noise_level}")
        plt.xlabel("Number of Active Sources")
        plt.ylabel("Number of Microphones")
        plt.subplot(1, 3, 2)
        sns.heatmap(
            error_matrix_lasso,
            annot=True,
            fmt=".2f",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            xticklabels=list(sparsity_range),
            yticklabels=list(mics_range),
        )
        plt.title(r"Error for LASSO Solution")
        plt.xlabel("Number of Active Sources")
        plt.ylabel("Number of Microphones")
        plt.subplot(1, 3, 3)
        sns.heatmap(
            error_matrix_sparse_prior,
            annot=True,
            fmt=".2f",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            xticklabels=list(sparsity_range),
            yticklabels=list(mics_range),
        )
        plt.title(r"Error for Sparse Prior Solution")
        plt.xlabel("Number of Active Sources")
        plt.ylabel("Number of Microphones")
        plt.tight_layout()
        plt.show()

    if show_diff_error:
        # compute the difference in error for X0-X_lasso and X_lasso - X_sparse prior
        error_diff_lasso_sparse_prior = error_matrix_lasso - error_matrix_sparse_prior
        error_diff_x0_lasso = error_matrix_x0 - error_matrix_lasso

        vmax = max(error_diff_lasso_sparse_prior.max(), error_diff_x0_lasso.max())
        vmin = min(error_diff_lasso_sparse_prior.min(), error_diff_x0_lasso.min())

        plt.subplot(1, 2, 1)
        sns.heatmap(
            error_diff_x0_lasso,
            annot=True,
            fmt=".2f",
            center=0,
            cmap='RdBu',
            vmin=vmin,
            vmax=vmax,
            xticklabels=list(sparsity_range),
            yticklabels=list(mics_range),
        )
        plt.title(
            f"Difference in error between X0 and LASSO, noise level={noise_level}"
        )
        plt.xlabel("Number of Active Sources")
        plt.ylabel("Number of Microphones")
        plt.subplot(1, 2, 2)
        sns.heatmap(
            error_diff_lasso_sparse_prior,
            annot=True,
            fmt=".2f",
            center=0,
            cmap='RdBu',
            vmin=vmin,
            vmax=vmax,
            xticklabels=list(sparsity_range),
            yticklabels=list(mics_range),
        )
        plt.title(r"Difference in error between LASSO and Sparse Prior")
        plt.xlabel("Number of Active Sources")
        plt.ylabel("Number of Microphones")
        plt.tight_layout()
        plt.show()


def paper_compare_lasso_sparse_prior():
    num_sources = 10
    mics_range = range(2, 11)
    sparsity_range = range(1, 10)

    errors_lasso, errors_sparse_prior, error_x0, best_alphas = (
        get_error_lasso_sparse_prior(
            num_sources=num_sources, num_mics=mics_range, sparsities=sparsity_range
        )
    )
    compare_error_fixed_no_sources(
        errors_lasso,
        errors_sparse_prior,
        error_x0,
        no_sources=num_sources,
        mics_range=mics_range,
        sparsity_range=sparsity_range,
    )


def invesitage(num_sources=10, num_mics=2, sparsity=10):
    key = (num_sources, num_mics, sparsity)
    alphas = np.logspace(-7, -1, 100)
    results_lasso, results_sparse_prior, results_X0, results_Xtrue, best_alphas = (
        tensor_lasso_runs(
            num_mics=[num_mics],
            num_sources=[num_sources],
            sparsities=[sparsity],
            alphas=alphas,
        )
    )
    plot_two_line_equation(
        to_real_augmented(results_Xtrue[key]),
        to_real_augmented(results_X0[key]),
        to_real_augmented(results_lasso[key]),
        ("X true", "X0", "X LASSO"),
        to_real_augmented(results_sparse_prior[key]),
        [best_alphas[key]],
        [0],
        (
            "X Sparse Prior",
            f"the alpha selected by cv: {best_alphas[key]}",
            "",
        ),
    )


def paper_compare_lasso_sparse_prior_noise():
    num_sources = 10
    mics_range = range(2, 11)
    sparsity_range = range(1, 10)
    noise_level = 1.0

    errors_lasso, errors_sparse_prior, error_x0, best_alphas = (
        get_error_lasso_sparse_prior(
            num_sources=num_sources,
            num_mics=mics_range,
            sparsities=sparsity_range,
            noise_level=noise_level,
        )
    )
    compare_error_fixed_no_sources(
        errors_lasso,
        errors_sparse_prior,
        error_x0,
        no_sources=num_sources,
        mics_range=mics_range,
        sparsity_range=sparsity_range,
        noise_level=noise_level,
    )


if __name__ == "__main__":
    # invesitage(num_sources=10, num_mics=2, sparsity=10)
    # paper_compare_lasso_sparse_prior()
    paper_compare_lasso_sparse_prior_noise()
