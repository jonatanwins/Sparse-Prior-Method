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
import cs_priors.plotting.plotting as plotting


def simulate_mixing_model(
    num_sources: int = 3,
    num_mics: int = 2,
    s_sparse: int | None = 1,
):
    sim = run_simulation(num_sources=num_sources, num_mics=num_mics, s_sparse=s_sparse)
    freq_index = 1
    Y = sim.Y[:, freq_index]  # Measurements
    A = sim.A[:, :, freq_index]  # Mixing matrix
    X0 = np.linalg.pinv(A) @ Y  # initial guess for X

    # for plotting
    X_true = sim.X[:, freq_index]
    # B is the span of the null space of A
    # find the null space of A
    U, S, Vt = np.linalg.svd(A)
    # compute the rank of A
    rank = np.sum(S > 1e-10)
    B = Vt[rank:].T

    # plot_paper_YAX_X0Bz(Y, A, X0, B, X_true)

    # optimize the objective
    X0_real = to_real_augmented(X0)
    B_real = np.block([[B.real, -B.imag], [B.imag, B.real]])
    D = covariance_matrices(num_sources=X0_real.shape[0])
    z_opt, x_opt, x_opt_complex, res = optimize_objective(X0_real, B_real, D)
    plot_initial_vs_optimized(z_opt, x_opt, x_opt_complex, res, X_true, X0, X0_real)


# Convert to complex vectors and matrices to augmented real form for optimization
def to_real_augmented(x_complex: np.ndarray) -> np.ndarray:
    x_real = np.array([x_complex.real, x_complex.imag]).reshape(-1, 1)
    return x_real


def from_real_augmented(x_real: np.ndarray) -> np.ndarray:
    n = x_real.shape[0] // 2
    return (x_real[:n] + 1j * x_real[n:]).reshape(-1, 1)


def covariance_matrices(num_sources: int):
    variance = 1.0
    spread = 0.005
    return [
        np.diag([variance if j == i else spread for j in range(num_sources)])
        for i in range(num_sources)
    ]


# Quadratic form x^T D x for each grid point
def quad_form(points, D_matrix):
    # This function handles both a single column vector (shape N, 1)
    # and a batch of row vectors (shape ..., N)
    if points.shape[-1] == 1 and points.ndim > 1:
        # It's a single column vector, use the mathematical formula
        return (points.T @ D_matrix @ points).item()
    else:
        # It's a batch of row vectors, use the efficient NumPy way
        return (points @ D_matrix * points).sum(axis=-1)


def first_derivative_objective(z: np.ndarray) -> np.ndarray:
    # (-1) e^{-(x_0 + Bz)^T D (x_0+Bz)}(B^T(D + D^T) (x_0 + B z))
    # where D_i is symmetric
    x = X0_real.reshape(-1, 1) + (B_real @ z.reshape(-1, 1))
    grad = np.zeros_like(z)
    for D_i in D:
        qf = quad_form(x, D_i)
        exp_neg_qf = np.exp(-qf)
        grad += 2 * exp_neg_qf * (B_real.T @ (D_i @ x)).flatten()
    return grad  # this is the negative, not actual


def second_derivative_objective(z: np.ndarray) -> np.ndarray:
    # sum_i e^{-(x_0 + Bz)^T D_i (x_0+Bz)}
    #  * [ (B^T (D_i + D_i^T) (x_0 + B z)) (B^T (D_i + D_i^T) (x_0 + B z))^T
    #      - B^T (D_i + D_i^T) B ]
    x = X0_real.reshape(-1, 1) + (B_real @ z.reshape(-1, 1))
    hess = np.zeros((z.shape[0], z.shape[0]))
    for D_i in D:
        qf = quad_form(x, D_i)
        exp_neg_qf = np.exp(-qf)
        BdX = B_real.T @ (D_i @ x)
        hess += exp_neg_qf * (2 * np.outer(BdX, BdX) - 2 * (B_real.T @ D_i @ B_real))
    return -hess  # this is the negatice Hessian, not the actual


def optimize_objective(X0_real, B_real, D, callback=None, z_start=None):
    # factory function for the objective to minimize
    def negative_objective(z: np.ndarray) -> float:
        # numpy broadcasting will make row vectors if z is 1D
        # X0_real has shape (N,) whereas B_real @ z has shape (N, 1)
        x = X0_real.reshape(-1, 1) + (B_real @ z.reshape(-1, 1)).reshape(-1, 1)
        return -sum(np.exp(-quad_form(x, D_i)) for D_i in D)

    if z_start is None:
        z_start = np.zeros(B_real.shape[1])

    res = optimize.minimize(
        negative_objective,
        z_start,
        method='l-BFGS-B',
        callback=callback,
        # jac=first_derivative_objective,
        # hess=second_derivative_objective,
    )
    # print(res.message)
    z_opt = res.x
    x_opt = X0_real.reshape(-1, 1) + (B_real @ z_opt.reshape(-1, 1))
    x_opt_complex = from_real_augmented(x_opt)
    return z_opt, x_opt, x_opt_complex, res


def sparse_prior_solution(X0, A) -> tuple[np.ndarray, np.ndarray]:
    """
    Args:
        X0: complex initial solution
        A: complex mixing matrix
    """
    U, S, Vt = np.linalg.svd(A)
    rank = np.sum(S > 1e-10)

    # Check if there is a null space
    if rank == A.shape[1]:
        # No null space, return the pseudoinverse solution
        return X0, None

    # Compute the basis for the null space
    B = Vt[rank:].T
    B_real = np.block([[B.real, -B.imag], [B.imag, B.real]])
    X0_real = to_real_augmented(X0)
    D = covariance_matrices(num_sources=X0_real.shape[0])
    z_opt, x_opt, x_opt_complex, res = optimize_objective(X0_real, B_real, D)
    return x_opt_complex, B


# Test for what distance the pseudoinverse fails to distinguish between sources
def test_source_separation():
    num_sources = 10
    num_mics = 10
    s_sparse = 5

    angle_steps = np.logspace(-15, -0, num=25)

    # Only for plotting the overview

    # sim = run_simulation(
    #     num_sources=num_sources,
    #     num_mics=num_mics,
    #     s_sparse=s_sparse,
    #     angle_step=0.1,
    # )
    # plotting.wrapper_plot_geometry(sim, pad_factor=10.7)

    separations_X0 = []
    separations_Xopt = []

    for angle_step in angle_steps:
        sim = run_simulation(
            num_sources=num_sources,
            num_mics=num_mics,
            s_sparse=s_sparse,
            angle_step=angle_step,
            # phase_step=3,
            # amplitude_step=0,
        )

        freq_index = 1
        Y = sim.Y[:, freq_index]  # Measurements
        A = sim.A[:, :, freq_index]  # Mixing matrix
        X0 = np.linalg.pinv(A) @ Y  # initial guess for X
        global X_true
        X_true = sim.X[:, freq_index]

        # Calculate distances
        dist = np.linalg.norm(X0.reshape(-1, 1) - X_true.reshape(-1, 1))
        separations_X0.append(dist)

        x_opt_complex, B = sparse_prior_solution(X0, A)
        dist_opt = np.linalg.norm(x_opt_complex.reshape(-1, 1) - X_true.reshape(-1, 1))
        # plotting.plot_two_line_equation(
        #     X_true,
        #     X_true - X0,
        #     X_true.reshape(-1, 1) - x_opt_complex,
        #     (
        #         r"$X_{true}$",
        #         rf"$X_true - X_0$ {dist:.2f}",
        #         f"$X_true - X_opt$ {dist_opt:.2f}",
        #     ),
        #     to_real_augmented(X_true),
        #     to_real_augmented(X0),
        #     to_real_augmented(x_opt_complex),
        #     (r"$X_{true}^{real}$", r"$X_0^{real}$", r"$X_{opt}^{real}$"),
        #     font_size=2,
        # )
        separations_Xopt.append(dist_opt)

        # temporary for plotting
        X0_real = to_real_augmented(X0)
        D = covariance_matrices(num_sources=X0_real.shape[0])

        if angle_step == 1e-9:
            plot_YAX_XoptX0B(Y, A, X_true, x_opt_complex, X0, B)

    plt.figure()
    plt.plot(angle_steps, separations_X0, marker='o')
    plt.plot(angle_steps, separations_Xopt, marker='x')
    plt.xlabel("Angle Step (radians)")
    # logartithmic scale for x-axis
    plt.xscale("log")
    plt.ylabel(r"Error $\|X_0 - X_{true}\|$")
    plt.title(
        r"Source Separation vs $\Delta \theta$"
        + f" ({num_sources} sources, {num_mics} mics, {s_sparse}-sparse)"
    )
    plt.grid()
    plt.legend([r"$X_0$", r"$X_{opt}$"])
    plt.show()


def compare_to_LASSO():

    num_sources = 10
    num_mics = 8
    s_sparse = 2
    angle_step = np.pi / 180 * 10
    sim = run_simulation(
        num_sources=num_sources,
        num_mics=num_mics,
        s_sparse=s_sparse,
        angle_step=angle_step,
        # phase_step=0.3,
    )
    # plotting.wrapper_plot_geometry(sim, pad_factor=10.7)

    freq_index = 1
    Y = sim.Y[:, freq_index]  # Measurements
    A = sim.A[:, :, freq_index]  # Mixing matrix
    X0 = np.linalg.pinv(A) @ Y  # initial guess for X
    X_true = sim.X[:, freq_index]

    # LASSO regression
    A_real = np.block([[A.real, -A.imag], [A.imag, A.real]])
    Y_real = to_real_augmented(Y)
    lasso = Lasso(alpha=0.1)
    lasso.fit(A_real, Y_real)  # coordinate descent
    X_lasso_real = lasso.coef_
    X_lasso = from_real_augmented(X_lasso_real)

    dist_lasso = np.linalg.norm(X_lasso - X_true)
    dist_pinv = np.linalg.norm(X0 - X_true)

    print(f"Pseudoinverse distance: {dist_pinv:.4f}")
    print(f"LASSO distance: {dist_lasso:.4f}")

    # sparse prior optimization
    X0_real = to_real_augmented(X0)
    U, S, Vt = np.linalg.svd(A)
    rank = np.sum(S > 1e-10)
    B = Vt[rank:].T

    B_real = np.block([[B.real, -B.imag], [B.imag, B.real]])
    D = covariance_matrices(num_sources=X0_real.shape[0])
    z_opt, x_opt, x_opt_complex, res = optimize_objective(X0_real, B_real, D)

    plotting.plot_two_line_equation(
        X_true,
        x_opt_complex,
        X_lasso,
        (r"$X_{True}$", r"$X_{opt}$", r"$X_{LASSO}$"),
        X0,
        x_opt,
        X_lasso_real,
        (r"$X_{0}$", r"$X_{opt}^{real}$", r"$X_{LASSO}^{real}$"),
        font_size=16,
    )


def compare_real_value_LASSO():
    from sklearn.linear_model import Lasso

    # Let X be a real vector of size 10, Let A be a real mixing matrix of size 8x10, Let Y be the measurements
    num_sources = 10
    mic_numbers = list(range(1, 20))
    # mic_numbers = [5, 7]
    s_sparse = 2

    # set the random seed for reproducibility
    np.random.seed(2025)
    X_true = np.random.randn(num_sources) * 10
    # choose sparse indicies
    zero_indices = np.random.choice(num_sources, num_sources - s_sparse, replace=False)
    X_true[zero_indices] = 0  # make X sparse

    dist_lasso_list = []
    dist_opt_list = []

    for num_mics in mic_numbers:
        A = np.random.randn(num_mics, num_sources)
        Y = A @ X_true  # Measurements

        # LASSO regression
        lasso = Lasso(alpha=0.1)
        lasso.fit(A, Y)  # coordinate descent
        X_lasso = lasso.coef_
        dist_lasso = np.linalg.norm(X_lasso.reshape(-1, 1) - X_true.reshape(-1, 1))
        print(f"LASSO distance: {dist_lasso:.4f}")
        dist_lasso_list.append(dist_lasso)

        # sparse prior optimization
        X0 = np.linalg.pinv(A) @ Y  # initial guess for X
        x_opt, B = sparse_prior_solution(X0, A)
        dist_opt = np.linalg.norm(x_opt.reshape(-1, 1) - X_true.reshape(-1, 1))
        print(f"Sparse prior optimization distance: {dist_opt:.4f}")
        dist_opt_list.append(dist_opt)

    plt.figure()
    plt.plot(mic_numbers, dist_lasso_list, marker='o')
    plt.plot(mic_numbers, dist_opt_list, marker='x')
    plt.xlabel("Number of Microphones")
    plt.ylabel(r"Error $\|X_{est} - X_{true}\|$")
    plt.title(
        r"Source Reconstruction Error vs Number of Microphones"
        + f" ({num_sources} sources, {s_sparse}-sparse)"
    )
    plt.grid()
    plt.legend([r"$X_{LASSO}$", r"$X_{opt}$"])
    plt.show()

    plotting.plot_two_line_equation(
        Y,
        A,
        X_true,
        (r"$Y$", r"$A$", r"$X_{true}$"),
        X0,
        x_opt,
        X_lasso,
        (r"$X_{0}$", r"$X_{opt}$", r"$X_{LASSO}$"),
        font_size=16,
    )


# plot a heatmap of LASSO and sparse prior optimization error over num_mics and s_sparse
# put the exact number in each
def plot_error_heatmap_LASSO():
    import seaborn as sns
    from sklearn.linear_model import Lasso

    num_sources = 10
    mic_numbers = list(range(1, 11))
    s_sparse_numbers = list(range(0, 11))

    error_lasso = np.zeros((len(mic_numbers), len(s_sparse_numbers)))
    error_opt = np.zeros((len(mic_numbers), len(s_sparse_numbers)))

    # set the random seed for reproducibility
    np.random.seed(2025)
    X_true = np.random.randn(num_sources) * 10

    for i, num_mics in enumerate(mic_numbers):
        for j, s_sparse in enumerate(s_sparse_numbers):
            A = np.random.randn(num_mics, num_sources)
            Y = A @ X_true  # Measurements

            # LASSO regression
            lasso = Lasso(alpha=0.1)
            lasso.fit(A, Y)  # coordinate descent
            X_lasso = lasso.coef_
            dist_lasso = np.linalg.norm(X_lasso.reshape(-1, 1) - X_true.reshape(-1, 1))
            error_lasso[i, j] = dist_lasso

            # sparse prior optimization
            X0 = np.linalg.pinv(A) @ Y  # initial guess for X
            x_opt, B = sparse_prior_solution(X0, A)
            dist_opt = np.linalg.norm(x_opt.reshape(-1, 1) - X_true.reshape(-1, 1))
            error_opt[i, j] = dist_opt

    # Compute common vmin and vmax for equal scaling
    vmin = min(error_lasso.min(), error_opt.min())
    vmax = max(error_lasso.max(), error_opt.max())

    # Reverse the data arrays to invert the y-axis (mic_numbers order)
    error_lasso_reversed = error_lasso[::-1]
    error_opt_reversed = error_opt[::-1]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.heatmap(
        error_lasso_reversed,
        annot=True,
        fmt='.1f',
        xticklabels=s_sparse_numbers,
        yticklabels=mic_numbers[::-1],
        vmin=vmin,
        vmax=vmax,
        cbar_kws={'label': 'Reconstruction Error'},
    )
    plt.xlabel('Sparsity Level (s_sparse)')
    plt.ylabel('Number of Microphones')
    plt.title('LASSO Reconstruction Error')

    plt.subplot(1, 2, 2)
    sns.heatmap(
        error_opt_reversed,
        annot=True,
        fmt='.1f',
        xticklabels=s_sparse_numbers,
        yticklabels=mic_numbers[::-1],
        vmin=vmin,
        vmax=vmax,
        cbar_kws={'label': 'Reconstruction Error'},
    )
    plt.xlabel('Sparsity Level (s_sparse)')
    plt.ylabel('Number of Microphones')
    plt.title('Sparse Prior Optimization Reconstruction Error')
    plt.show()
    plt.figure()
    sns.heatmap(
        error_lasso_reversed - error_opt_reversed,
        annot=True,
        fmt='.2f',
        xticklabels=s_sparse_numbers,
        yticklabels=mic_numbers[::-1],
        center=0,
        cmap='RdBu',
        cbar_kws={'label': 'Error Difference (LASSO - Sparse Prior)'},
    )
    plt.xlabel('Sparsity Level (s_sparse)')
    plt.ylabel('Number of Microphones')
    plt.show()


# same as plot_heatmap_LASSO but only the error difference and for simulated data
def plot_error_difference_heatmap_LASSO_simulated():

    num_sources = 10
    mic_numbers = list(range(1, 11))
    s_sparse_numbers = list(range(0, 11))

    error_lasso = np.zeros((len(mic_numbers), len(s_sparse_numbers)))
    error_opt = np.zeros((len(mic_numbers), len(s_sparse_numbers)))

    for i, num_mics in enumerate(mic_numbers):
        for s, s_sparse in enumerate(s_sparse_numbers):
            sim = run_simulation(
                num_sources=num_sources,
                num_mics=num_mics,
                s_sparse=s_sparse,
                angle_step=np.pi / 180 * 10,
            )
            freq_index = 1
            Y = sim.Y[:, freq_index]  # Measurements
            A = sim.A[:, :, freq_index]  # Mixing matrix
            X_true = sim.X[:, freq_index]

            # LASSO regression
            A_real = np.block([[A.real, -A.imag], [A.imag, A.real]])
            Y_real = to_real_augmented(Y)  # Flatten to 1D for sklearn

            # initialize LASSO model to X0
            X0_real = to_real_augmented(np.linalg.pinv(A) @ Y)

            lasso = Lasso(alpha=0.0, warm_start=True, fit_intercept=False)
            lasso.coef_ = X0_real.flatten()  # initialize with X0
            lasso.intercept_ = 0.0
            lasso.fit(
                A_real, Y_real
            )  # coordinate descent with warm start so it starts from X0

            X_lasso_real = lasso.coef_.reshape(-1, 1)
            X_lasso = from_real_augmented(X_lasso_real)
            dist_lasso = np.linalg.norm(X_lasso.reshape(-1, 1) - X_true.reshape(-1, 1))
            error_lasso[i, s] = dist_lasso

            if s == 2 and i == 8:
                n = X_lasso_real.shape[0] // 2
                plotting.plot_two_line_equation(
                    X_lasso,
                    X_lasso_real[:n],
                    X_lasso_real[n:],
                    (
                        r"$X_{LASSO} =$",
                        r"$X_{LASSO}^{real}$",
                        r"$+i \ X_{LASSO}^{imag}$",
                    ),
                    to_real_augmented(X_true),
                    X0_real,
                    X_lasso_real,
                    (r"$X_{true}^{real}$", r"$X_{0}^{real}$", r"$X_{LASSO}^{real}$"),
                )

                X_true_real = to_real_augmented(X_true)
                print(
                    f"High LASSO error: {dist_lasso:.2f} for num_mics={num_mics}, s_sparse={s_sparse}"
                )
                plotting.plot_two_line_equation(
                    Y_real,
                    A_real @ X_lasso_real,
                    Y_real - (A_real @ X_lasso_real).reshape(-1, 1),
                    (
                        r"$\|Y^{real}$",
                        r"$-A^{real} X_{LASSO}^{real}\|_2$",
                        r"$=Residual$",
                    ),
                    X_lasso_real,
                    X_true_real,
                    X0_real,
                    (
                        r"$0.1\|X_{LASSO}^{real}\|_1$",
                        r"$X_{true}^{real}$",
                        r"$X_{0}^{real}$",
                    ),
                    font_size=16,
                )

            # sparse prior optimization
            X0 = np.linalg.pinv(A) @ Y  # initial guess for X
            x_opt, B = sparse_prior_solution(X0, A)
            dist_opt = np.linalg.norm(x_opt.reshape(-1, 1) - X_true.reshape(-1, 1))
            error_opt[i, s] = dist_opt

            # i mics and s sparse
            if (i == 8) and (s == 2):
                plotting.plot_two_line_equation(
                    X_true,
                    x_opt,
                    X_lasso,
                    (r"$X_{True}$", r"$X_{opt}$", r"$X_{LASSO}$"),
                    X0,
                    x_opt,
                    X_lasso_real,
                    (r"$X_{0}$", r"$X_{opt}^{real}$", r"$X_{LASSO}^{real}$"),
                    font_size=16,
                )

    # Compute common vmin and vmax for equal scaling
    vmin = min(error_lasso.min(), error_opt.min())
    vmax = max(error_lasso.max(), error_opt.max())

    # Reverse the data arrays to invert the y-axis (mic_numbers order)
    error_difference_reversed = (error_lasso - error_opt)[::-1]

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        error_difference_reversed,
        annot=True,
        fmt='.2f',
        xticklabels=s_sparse_numbers,
        yticklabels=mic_numbers[::-1],
        center=0,
        cmap='RdBu',
        cbar_kws={'label': 'Error Difference (LASSO - Sparse Prior)'},
    )
    plt.xlabel('Sparsity Level (s_sparse)')
    plt.ylabel('Number of Microphones')
    plt.title(
        'Reconstruction Error Difference (LASSO - Sparse Prior) on simulated Data'
    )
    plt.show()
    # also plot the two errors side by side for comparison
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.heatmap(
        error_lasso[::-1],
        annot=True,
        fmt='.1f',
        xticklabels=s_sparse_numbers,
        yticklabels=mic_numbers[::-1],
        vmin=vmin,
        vmax=vmax,
        cbar_kws={'label': 'Reconstruction Error'},
    )
    plt.xlabel('Sparsity Level (s_sparse)')
    plt.ylabel('Number of Microphones')
    plt.title('LASSO Reconstruction Error')
    plt.subplot(1, 2, 2)
    sns.heatmap(
        error_opt[::-1],
        annot=True,
        fmt='.1f',
        xticklabels=s_sparse_numbers,
        yticklabels=mic_numbers[::-1],
        vmin=vmin,
        vmax=vmax,
        cbar_kws={'label': 'Reconstruction Error'},
    )
    plt.xlabel('Sparsity Level (s_sparse)')
    plt.ylabel('Number of Microphones')
    plt.title('Sparse Prior Optimization Reconstruction Error')
    plt.show()


# Plotting
def examine_null_space(X_true, X0, A, B_real, U, S, Vt):
    plotting.plot_two_line_equation(
        X_true - X0,
        B_real,
        A,
        (r"$X_{true}-X_{0}$", r"$B^{real}$", r"$A$"),
        U,
        S,
        Vt,
        (r"$U$", r"$S$", r"$V^T$"),
        font_size=16,
    )


def plot_objective_topology(ax, X0, B, D):
    """
    Plots the objective function as a topological surface over the z1-z2 plane.
    """

    scale = 15
    num_points = 10
    # 1. Create the z1, z2 grid
    z1_range = np.linspace(-scale, scale, num_points)
    z2_range = np.linspace(-scale, scale, num_points)
    Z1, Z2 = np.meshgrid(z1_range, z2_range)
    z_grid = np.stack([Z1.ravel(), Z2.ravel()], axis=0)

    # Decide what type of visualization based on B's shape
    if B is None:
        print("B is None, cannot plot topology.")
        return
    if B.shape[1] == 0:
        print(f"B {B.shape} has 0 dimensions, cannot plot topology.")
        return
    elif B.shape[1] == 1:
        print(
            f"B {B.shape} has only 1 dimension, plotting a line instead of a surface."
        )
    elif B.shape[1] == 2:
        print(f"B {B.shape} has 2 dimensions, plotting a surface.")
        X_plane_rows = (X0.reshape(-1, 1) + B @ z_grid).T
    elif B.shape[1] > 2:
        # note that X0.shape[0] is B.shape[0] no matter what B.shape[1] is, likwise symmetric D is not changed
        print(
            f"B {B.shape} has more than 2 dimensions, projecting onto first two dimensions for plotting."
        )
        B_reduced = B[:, :2]

        X_plane_rows = (X0.reshape(-1, 1) + B_reduced @ z_grid).T
        # D = [D_i[: B_reduced.shape[0], : B_reduced.shape[0]] for D_i in D]
    else:
        print(f"Unexpected shape of B.{B.shape}, cannot plot topology.")
        return

    # 2. Calculate the objective function P for all points on the z1-z2 grid
    # X_plane_rows = (X0.reshape(-1, 1) + B @ z_grid).T
    P_values = sum(np.exp(-quad_form(X_plane_rows, D_i)) for D_i in D)
    P_surf = P_values.reshape(Z1.shape)

    # 3. Plot the topological surface
    ax.plot_surface(Z1, Z2, P_surf, cmap='viridis', alpha=0.8)

    # 4. Run optimizer and plot its path
    path = []

    def callback(xk):
        path.append(np.copy(xk))

    # Run from a non-symmetric starting point to see the path
    z_start = np.ones(B.shape[1]) * 10 * np.random.rand()

    z_opt, x_opt, x_opt_complex, res = optimize_objective(
        X0, B, D, callback=callback, z_start=z_start
    )
    path = np.array(path)

    # Calculate P values along the path
    path_x = (X0.reshape(-1, 1) + B @ path.T).T
    path_p = sum(np.exp(-quad_form(path_x, D_i)) for D_i in D)

    # Plot the path on the surface
    ax.plot(
        path[:, 0],
        path[:, 1],
        path_p,
        color='red',
        marker='.',
        markersize=4,
        label='Optimizer Path',
    )
    ax.scatter(
        path[0, 0],
        path[0, 1],
        path_p[0],
        color='orange',
        s=100,
        label='Start',
        zorder=10,
    )
    ax.scatter(
        path[-1, 0],
        path[-1, 1],
        path_p[-1],
        color='magenta',
        s=150,
        marker='*',
        label='Optimum',
        zorder=10,
    )

    # 5. Set labels and title
    ax.set_xlabel("z1 (Direction of 1st Null Space Vector)")
    ax.set_ylabel("z2 (Direction of 2nd Null Space Vector)")
    ax.set_zlabel("Objective Function (P)")
    ax.set_title("Topological Map of Objective Function on Solution Plane")
    ax.legend()


def plot_initial_vs_optimized(z_opt, x_opt, x_opt_complex, res, X_true, X0, X0_real):
    print("Optimal z:", z_opt, "Objective value:", -res.fun)
    print("Optimal x (real):", x_opt)
    print("Optimal x (complex):", x_opt_complex)
    plotting.plot_two_line_equation(
        X_true,
        X0,
        x_opt_complex,
        (r"$X_{true}$", r"$X_0$", r"$X_{opt}$"),
        to_real_augmented(X_true),
        X0_real,
        x_opt,
        # (f"{num_sources} sources", f"{num_mics} mics", f"{s_sparse}-sparse"),
        (r"$X_{true}^{real}$", r"$X_0^{real}$", r"$X_{opt}^{real}$"),
        font_size=8,
    )


def plot_YAX_XoptX0B(Y, A, X_true, Xopt, X0, B):
    plotting.plot_two_line_equation(
        Y,
        A,
        X_true,
        (r"$Y =$", r"$A$", r"$X_{true}$"),
        Xopt,
        X0,
        B,
        (r"$X_{opt} =$", r"$X_0$", r"$+ Bz$"),
        font_size=16,
    )


def tensor_lasso_runs(num_mics=[8], num_sources=[10], sparsities=[2], alphas=[0.1]):

    results = {}
    for n_src in num_sources:
        for n_mic in num_mics:
            for s_sparse in sparsities:
                for alpha in alphas:
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

                    # LASSO regression
                    A_real = np.block([[A.real, -A.imag], [A.imag, A.real]])
                    Y_real = to_real_augmented(Y)  # Flatten to 1D for sklearn

                    lasso = Lasso(alpha=alpha, warm_start=True, fit_intercept=False)
                    lasso.coef_ = to_real_augmented(X0).flatten()  # initialize with X0

                    lasso.fit(A_real, Y_real)  # coordinate descent

                    X_lasso_real = lasso.coef_.reshape(-1, 1)
                    X_lasso = from_real_augmented(X_lasso_real)

                    results[(n_src, n_mic, s_sparse, alpha)] = X_lasso

    return results


def tensor_sparse_prior_runs(num_mics=[8], sparsities=[2], num_sources=[10]):
    results_X0 = {}
    results = {}
    results_X_true = {}
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
                X_true = sim.X[:, freq_index]

                # sparse prior optimization
                X0 = np.linalg.pinv(A) @ Y  # initial guess for X
                x_opt, B = sparse_prior_solution(X0, A)

                results[(n_src, n_mic, s_sparse)] = x_opt
                results_X0[(n_src, n_mic, s_sparse)] = X0
                results_X_true[(n_src, n_mic, s_sparse)] = X_true
    return results, results_X0, results_X_true


def compare_lasso_sparse_prior_vary_alpha():
    num_mics = [8]
    num_sources = [10]
    alphas = [0.1, 0.05, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0]
    sparsities = [2, 3, 4, 5, 6, 7, 8]

    results_lasso = tensor_lasso_runs(
        num_mics=num_mics,
        num_sources=num_sources,
        sparsities=sparsities,
        alphas=alphas,
    )
    results_sparse_prior, results_X0, results_X_true = tensor_sparse_prior_runs(
        num_mics=num_mics,
        sparsities=sparsities,
        num_sources=num_sources,
    )

    # plot thee true, x0, sparse prior and lasso for alpha 0.0, 0.0001, 0.1 for 8 mics, 10 sources, sparsity 4
    n_src = 10
    n_mic = 8
    s_sparse = 2

    X_true = results_X_true[(n_src, n_mic, s_sparse)]
    X0 = results_X0[(n_src, n_mic, s_sparse)]
    X_sparse_prior = results_sparse_prior[(n_src, n_mic, s_sparse)]
    X_lasso_0 = results_lasso[(n_src, n_mic, s_sparse, 0.0)]
    X_lasso_01 = results_lasso[(n_src, n_mic, s_sparse, 0.1)]
    X_lasso_0001 = results_lasso[(n_src, n_mic, s_sparse, 0.001)]
    plotting.plot_two_line_equation(
        to_real_augmented(X_true),
        to_real_augmented(X0),
        to_real_augmented(X_sparse_prior),
        (r"$X_{true}$", r"$X_0$", r"$X_{sparse\ prior}$"),
        to_real_augmented(X_lasso_0),
        to_real_augmented(X_lasso_01),
        to_real_augmented(X_lasso_0001),
        (
            r"$X_{LASSO\ \alpha=0.0}$",
            r"$X_{LASSO\ \alpha=0.1}$",
            r"$X_{LASSO\ \alpha=0.001}$",
        ),
        font_size=8,
    )

    # plot 2d plot of sparse prior and all alphas for 8 mics, 10 sources, varying the sparsity
    sparse_prior_errors = []
    X0_errors = []
    lasso_errors = {alpha: [] for alpha in alphas}
    for s_sparse in sparsities:
        error = np.linalg.norm(
            results_sparse_prior[(10, 8, s_sparse)].reshape(-1, 1)
            - results_X_true[(10, 8, s_sparse)].reshape(-1, 1)
        )
        sparse_prior_errors.append(error)
        error = np.linalg.norm(
            results_X0[(10, 8, s_sparse)].reshape(-1, 1)
            - results_X_true[(10, 8, s_sparse)].reshape(-1, 1)
        )
        X0_errors.append(error)
        for alpha in alphas:
            error = np.linalg.norm(
                results_lasso[(10, 8, s_sparse, alpha)].reshape(-1, 1)
                - results_X_true[(10, 8, s_sparse)].reshape(-1, 1)
            )
            lasso_errors[alpha].append(error)
    plt.figure()
    plt.plot(sparsities, sparse_prior_errors, marker='o', label='Sparse Prior')
    plt.plot(sparsities, X0_errors, marker='s', label='Pseudoinverse X0')
    for alpha in alphas:
        plt.plot(
            sparsities,
            lasso_errors[alpha],
            marker='x',
            label=f'LASSO alpha={alpha}',
        )
    plt.xlabel('Sparsity Level (s_sparse)')
    plt.ylabel(r'Error $\|X_{est} - X_{true}\|$')
    plt.title(
        r'Source Reconstruction Error vs Sparsity Level' + f' (10 sources, 8 mics)'
    )
    plt.grid()
    plt.legend()
    plt.show()


def group_lasso():
    """
    Groups the real and imaginary parts of each source together in the LASSO regression.
    The real form is of [Re Im].T Hence the groups are (X[i], X[i+num_sources]) for i in range(num_sources)
    """
    from group_lasso import GroupLasso

    num_sources = 10
    num_mics = 8
    s_sparse = 2
    angle_step = np.pi / 180 * 10
    sim = run_simulation(
        num_sources=num_sources,
        num_mics=num_mics,
        s_sparse=s_sparse,
        angle_step=angle_step,
    )
    freq_index = 1
    Y = sim.Y[:, freq_index]  # Measurements
    A = sim.A[:, :, freq_index]  # Mixing matrix
    X0 = np.linalg.pinv(A) @ Y  # initial guess for X
    X_true = sim.X[:, freq_index]

    # Group LASSO regression
    A_real = np.block([[A.real, -A.imag], [A.imag, A.real]])
    Y_real = to_real_augmented(Y)

    # Define groups for real and imaginary parts
    groups = []
    for i in range(num_sources):
        groups.append([i, i + num_sources])  # real part and imaginary part

    group_lasso = GroupLasso(
        groups=groups,
        group_reg=0.1,
        l1_reg=0.0,
        frobenius_lipschitz=True,
        scale_reg="group_size",
        subsampling_scheme=None,
        fit_intercept=False,
        # max_iter=1000,
        tol=1e-3,
        warm_start=True,
        random_state=2025,
    )
    group_lasso.fit(A_real, Y_real)  # coordinate descent
    X_glasso_real = group_lasso.coef_.reshape(-1, 1)
    X_glasso = from_real_augmented(X_glasso_real)
    dist_glasso = np.linalg.norm(X_glasso - X_true)
    dist_pinv = np.linalg.norm(X0 - X_true)
    print(f"Pseudoinverse distance: {dist_pinv:.4f}")
    print(f"Group LASSO distance: {dist_glasso:.4f}")
    plotting.plot_two_line_equation(
        X_true,
        X0,
        X_glasso,
        (r"$X_{True}$", r"$X_{0}$", r"$X_{Group\ LASSO}$"),
        to_real_augmented(X_true),
        to_real_augmented(X0),
        X_glasso_real,
        (r"$X_{true}^{real}$", r"$X_{0}^{real}$", r"$X_{Group\ LASSO}^{real}$"),
        font_size=16,
    )


if __name__ == "__main__":
    # simulate_mixing_model()
    # test_source_separation()
    # compare_real_value_LASSO()
    # plot_error_heatmap_LASSO()
    # plot_error_difference_heatmap_LASSO_simulated()
    # compare_lasso_sparse_prior_vary_alpha()
    group_lasso()
