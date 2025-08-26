import numpy as np
from sklearn.linear_model import Lasso, MultiTaskLasso
from src.main import run_simulation
from plotting import plot_geometry_auto, plot_overview, plot_equation, plot_matrix_3D
from numpy.linalg import pinv


def pred_lasso(sim, alpha=1, plot=False):
    X_lasso = np.zeros_like(sim.X)
    lasso = Lasso(alpha=alpha, fit_intercept=False)
    Y_block = np.vstack([sim.Y.real, sim.Y.imag])
    X_block = np.vstack([X_lasso, X_lasso])
    M = len(sim.X)

    for f in range(len(sim.freqs)):
        C_block_f = np.block(
            [
                [sim.C[:, :, f].real, -sim.C[:, :, f].imag],
                [sim.C[:, :, f].imag, sim.C[:, :, f].real],
            ]
        )
        lasso.fit(
            C_block_f, Y_block[:, f]
        )  # TODO Here coordinate descent is being used.
        X_block_f = lasso.coef_.T  # shape (2M × K)
        X_block[:, f] = X_block_f[:]
        X_lasso[:, f] = X_block_f[:M] + 1j * X_block_f[M:]

    if plot:
        plot_equation(
            Y_block[:, 1],
            C_block_f,
            X_block_f,
            titles=(f'Y_block[:,{1}]', f'C_block_f={1}', f'X_block_f={1}, {M=}'),
            ratios=(1, 10, 1),
        )
        plot_equation(
            Y_block, X_block, [0], ('Y_block', 'X_block', ''), ratios=(1, 1, 0)
        )

        plot_equation(
            sim.X_pred,
            sim.X,
            X_lasso,
            titles=("X_pred from psuedoinverse", "X", "X_lasso"),
            ratios=(1, 1, 1),
        )

    return X_lasso


def diy_omp(A, y, max_iterations=100, tol=1e-3):
    m, N = A.shape
    x = np.zeros((N, 1), dtype=complex)
    y = y.reshape(-1, 1)
    residual = y.copy()
    support = []

    for k in range(max_iterations):
        # OMP1
        correlations = A.T @ residual
        correlations[support] = 0  # so they’ll never be max again
        j = np.argmax(np.abs(correlations))  # index that contributes the most
        support.append(j)
        # OMP2 recompute x entirely
        z = np.linalg.pinv(A[:, support]) @ y  # the updated projection, no zeros.
        x[support, 0] = z.flatten()

        residual = y - A @ x
        if np.linalg.norm(residual) < tol:
            break
        else:
            print(f"{np.linalg.norm(residual)=}")

    return x, support


def multi_omp(sim):  # Can be modified to multi_function for any method
    X_omp = np.zeros((len(sim.sources), sim.N))
    for f in range(sim.N):  # for each frequency, N = len(freqs)
        X_omp[:, f] = diy_omp(sim.C[:, :, f], sim.Y[:, f])

    return X_omp


def s_largest_entries(z, s):
    mod_z = z.copy()
    indicies = np.zeros(s, dtype=int)
    for i in range(s):
        indicies[i] = np.argmax(abs(mod_z))
        mod_z[indicies[i]] = 0
    return indicies


def s_approximation(z, s):
    sparse_approx = np.zeros_like(z)
    indicies = s_largest_entries(z, s)
    sparse_approx[indicies] = z[indicies]
    return sparse_approx


def support(x):
    return np.nonzero(x)[0]


def CoSaMP(A, y, s, iterations=100):
    m, N = A.shape
    x_CoSaMP = np.zeros(N)
    for i in range(iterations):
        proxy = A.conj().T @ (y - A @ x_CoSaMP)  # conjugate transpose
        L_2s = s_largest_entries(proxy, 2 * s)
        support_x = support(x_CoSaMP)
        U = np.union1d(support_x, L_2s)
        u = np.zeros_like(x_CoSaMP)
        z = pinv(A[:, U]) @ y  # the updated projection, no zeros.
        u[U] = z  # equivalent to argmin ||y-Ax||_2 with support on U
        x_CoSaMP = s_approximation(u, s)
    return x_CoSaMP


def test_sparse_recovery(method, N=8, s=3, m=8, sigma=0.01, seed=None, **method_kwargs):

    if m is None:
        m = 4 * s
    if seed is not None:
        np.random.seed(seed)

    # Generate random measurement matrix and sparse signal
    A = np.random.randn(m, N) + 1j * np.random.randn(m, N)
    x_true = np.zeros(N, dtype=complex)
    true_support = np.random.choice(N, s, replace=False)
    x_true[true_support] = np.random.randn(s) + 1j * np.random.randn(s)

    # Generate noisy measurements
    noise = sigma * (np.random.randn(m) + 1j * np.random.randn(m))
    y = A @ x_true + noise

    # Recover signal
    x_rec = method(A, y, s, **method_kwargs)

    # Metrics
    rel_error = np.linalg.norm(x_rec - x_true) / np.linalg.norm(x_true)
    recovered_support = np.nonzero(x_rec)[0]

    print(f"True support:      {sorted(true_support)}")
    print(f"Recovered support: {sorted(recovered_support)}")
    print(f"Relative error:    {rel_error:.2e}")
    plot_equation(x_rec, x_true, A, ("x_rec", "x_true", "A"), ratios=(1, 1, 10))
    plot_equation(
        A @ x_true,
        y,
        x_rec - x_true,
        titles=("No noise", "With noise", "x_rec - x_true"),
        ratios=(1, 1, 1),
    )

    return rel_error, true_support, recovered_support


def HTP(): ...


if __name__ == "__main__":
    # sim = run_simulation(num_mics=3, no_sources=8, s_sparse=3)
    # X_lasso = pred_lasso(sim)

    test_sparse_recovery(CoSaMP)
