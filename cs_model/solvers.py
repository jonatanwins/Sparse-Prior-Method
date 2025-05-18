import numpy as np
from sklearn.linear_model import Lasso, MultiTaskLasso
from app import run_simulation
from plotting import plot_geometry_auto, plot_overview, plot_equation, plot_matrix_3D


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


def multi_omp(sim):
    X_omp = np.zeros((len(sim.sources), sim.N))
    for f in range(sim.N):  # for each frequency, N = len(freqs)
        X_omp[:, f] = diy_omp(sim.C[:, :, f], sim.Y[:, f])

    return X_omp


if __name__ == "__main__":
    # sim = run_simulation(num_mics=3, no_sources=8, s_sparse=3)
    # X_lasso = pred_lasso(sim)
    A = np.array([[1, 0, 4, 5], [0, 2, 0, 6], [1, 1, 1, 7], [1, 2, 3, 4]])
    x = np.array([3, 5, 0, 0])
    y = A @ x
    print(y)
    X_omp, _ = diy_omp(A, y)
    print(X_omp)
    print(_)

    # X_omp = X_omp.reshape(-1, 1)
    # plot_equation(
    #     sim.X,
    #     X_lasso,
    #     X_omp,
    #     titles=("X", "X_lasso", "X_omp"),
    #     ratios=(1, 1, 1),
    # )
