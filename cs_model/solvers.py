import numpy as np
from sklearn.linear_model import Lasso
from app import run_simulation
from plotting import plot_geometry_auto, plot_overview, plot_equation


def pred_lasso(sim, alpha=1):
    X_lasso = np.zeros_like(sim.X)
    model = Lasso(alpha=alpha, fit_intercept=False)
    # if Y is 2D,
    for j in range(sim.Y.shape[1]):
        model.fit(sim.C, sim.Y[:, j])
        X_lasso[:, j] = model.coef_

    plot_equation(
        sim.X_pred, sim.X, X_lasso, titles=("X_pred", "X", "X_lasso"), ratios=(1, 1, 1)
    )


if __name__ == "__main__":
    sim = run_simulation(num_mics=3, no_sources=8, s_sparse=3)
    pred_lasso(sim)
