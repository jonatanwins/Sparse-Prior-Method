from sklearn.linear_model import Lasso
import numpy as np
from spaghetti_code.old_spaghetti.simulated_prior_optimization import to_real_augmented, from_real_augmented
from cs_priors.plotting.plotting import plot_equation, plot_two_line_equation


def complex_lasso():
    # Let X be a real vector of size 10, Let A be a real mixing matrix of size 8x10, Let Y be the measurements
    num_sources = 10
    mic_numbers = list(range(1, 20))
    mic_numbers = [8]
    s_sparse = 2

    # set the random seed for reproducibility
    np.random.seed(42)
    X_true = (
        np.random.randn(num_sources) * 10 + 1j * np.random.randn(num_sources) * 10
    )  # true source signals
    # choose sparse indicies
    zero_indices = np.random.choice(num_sources, num_sources - s_sparse, replace=False)
    X_true[zero_indices] = 0  # make X sparse

    X_true_real = to_real_augmented(X_true)
    plot_equation(
        X_true,
        X_true_real,
        np.array([X_true.real, X_true.imag]).reshape(-1, 1),
        ("X_true", "X_true_real", "X_true_separated"),
    )
    for mics in mic_numbers:
        A = np.random.randn(mics, num_sources) + 1j * np.random.randn(
            mics, num_sources
        )  # mixing matrix
        Y = A @ X_true  # measurements
        # Convert to real augmented form
        A_real = np.block([[A.real, -A.imag], [A.imag, A.real]])
        Y_real = to_real_augmented(Y)
        # Apply LASSO
        lasso = Lasso(alpha=0.1)
        lasso.fit(A_real, Y_real)
        X_lasso_real = lasso.coef_

        plot_two_line_equation(
            Y_real,
            X_true,
            X_true_real,
            ("Y_real", "X_true", "X_true_real"),
            X_lasso_real,
            X_lasso_real - X_true_real,
            from_real_augmented(X_lasso_real),
            ("X_lasso_real", "Error (LASSO - True)", "X_lasso_complex"),
        )


complex_lasso()
