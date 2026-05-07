import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cs_priors.simulation.mixing_model import quick_frequency_sim
from cs_priors.solvers.freq_lasso import frequency_lasso_solve
from cs_priors.solvers.freq_sparse_prior import sparse_prior_solve

DEFAULT_COLORS = {
    "X_pinv": "tab:blue",
    "r-LASSO": "tab:orange",
    "Group LASSO": "tab:green",
    "Sparse Prior": "tab:red",
}
DEFAULT_LINESTYLES = {
    "X_pinv": ":",
    "r-LASSO": "-",
    "Group LASSO": "-",
    "Sparse Prior": "-",
}


def plot_reconstruction_error_vs_separation(
    results_df,
    *,
    x_col="separation_deg",
    y_col="relative_complex_error",
    method_col="method",
    method_order=None,
    colors=None,
    linestyles=None,
    floor=1e-16,
    band=(0.25, 0.75),
    xlabel="Angular separation (degrees)",
    ylabel="Relative complex error",
    title="Reconstruction error vs. source angular separation",
    ax=None,
):
    """Plot median reconstruction error with a quantile band."""
    low, high = band
    summary_df = (
        results_df.groupby([method_col, x_col], observed=True)[y_col]
        .agg(
            median="median",
            lower=lambda x: x.quantile(low),
            upper=lambda x: x.quantile(high),
        )
        .reset_index()
    )
    method_order = (
        list(pd.unique(results_df[method_col].dropna()))
        if method_order is None
        else method_order
    )
    colors = DEFAULT_COLORS | (colors or {})
    linestyles = DEFAULT_LINESTYLES | (linestyles or {})
    fig, ax = plt.subplots(figsize=(7.0, 4.4)) if ax is None else (ax.figure, ax)

    for method in method_order:
        method_df = summary_df[summary_df[method_col] == method].sort_values(x_col)
        if method_df.empty:
            continue
        x = method_df[x_col].to_numpy(dtype=float)
        median = np.maximum(method_df["median"].to_numpy(dtype=float), floor)
        lower = np.maximum(method_df["lower"].to_numpy(dtype=float), floor)
        upper = np.maximum(method_df["upper"].to_numpy(dtype=float), floor)
        (line,) = ax.plot(
            x,
            median,
            marker="o",
            label=method,
            color=colors.get(method),
            linestyle=linestyles.get(method, "-"),
        )
        ax.fill_between(
            x, lower, upper, color=line.get_color(), alpha=0.18, linewidth=0
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", linewidth=0.5, alpha=0.35)
    ax.legend(frameon=False)
    return fig, ax, summary_df


def relative_complex_error(X_hat, X_true):
    return np.linalg.norm(X_hat - X_true) / np.linalg.norm(X_true)


def make_simulation(separation_deg, phase_seed, **sim_kwargs):
    sep_rad = np.deg2rad(separation_deg)
    kwargs = dict(
        array_type="arc",
        mic_angle_span=sep_rad,
        source_angle_span=sep_rad,
        sensor_snr_db=None,
        model_snr_db=None,
        inverse_method="mp",
    )
    kwargs.update(sim_kwargs)
    kwargs["seed"] = int(phase_seed)
    return quick_frequency_sim(**kwargs)


def solve_methods(
    sim, *, lasso_alpha=1e-4, lasso_max_iter=10000, sparse_grouping="none"
):
    X_pinv = sim.X_pinv
    return {
        "r-LASSO": frequency_lasso_solve(
            sim.Y, sim.A, alpha=lasso_alpha, max_iter=lasso_max_iter, X_start=X_pinv
        ),
        "Sparse Prior": sparse_prior_solve(X_pinv, sim.A, grouping=sparse_grouping),
        "X_pinv": X_pinv,
    }


def run_angle_separation_benchmark(
    separations_deg, phase_seeds, *, sim_kwargs, **solve_kwargs
):
    rows = []
    for separation_deg in separations_deg:
        for phase_seed in phase_seeds:
            sim = make_simulation(float(separation_deg), int(phase_seed), **sim_kwargs)
            for method, X_hat in solve_methods(sim, **solve_kwargs).items():
                rows.append(
                    {
                        "separation_deg": float(separation_deg),
                        "phase_seed": int(phase_seed),
                        "method": method,
                        "relative_complex_error": relative_complex_error(X_hat, sim.X),
                    }
                )
    return pd.DataFrame(rows)
