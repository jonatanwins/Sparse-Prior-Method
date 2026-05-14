import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cs_priors.simulation.mixing_model import (
    quick_frequency_sim,
    moore_penrose_inverse,
    ridge_inverse,
)
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
METHOD_LABELS = {
    "r-LASSO": "LASSO",
    "Sparse Prior": "Sparse Prior",
    "X_pinv": "Moore-Penrose",
}

METHOD_ORDER = ["Sparse Prior", "r-LASSO", "X_pinv"]


def relabel_legend(ax, labels):
    handles, old_labels = ax.get_legend_handles_labels()
    ax.legend(
        handles, [labels.get(label, label) for label in old_labels], frameon=False
    )


def plot_metric(
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
    xscale="log",
    yscale="log",
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

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", linewidth=0.5, alpha=0.35)
    ax.legend(frameon=False)
    return fig, ax, summary_df


plot_reconstruction_error = plot_metric


def relative_complex_error(X_hat, X_true):
    return np.linalg.norm(X_hat - X_true) / np.linalg.norm(X_true)


def source_scores(X_hat):
    return np.linalg.norm(X_hat, axis=1)


def f1_threshold_fraction(X_hat, active_indices, fraction=0.10):
    scores = source_scores(X_hat)

    y_true = np.zeros(scores.shape[0], dtype=bool)
    y_true[list(active_indices)] = True

    if scores.max() <= 0:
        y_pred = np.zeros_like(y_true)
    else:
        y_pred = scores >= fraction * scores.max()

    tp = np.sum(y_true & y_pred)
    fp = np.sum(~y_true & y_pred)
    fn = np.sum(y_true & ~y_pred)

    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else 2 * tp / denom


def solution_metrics(X_hat, sim):
    return {
        "relative_complex_error": relative_complex_error(X_hat, sim.X),
        "f1_threshold_10_percent": f1_threshold_fraction(
            X_hat,
            sim.active_indices,
            fraction=0.10,
        ),
    }


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


def make_initial_solution(A, Y, *, method="mp", ridge_alpha=None):
    if method == "mp":
        return moore_penrose_inverse(A, Y)

    if method == "ridge":
        if ridge_alpha is None:
            raise ValueError("ridge_alpha must be provided for ridge initialization")
        if ridge_alpha <= 0:
            return moore_penrose_inverse(A, Y)
        return ridge_inverse(A, Y, ridge_alpha)

    raise ValueError("method must be 'mp' or 'ridge'")


def solve_methods(
    sim, *, X0=None, lasso_alpha=1e-4, lasso_max_iter=10000, sparse_grouping="none"
):
    X0 = sim.X_pinv if X0 is None else X0
    return {
        "r-LASSO": frequency_lasso_solve(
            sim.Y, sim.A, alpha=lasso_alpha, max_iter=lasso_max_iter, X_start=X0
        ),
        "Sparse Prior": sparse_prior_solve(X0, sim.A, grouping=sparse_grouping),
        "X_pinv": X0,
    }


def plot_source_amplitudes(
    sim,
    solutions,
    *,
    labels=None,
    title=None,
    subplots=True,
    highlight_active=True,
):
    labels = labels or {}
    amplitudes = {
        r"$\boldsymbol{x}_{true}$": np.abs(sim.X[:, 0]),
        **{
            labels.get(name, name): np.abs(X_hat[:, 0])
            for name, X_hat in solutions.items()
        },
    }

    source_index = np.arange(sim.X.shape[0])
    title = title or "Example source amplitudes"

    if not subplots:
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        width = 0.8 / len(amplitudes)
        offsets = (np.arange(len(amplitudes)) - (len(amplitudes) - 1) / 2) * width

        for offset, (label, amplitude) in zip(offsets, amplitudes.items()):
            ax.bar(source_index + offset, amplitude, width=width, label=label)

        ax.set_xticks(source_index)
        ax.set_xlabel("Source index")
        ax.set_ylabel("Single-frequency coefficient amplitude")
        ax.set_title(title)
        ax.grid(True, axis="y", linewidth=0.5, alpha=0.35)
        ax.legend(frameon=False, ncols=2)
        fig.tight_layout()
        return fig, ax

    fig, axes = plt.subplots(
        1,
        len(amplitudes),
        figsize=(2.2 * len(amplitudes), 3.2),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes)

    colors = np.full(len(source_index), "tab:blue", dtype=object)
    if highlight_active:
        colors[list(sim.active_indices)] = "gold"

    for ax, (label, amplitude) in zip(axes, amplitudes.items()):
        ax.bar(source_index, amplitude, color=colors, edgecolor="black", linewidth=0.3)
        ax.set_title(label)
        ax.set_xlabel("Source index")
        ax.grid(True, axis="y", linewidth=0.5, alpha=0.35)

    axes[0].set_ylabel("Amplitude")
    fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


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
                        **solution_metrics(X_hat, sim),
                    }
                )
    return pd.DataFrame(rows)


def run_microphone_count_benchmark(
    num_mics_values,
    phase_seeds,
    *,
    separation_deg,
    sim_kwargs,
    **solve_kwargs,
):
    rows = []
    for num_mics in num_mics_values:
        for phase_seed in phase_seeds:
            sim = make_simulation(
                separation_deg, int(phase_seed), num_mics=int(num_mics), **sim_kwargs
            )
            for method, X_hat in solve_methods(sim, **solve_kwargs).items():
                rows.append(
                    {
                        "num_mics": int(num_mics),
                        "phase_seed": int(phase_seed),
                        "method": method,
                        **solution_metrics(X_hat, sim),
                    }
                )
    return pd.DataFrame(rows)


def run_snr_benchmark(
    snr_db_values,
    phase_seeds,
    separation_deg,
    recovery_methods,
    *,
    sim_kwargs,
):
    finite_snr = [float(snr) for snr in snr_db_values if snr is not None]
    noiseless_plot_snr_db = max(finite_snr) + 40.0 if finite_snr else 120.0
    rows = []

    for snr_db in snr_db_values:
        sim_snr_db = None if snr_db is None else float(snr_db)
        plot_snr_db = noiseless_plot_snr_db if snr_db is None else sim_snr_db
        snr_label = "Noiseless" if snr_db is None else f"{sim_snr_db:g} dB"

        for phase_seed in phase_seeds:
            sim = make_simulation(
                separation_deg,
                int(phase_seed),
                sensor_snr_db=sim_snr_db,
                **sim_kwargs,
            )
            initializers = {}

            for plot_label, settings in recovery_methods.items():
                init_method = settings["initializer"]
                ridge_alpha = settings.get("ridge_alpha", None)
                init_key = (init_method, ridge_alpha)

                if init_key not in initializers:
                    initializers[init_key] = make_initial_solution(
                        sim.A,
                        sim.Y,
                        method=init_method,
                        ridge_alpha=ridge_alpha,
                    )
                X0 = initializers[init_key]

                if settings["solver"] == "initializer":
                    X_hat = X0

                elif settings["solver"] == "lasso":
                    X_hat = frequency_lasso_solve(
                        sim.Y,
                        sim.A,
                        alpha=settings["lasso_alpha"],
                        max_iter=settings["lasso_max_iter"],
                        X_start=X0,
                    )

                elif settings["solver"] == "sparse_prior":
                    X_hat = sparse_prior_solve(
                        X0,
                        sim.A,
                        grouping=settings["sparse_grouping"],
                        precision=settings["sparse_precision"],
                        eps=settings["sparse_eps"],
                    )

                else:
                    raise ValueError(f"Unknown solver {settings['solver']!r}")

                rows.append(
                    {
                        "sensor_snr_db": plot_snr_db,
                        "sensor_snr_label": snr_label,
                        "phase_seed": int(phase_seed),
                        "method": plot_label,
                        "solver": settings["solver"],
                        "initializer": init_method,
                        "ridge_alpha": np.nan if ridge_alpha is None else ridge_alpha,
                        **solution_metrics(X_hat, sim),
                    }
                )

    return pd.DataFrame(rows)
