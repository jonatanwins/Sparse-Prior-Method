"""
Batch benchmarking for solver comparisons.

Runs a dict of solver functions on the same simulation(s) and collects
metrics into a flat list of dicts (easy to convert to a DataFrame).

Usage
-----
    from cs_priors.metrics.benchmark import grid_benchmark
    from cs_priors.solvers.group_lasso import group_lasso_solve

    methods = {
        "GL-freq": lambda sim: group_lasso_solve(sim, alpha=1e-4, grouping="frequency"),
        "GL-none": lambda sim: group_lasso_solve(sim, alpha=1e-4, grouping="none"),
    }
    rows = grid_benchmark(quick_sector_sim, param_grid, methods, num_seeds=5)
"""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed

from ..simulation.Simulation import Simulation
from .count_sparsity import score


# ---------------------------------------------------------------------------
# Single‑sim helpers
# ---------------------------------------------------------------------------

def run_methods(
    sim: Simulation,
    methods: dict[str, callable],
) -> dict[str, np.ndarray]:
    """Run every method on *sim*, return ``{name: X_pred}``."""
    return {name: fn(sim) for name, fn in methods.items()}


def score_methods(
    sim: Simulation,
    predictions: dict[str, np.ndarray],
    tol: float | None = None,
) -> dict[str, dict[str, float]]:
    """Score each prediction, return ``{name: scores_dict}``."""
    return {
        name: score(sim, X_pred, tol=tol)
        for name, X_pred in predictions.items()
    }


# ---------------------------------------------------------------------------
# Grid sweep
# ---------------------------------------------------------------------------

def _run_one(
    sim_factory: callable,
    sim_kwargs: dict,
    methods: dict[str, callable],
    seed: int,
    tol: float | None,
) -> list[dict]:
    """Run all methods on one simulation, return list of result rows."""
    sim = sim_factory(**sim_kwargs, seed=seed)
    predictions = run_methods(sim, methods)
    all_scores = score_methods(sim, predictions, tol=tol)

    rows = []
    for name, scores in all_scores.items():
        row = {**sim_kwargs, "seed": seed, "method": name, **scores}
        rows.append(row)
    return rows


def grid_benchmark(
    sim_factory: callable,
    param_grid: dict[str, list],
    methods: dict[str, callable],
    num_seeds: int = 10,
    tol: float | None = None,
    n_jobs: int = -1,
) -> list[dict]:
    """
    Run every method on every combination of parameters × seeds.

    Parameters
    ----------
    sim_factory : callable
        Function like ``quick_sector_sim`` that accepts keyword arguments
        including ``seed``.
    param_grid : dict[str, list]
        Keys are kwarg names for *sim_factory*, values are lists to sweep.
        Example: ``{"num_mics": [4, 6, 8], "num_active": [1, 2, 3]}``
    methods : dict[str, callable]
        ``{display_name: fn}`` where ``fn(sim) -> X_pred (S×N)``.
    num_seeds : int
        Number of random seeds per parameter combination.
    tol : float, optional
        Detection threshold override (default: auto from each sim.X).
    n_jobs : int
        Joblib parallelism (``-1`` = all cores).

    Returns
    -------
    list[dict]
        One dict per (param_combo, seed, method).  Ready for
        ``pd.DataFrame(rows)``.
    """
    import itertools

    keys = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))

    tasks = [
        (dict(zip(keys, combo)), seed)
        for combo in combos
        for seed in range(num_seeds)
    ]

    results = Parallel(n_jobs=n_jobs)(
        delayed(_run_one)(sim_factory, kw, methods, seed, tol)
        for kw, seed in tasks
    )

    # Flatten list of lists
    return [row for batch in results for row in batch]


# ---------------------------------------------------------------------------
# Heatmap convenience wrapper
# ---------------------------------------------------------------------------

def heatmap_benchmark(
    sim_factory: callable,
    param_grid: dict[str, list],
    methods: dict[str, callable],
    num_seeds: int = 10,
    pivot_x: str = "num_active",
    pivot_y: str = "num_mics",
    metric: str = "f1",
    tol: float | None = None,
    n_jobs: int = -1,
    vmin: float = 0.0,
    vmax: float = 1.0,
    figsize: tuple[float, float] = (5, 3),
):
    """
    Run grid_benchmark and display one heatmap per method.

    Parameters
    ----------
    sim_factory, param_grid, methods, num_seeds, tol, n_jobs
        Forwarded to ``grid_benchmark``.
    pivot_x : str
        param_grid key used as heatmap columns (default ``"num_active"``).
    pivot_y : str
        param_grid key used as heatmap rows (default ``"num_mics"``).
    metric : str
        Score column to plot. One of: ``"f1"``, ``"precision"``,
        ``"recall"``, ``"relative_error"``.
    vmin, vmax : float
        Colour-scale limits (default 0–1).
    figsize : tuple
        Size of each individual heatmap figure.

    Returns
    -------
    pd.DataFrame
        Full results table (one row per param_combo × seed × method).
    """
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    rows = grid_benchmark(
        sim_factory=sim_factory,
        param_grid=param_grid,
        methods=methods,
        num_seeds=num_seeds,
        tol=tol,
        n_jobs=n_jobs,
    )
    df = pd.DataFrame(rows)

    for method_name in methods:
        subset = df[df["method"] == method_name]
        pivot = subset.pivot_table(
            index=pivot_y,
            columns=pivot_x,
            values=metric,
            aggfunc="mean",
        )
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            vmin=vmin,
            vmax=vmax,
            ax=ax,
        )
        ax.set_title(f"{method_name}  —  mean {metric}")
        ax.set_xlabel(pivot_x)
        ax.set_ylabel(pivot_y)
        fig.tight_layout()
        plt.show()

    return df
