import matplotlib.pyplot as plt
import seaborn as sns


def plot_benchmark_confidence_intervals(
    df,
    x="sensor_snr_db",
    metrics=("precision", "f1"),
    hue="method",
    x_label=None,
    y_lim=(0.0, 1.05),
    figsize=(12, 4),
):
    """
    Plot mean performance curves with 95% confidence intervals for benchmark results.

    Parameters
    ----------
    df : pd.DataFrame
        Benchmark table with columns like x, hue, and the metric names.
    x : str
        Column name for the x-axis.
    metrics : tuple[str, ...]
        Metric columns to plot in separate panels.
    hue : str
        Column used to separate methods.
    x_label : str | None
        Optional axis label override.
    y_lim : tuple[float, float]
        Shared y-limits for all subplots.
    figsize : tuple[float, float]
        Figure size.
    """
    if x_label is None:
        x_label = x

    fig, axes = plt.subplots(1, len(metrics), figsize=figsize, sharex=True, sharey=True)

    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        sns.lineplot(
            data=df,
            x=x,
            y=metric,
            hue=hue,
            estimator="mean",
            errorbar=("ci", 95),
            marker="o",
            ax=ax,
        )
        ax.set_title(f"{metric.upper()} vs {x}")
        ax.set_xlabel(x_label)
        ax.set_ylabel(metric)
        ax.set_ylim(*y_lim)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, axes
