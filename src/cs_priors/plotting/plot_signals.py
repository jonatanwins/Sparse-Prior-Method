import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import ifft
from typeguard import typechecked

from ..simulation.Simulation import Simulation


@typechecked
def plot_signal_on_ax(
    ax,
    signal: np.ndarray,
    t: np.ndarray | None = None,
    label: str | None = None,
    line_color: str = "black",
    sample_color: str = "gold",
):
    """
    Plot a single 1D waveform on the given Axes with sample markers.

    Args:
        ax: Matplotlib Axes to plot on.
        signal: 1D array of shape (N,).
        t: Optional time vector of shape (N,). If None, uses sample indices.
        label: Optional y-axis label for this subplot.
        line_color: Color of the line.
        sample_color: Color of the sample-point markers.
    """
    x_axis = t if t is not None else np.arange(len(signal))
    ax.plot(x_axis, signal, color=line_color, linewidth=0.8, zorder=2)
    ax.scatter(x_axis, signal, color=sample_color, s=2, zorder=1)
    if label is not None:
        ax.set_ylabel(label)
    ax.grid(True)


@typechecked
def plot_signals(
    x: np.ndarray,
    t: np.ndarray | None = None,
    labels: list[str] | None = None,
    title: str | None = None,
    line_color: str = "black",
    sample_color: str = "gold",
    figsize_width: float = 5,
    height_per_signal: float = 1.0,
):
    """
    Create a column of subplots, one per signal row in x.

    Args:
        x: (S x N) array of signals.
        t: Optional time vector of shape (N,). If None, uses sample indices.
        labels: Per-signal y-axis labels. Defaults to "Source 0", "Source 1", ...
        title: Optional suptitle for the figure.
        line_color: Color for the signal line.
        sample_color: Color for the sample-point markers.

    Returns:
        fig, axes: The created Figure and array of Axes.
    """
    S = x.shape[0]

    if labels is None:
        labels = [f"Source {i}" for i in range(S)]

    fig, axes = plt.subplots(
        S,
        1,
        figsize=(figsize_width, height_per_signal * S),
        sharex=True,
    )
    # Handle S == 1 case where axes is not an array
    if S == 1:

        axes = np.array([axes])

    for i in range(S):

        plot_signal_on_ax(
            axes[i],
            x[i],
            t=t,
            label=labels[i],
            line_color=line_color,
            sample_color=sample_color,
        )

    # Shared amplitude axis: use the global max across all signals
    y_max = np.max(np.abs(x))
    if y_max > 0:
        for ax in axes:
            ax.set_ylim(-y_max * 1.1, y_max * 1.1)

    axes[-1].set_xlabel("Time (s)" if t is not None else "Sample index")

    if title is not None:
        fig.suptitle(title, fontsize=14)

    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Magnitude spectrum
# ---------------------------------------------------------------------------


@typechecked
def plot_magnitude_spectrum(
    Z: np.ndarray,
    freqs: np.ndarray,
    labels: list[str] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (5, 2),
):
    """
    Plot |Z| vs frequency (positive frequencies only).

    Args:
        Z: Complex array (K x N) — e.g. sim.X (S x N) or sim.Y (M x N).
        freqs: 1D real array (N,) from scipy.fft.fftfreq.
        labels: Per-row legend labels.  Defaults to "Row 0", …
        title: Optional figure title.
        figsize: Figure size in inches.

    Returns:
        fig, ax
    """
    K, N = Z.shape
    half = N // 2
    f_pos = freqs[:half]
    mag = np.abs(Z[:, :half])

    if labels is None:
        labels = [f"Row {i}" for i in range(K)]

    fig, ax = plt.subplots(figsize=figsize)
    for i in range(K):
        ax.plot(f_pos, mag[i], label=labels[i])

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    if title is not None:
        ax.set_title(title)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Recovery comparison
# ---------------------------------------------------------------------------


def plot_recovery(
    sim: Simulation,
    X_pred: np.ndarray,
    title: str | None = None,
    figsize_width: float = 10,
    height_per_signal: float = 2.0,
    dpi=None,
    mode: str = "compact",
):
    """
    Compare recovered source signals against ground truth.

    Args:
        sim: Original Simulation (provides x, active_indices, duration, etc.)
        X_pred: (S x N) complex predicted spectra from a solver.
        title: Optional suptitle.
        figsize_width: Figure width (inches).
        height_per_signal: Height per subplot (inches).
        dpi: Figure resolution.
        mode: Display mode:
            "all"         – one subplot per source (original behaviour).
            "compact"     – one subplot per active source + one combined
                            subplot for all mute sources (default).
            "active_only" – only active sources, mute sources omitted.

    Returns:
        fig, axes
    """
    if mode not in ("all", "compact", "active_only"):
        raise ValueError(
            f"Unknown mode {mode!r}. Choose 'all', 'compact', or 'active_only'."
        )

    S, N = sim.x.shape
    x_pred = ifft(X_pred, axis=1).real
    t = np.linspace(0, sim.duration, N, endpoint=False)
    active_set = set(sim.active_indices)

    # Shared y-axis scale across all subplots
    y_max = max(np.max(np.abs(sim.x)), np.max(np.abs(x_pred)))

    if mode == "all":
        return _plot_recovery_all(
            sim,
            x_pred,
            t,
            S,
            active_set,
            y_max,
            title,
            figsize_width,
            height_per_signal,
            dpi,
        )
    elif mode == "active_only":
        return _plot_recovery_active_only(
            sim,
            x_pred,
            t,
            active_set,
            y_max,
            title,
            figsize_width,
            height_per_signal,
            dpi,
        )
    else:  # compact
        return _plot_recovery_compact(
            sim,
            x_pred,
            t,
            S,
            active_set,
            y_max,
            title,
            figsize_width,
            height_per_signal,
            dpi,
        )


def _set_ylim(axes, y_max):
    if y_max > 0:
        for ax in axes if hasattr(axes, '__iter__') else [axes]:
            ax.set_ylim(-y_max * 1.1, y_max * 1.1)


def _annotate_source(ax, t, x_true, x_pred, label, y_max, is_active=True, legend=False):
    alpha = 1.0 if is_active else 0.35
    ax.plot(t, x_true, "k-", label="Original", alpha=alpha, linewidth=0.8)
    ax.plot(t, x_pred, "r--", label="Recovered", alpha=alpha, linewidth=0.8)
    mse = np.mean((x_true - x_pred) ** 2)
    ax.set_ylabel(label)
    ax.text(
        0.98,
        0.92,
        f"MSE={mse:.2e}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="red",
    )
    ax.grid(True)
    if legend:
        ax.legend(loc="upper left", fontsize=8)


def _plot_recovery_all(
    sim, x_pred, t, S, active_set, y_max, title, figsize_width, height_per_signal, dpi
):
    """One subplot per source."""
    fig, axes = plt.subplots(
        S, 1, figsize=(figsize_width, height_per_signal * S), sharex=True, dpi=dpi
    )
    if S == 1:
        axes = np.array([axes])
    _set_ylim(axes, y_max)
    for i in range(S):
        is_active = i in active_set
        tag = "active" if is_active else "mute"
        _annotate_source(
            axes[i],
            t,
            sim.x[i],
            x_pred[i],
            f"S{i} ({tag})",
            y_max,
            is_active=is_active,
            legend=(i == 0),
        )
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title or "Recovery: original x vs ifft(X_pred)")
    fig.tight_layout()
    return fig, axes


def _plot_recovery_active_only(
    sim, x_pred, t, active_set, y_max, title, figsize_width, height_per_signal, dpi
):
    """Only active sources."""
    active_list = sorted(active_set)
    n_plots = max(len(active_list), 1)
    fig, axes = plt.subplots(
        n_plots,
        1,
        figsize=(figsize_width, height_per_signal * n_plots),
        sharex=True,
        dpi=dpi,
    )
    if n_plots == 1:
        axes = np.array([axes])
    _set_ylim(axes, y_max)
    for j, i in enumerate(active_list):
        _annotate_source(
            axes[j],
            t,
            sim.x[i],
            x_pred[i],
            f"S{i} (active)",
            y_max,
            is_active=True,
            legend=(j == 0),
        )
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title or "Recovery (active sources only)")
    fig.tight_layout()
    return fig, axes


def _plot_recovery_compact(
    sim, x_pred, t, S, active_set, y_max, title, figsize_width, height_per_signal, dpi
):
    """One subplot per active source + one combined subplot for all mute sources."""
    active_list = sorted(active_set)
    mute_list = sorted(set(range(S)) - active_set)
    n_plots = len(active_list) + (1 if mute_list else 0)
    n_plots = max(n_plots, 1)

    fig, axes = plt.subplots(
        n_plots,
        1,
        figsize=(figsize_width, height_per_signal * n_plots),
        sharex=True,
        dpi=dpi,
    )
    if n_plots == 1:
        axes = np.array([axes])
    _set_ylim(axes, y_max)

    # Active sources — one subplot each
    for j, i in enumerate(active_list):
        _annotate_source(
            axes[j],
            t,
            sim.x[i],
            x_pred[i],
            f"S{i} (active)",
            y_max,
            is_active=True,
            legend=(j == 0),
        )

    # Mute sources — all stacked in one subplot
    if mute_list:
        ax_mute = axes[-1]
        for k, i in enumerate(mute_list):
            lbl_orig = "Original" if k == 0 else None
            lbl_rec = "Recovered" if k == 0 else None
            ax_mute.plot(t, sim.x[i], "k-", alpha=0.2, linewidth=0.5, label=lbl_orig)
            ax_mute.plot(t, x_pred[i], "r--", alpha=0.2, linewidth=0.5, label=lbl_rec)
        # Aggregate MSE over all mute sources
        mute_mse = np.mean((sim.x[mute_list] - x_pred[mute_list]) ** 2)
        ax_mute.set_ylabel(f"Mute ({len(mute_list)} src)")
        ax_mute.text(
            0.98,
            0.92,
            f"MSE={mute_mse:.2e}",
            transform=ax_mute.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="red",
        )
        ax_mute.grid(True)
        ax_mute.legend(loc="upper left", fontsize=8)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title or "Recovery (compact)")
    fig.tight_layout()
    return fig, axes
