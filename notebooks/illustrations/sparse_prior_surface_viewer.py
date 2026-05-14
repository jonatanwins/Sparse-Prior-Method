from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sparse_prior_visuals import (
    COLORS,
    make_2d_example,
    prior_objective,
    validate_example,
)

DEFAULT_WEIGHTS = (1.0, 1.0, 0.05)
DEFAULT_X1_BOUNDS = (-6, 9)
DEFAULT_X2_BOUNDS = (-6, 9)
DEFAULT_BOX_ASPECT = (1.0, 1.0, 1.0)


@dataclass(frozen=True)
class SurfaceViewData:
    X1: np.ndarray
    X2: np.ndarray
    objective: np.ndarray
    line_points: np.ndarray
    line_objective: np.ndarray


def _bounds(values: Iterable[float], name: str) -> tuple[float, float]:
    lower, upper = tuple(float(value) for value in values)
    if lower >= upper:
        raise ValueError(f"{name} lower bound must be smaller than upper bound.")
    return lower, upper


def build_grid(example, x1_bounds, x2_bounds, grid_size: int) -> tuple[np.ndarray, ...]:
    if grid_size < 8:
        raise ValueError("grid_size must be at least 8.")

    x1 = np.linspace(*x1_bounds, grid_size)
    x2 = np.linspace(*x2_bounds, grid_size)
    X1, X2 = np.meshgrid(x1, x2)
    points = np.stack([X1, X2], axis=-1)
    objective = prior_objective(points, example.precisions, example.prior_weights)
    return X1, X2, objective


def visible_solution_line(
    example, x1_bounds, x2_bounds, samples: int = 1200
) -> np.ndarray:
    direction = example.B[:, 0]
    z_candidates = []
    for dim, (lower, upper) in enumerate((x1_bounds, x2_bounds)):
        if np.isclose(direction[dim], 0.0):
            continue
        z_candidates.extend(
            [
                (lower - example.x0[dim]) / direction[dim],
                (upper - example.x0[dim]) / direction[dim],
            ]
        )

    if not z_candidates:
        raise ValueError("Cannot infer a visible z range for the solution line.")

    z = np.linspace(min(z_candidates), max(z_candidates), samples)
    line_points = example.x0[None, :] + z[:, None] * direction[None, :]
    mask = (
        (x1_bounds[0] <= line_points[:, 0])
        & (line_points[:, 0] <= x1_bounds[1])
        & (x2_bounds[0] <= line_points[:, 1])
        & (line_points[:, 1] <= x2_bounds[1])
    )
    line_points = line_points[mask]
    if line_points.size == 0:
        raise ValueError("The solution line does not pass through the plotted bounds.")
    return line_points


def build_surface_view(
    weights: Iterable[float] = DEFAULT_WEIGHTS,
    x1_bounds: tuple[float, float] = DEFAULT_X1_BOUNDS,
    x2_bounds: tuple[float, float] = DEFAULT_X2_BOUNDS,
    grid_size: int = 320,
    contours: int = 6,
    contour_offset: float = -0.935,
    surface_lift: float = 0.35,
    line_lift: float = 1.035,
    elev: float = 28.0,
    azim: float = -31.0,
    show_guides: bool = True,
) -> tuple[plt.Figure, SurfaceViewData]:
    if contours < 1:
        raise ValueError("contours must be at least 1.")

    example = make_2d_example(prior_weights=weights)
    validate_example(example)

    X1, X2, objective = build_grid(example, x1_bounds, x2_bounds, grid_size)
    line_points = visible_solution_line(example, x1_bounds, x2_bounds)
    line_objective = prior_objective(
        line_points, example.precisions, example.prior_weights
    )
    surface_z = objective + surface_lift
    line_z = line_objective + surface_lift + line_lift

    fig = plt.figure(figsize=(9.2, 7.0), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    surface_resolution = min(grid_size, 220)
    ax.plot_surface(
        X1,
        X2,
        surface_z,
        cmap="viridis",
        alpha=0.32,
        linewidth=0,
        antialiased=True,
        rcount=surface_resolution,
        ccount=surface_resolution,
    )
    ax.contour(
        X1,
        X2,
        objective,
        zdir="z",
        offset=contour_offset,
        levels=contours,
        cmap="viridis",
        linewidths=0.9,
    )

    ax.plot(
        line_points[:, 0],
        line_points[:, 1],
        np.full(line_points.shape[0], contour_offset),
        color=COLORS["solution"],
        linestyle="--",
        linewidth=2.0,
        label=r"$\mathbf{x}_0 + \mathbf{B}\mathbf{z}$",
    )
    ax.plot(
        line_points[:, 0],
        line_points[:, 1],
        line_z,
        color=COLORS["solution"],
        linewidth=2.4,
        label="surface intersection",
    )

    marked_points = [(r"$\mathbf{x}_0$", example.x0, COLORS["x0"], "o")]
    marked_points.extend(
        (label, point, COLORS["sparse"], "o")
        for label, point in example.sparse_points.items()
    )

    sparse_label_used = False
    for label, point, color, marker in marked_points:
        value = float(prior_objective(point, example.precisions, example.prior_weights))
        plotted_value = value + surface_lift + line_lift
        if label == r"$\mathbf{x}_0$":
            legend_label = label
            size = 44
        elif not sparse_label_used:
            legend_label = "sparse solutions"
            sparse_label_used = True
            size = 46
        else:
            legend_label = "_nolegend_"
            size = 46

        ax.scatter(
            point[0],
            point[1],
            plotted_value,
            s=size,
            color=color,
            marker=marker,
            edgecolor=COLORS["text"],
            linewidth=0.5,
            label=legend_label,
            depthshade=False,
        )
        if show_guides:
            ax.plot(
                [point[0], point[0]],
                [point[1], point[1]],
                [contour_offset, plotted_value],
                color=color,
                alpha=0.35,
                linestyle=":",
                linewidth=0.9,
            )

    ax.set_title(
        r"Sparse-prior objective with solution line for $\mathbf{y}=\mathbf{A}\mathbf{x}$",
        pad=16,
    )
    ax.set_xlabel(r"$x_1$", labelpad=8)
    ax.set_ylabel(r"$x_2$", labelpad=8)
    ax.set_zlabel(r"$P(x)$", labelpad=8)
    ax.set_xlim(*x1_bounds)
    ax.set_ylim(*x2_bounds)
    ax.set_zlim(contour_offset, float(np.nanmax(surface_z)) * 1.05 + line_lift)
    ax.set_box_aspect(DEFAULT_BOX_ASPECT)
    ax.view_init(elev=elev, azim=azim)
    ax.legend(loc="upper right", frameon=True, framealpha=0.94)

    data = SurfaceViewData(
        X1=X1,
        X2=X2,
        objective=objective,
        line_points=line_points,
        line_objective=line_objective,
    )
    return fig, data


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open the legacy sparse-prior 3D surface-over-contour viewer."
    )
    parser.add_argument("--weights", nargs=3, type=float, default=DEFAULT_WEIGHTS)
    parser.add_argument("--x1", nargs=2, type=float, default=DEFAULT_X1_BOUNDS)
    parser.add_argument("--x2", nargs=2, type=float, default=DEFAULT_X2_BOUNDS)
    parser.add_argument("--grid-size", type=int, default=320)
    parser.add_argument("--contours", type=int, default=7)
    parser.add_argument("--offset", type=float, default=-7.35)
    parser.add_argument(
        "--surface-lift",
        type=float,
        default=0.0,
        help="Add a visual z-offset to the surface, surface line, and markers.",
    )
    parser.add_argument(
        "--line-lift",
        type=float,
        default=0.0,
        help="Draw the surface intersection this much above the surface to avoid z-fighting.",
    )
    parser.add_argument("--elev", type=float, default=35.0)
    parser.add_argument("--azim", type=float, default=45.0)
    parser.add_argument("--save", type=Path)
    parser.add_argument(
        "--no-guides",
        action="store_true",
        help="Hide vertical guide lines from marked points to the contour plane.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Build and optionally save the figure without opening a window.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    x1_bounds = _bounds(args.x1, "--x1")
    x2_bounds = _bounds(args.x2, "--x2")

    fig, _ = build_surface_view(
        weights=args.weights,
        x1_bounds=x1_bounds,
        x2_bounds=x2_bounds,
        grid_size=args.grid_size,
        contours=args.contours,
        contour_offset=args.offset,
        surface_lift=args.surface_lift,
        line_lift=args.line_lift,
        elev=args.elev,
        azim=args.azim,
        show_guides=not args.no_guides,
    )

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=300, bbox_inches="tight", pad_inches=0.04)
        print(f"Saved {args.save}")

    if args.no_show:
        plt.close(fig)
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
