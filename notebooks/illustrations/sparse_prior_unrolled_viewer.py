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
    SparsePriorExample,
    configure_matplotlib,
    make_3d_example,
    prior_objective,
    validate_example,
)

DEFAULT_CASE = ["grouped", "thesis"][1]
DEFAULT_Z1_BOUNDS = (-2.6, 3.2)
DEFAULT_Z2_BOUNDS = (-2.6, 3.2)
DEFAULT_GRID_SIZE = 260
DEFAULT_CONTOURS = 12
DEFAULT_CONTOUR_OFFSET = -0.5
DEFAULT_SURFACE_ALPHA = 0.82
DEFAULT_MARKER_LIFT = 0.025
DEFAULT_ELEV = 10.0
DEFAULT_AZIM = -26.0
DEFAULT_BOX_ASPECT = (1.0, 1.0, 1.0)
DEFAULT_GROUPED_WEIGHTS = (1.0, 1.0)
DEFAULT_THESIS_WEIGHTS = (1.0, 1.0, 1.0)
DEFAULT_SHOW_CONTOURS = True
DEFAULT_SHOW_GUIDES = True


@dataclass(frozen=True)
class UnrolledObjectiveData:
    Z1: np.ndarray
    Z2: np.ndarray
    plane_points: np.ndarray
    objective: np.ndarray


def make_grouped_3d_example(
    prior_weights: Iterable[float] = DEFAULT_GROUPED_WEIGHTS,
) -> SparsePriorExample:
    x_true = np.array([0.0, 0.0, 3.0])
    A = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    y = np.array([3.0, 6.0])
    x0 = np.linalg.pinv(A) @ y
    B = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
    precisions = (
        np.diag([1.0, 1.0, 0.005]),
        np.diag([0.005, 0.005, 1.0]),
    )
    prior_weights = tuple(float(weight) for weight in prior_weights)
    sparse_points = {
        r"$(0,0,3)^\top$": np.array([0.0, 0.0, 3.0]),
        r"$(3,0,0)^\top$": np.array([3.0, 0.0, 0.0]),
        r"$(0,3,0)^\top$": np.array([0.0, 3.0, 0.0]),
        r"$(1.5,1.5,0)^\top$": np.array([1.5, 1.5, 0.0]),
    }
    sparse_z = {
        r"$(0,0,3)^\top$": np.array([-1.0, 2.0]),
        r"$(3,0,0)^\top$": np.array([-1.0, -1.0]),
        r"$(0,3,0)^\top$": np.array([2.0, -1.0]),
        r"$(1.5,1.5,0)^\top$": np.array([0.5, -1.0]),
    }
    return SparsePriorExample(
        x_true, A, y, x0, B, precisions, prior_weights, sparse_points, sparse_z
    )


def _bounds(values: Iterable[float], name: str) -> tuple[float, float]:
    lower, upper = tuple(float(value) for value in values)
    if lower >= upper:
        raise ValueError(f"{name} lower bound must be smaller than upper bound.")
    return lower, upper


def _weights_for_case(case: str, weights: Iterable[float] | None) -> tuple[float, ...]:
    if weights is None:
        return DEFAULT_GROUPED_WEIGHTS if case == "grouped" else DEFAULT_THESIS_WEIGHTS
    return tuple(float(weight) for weight in weights)


def make_example(
    case: str, weights: Iterable[float] | None = None
) -> SparsePriorExample:
    case_weights = _weights_for_case(case, weights)
    if case == "grouped":
        example = make_grouped_3d_example(case_weights)
    elif case == "thesis":
        example = make_3d_example(case_weights)
    else:
        raise ValueError("case must be 'grouped' or 'thesis'.")

    if len(example.prior_weights) != len(example.precisions):
        raise ValueError(
            f"Expected {len(example.precisions)} weights for case {case!r}, "
            f"got {len(example.prior_weights)}."
        )
    validate_example(example)
    return example


def build_unrolled_objective(
    example: SparsePriorExample,
    z1_bounds: tuple[float, float] = DEFAULT_Z1_BOUNDS,
    z2_bounds: tuple[float, float] = DEFAULT_Z2_BOUNDS,
    grid_size: int = DEFAULT_GRID_SIZE,
) -> UnrolledObjectiveData:
    if grid_size < 8:
        raise ValueError("grid_size must be at least 8.")

    z1 = np.linspace(*z1_bounds, grid_size)
    z2 = np.linspace(*z2_bounds, grid_size)
    Z1, Z2 = np.meshgrid(z1, z2)
    z_grid = np.stack([Z1, Z2], axis=-1)
    plane_points = example.x0 + np.einsum("ij,...j->...i", example.B, z_grid)
    objective = prior_objective(plane_points, example.precisions, example.prior_weights)
    return UnrolledObjectiveData(Z1, Z2, plane_points, objective)


def _point_value(example: SparsePriorExample, z_value: np.ndarray) -> float:
    point = example.x0 + example.B @ z_value
    return float(prior_objective(point, example.precisions, example.prior_weights))


def build_unrolled_view(
    case: str = DEFAULT_CASE,
    weights: Iterable[float] | None = None,
    z1_bounds: tuple[float, float] = DEFAULT_Z1_BOUNDS,
    z2_bounds: tuple[float, float] = DEFAULT_Z2_BOUNDS,
    grid_size: int = DEFAULT_GRID_SIZE,
    contours: int = DEFAULT_CONTOURS,
    contour_offset: float = DEFAULT_CONTOUR_OFFSET,
    surface_alpha: float = DEFAULT_SURFACE_ALPHA,
    marker_lift: float = DEFAULT_MARKER_LIFT,
    elev: float = DEFAULT_ELEV,
    azim: float = DEFAULT_AZIM,
    box_aspect: tuple[float, float, float] = DEFAULT_BOX_ASPECT,
    show_contours: bool = DEFAULT_SHOW_CONTOURS,
    show_guides: bool = DEFAULT_SHOW_GUIDES,
) -> tuple[plt.Figure, UnrolledObjectiveData]:
    if contours < 1:
        raise ValueError("contours must be at least 1.")
    if not 0.0 <= surface_alpha <= 1.0:
        raise ValueError("surface_alpha must be between 0 and 1.")

    configure_matplotlib()
    example = make_example(case, weights)
    data = build_unrolled_objective(example, z1_bounds, z2_bounds, grid_size)

    fig = plt.figure(figsize=(8.0, 6.8), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    surface_resolution = min(grid_size, 220)
    ax.plot_surface(
        data.Z1,
        data.Z2,
        data.objective,
        cmap="viridis",
        alpha=surface_alpha,
        linewidth=0,
        antialiased=True,
        rcount=surface_resolution,
        ccount=surface_resolution,
    )

    if show_contours:
        ax.contour(
            data.Z1,
            data.Z2,
            data.objective,
            zdir="z",
            offset=contour_offset,
            levels=contours,
            cmap="viridis",
            linewidths=0.85,
        )

    x0_z = np.array([0.0, 0.0])
    x0_value = _point_value(example, x0_z)
    x0_plot_value = x0_value + marker_lift
    ax.scatter(
        x0_z[0],
        x0_z[1],
        x0_plot_value,
        s=56,
        color=COLORS["x0"],
        edgecolor=COLORS["text"],
        linewidth=0.6,
        label=r"$\mathbf{x}_0$",
        depthshade=False,
    )

    sparse_label_used = False
    for label, z_value in example.sparse_z.items():
        z_value = np.asarray(z_value, dtype=float)
        value = _point_value(example, z_value)
        plot_value = value + marker_lift
        ax.scatter(
            z_value[0],
            z_value[1],
            plot_value,
            s=54,
            color=COLORS["sparse"],
            edgecolor=COLORS["text"],
            linewidth=0.55,
            label="sparse points" if not sparse_label_used else "_nolegend_",
            depthshade=False,
        )
        ax.text(
            z_value[0],
            z_value[1],
            plot_value,
            f"  {label}",
            color=COLORS["text"],
            fontsize=8,
        )
        sparse_label_used = True
        if show_guides:
            ax.plot(
                [z_value[0], z_value[0]],
                [z_value[1], z_value[1]],
                [contour_offset, plot_value],
                color=COLORS["sparse"],
                linestyle=":",
                linewidth=0.9,
                alpha=0.35,
            )

    if show_guides:
        ax.plot(
            [0.0, 0.0],
            [0.0, 0.0],
            [contour_offset, x0_plot_value],
            color=COLORS["x0"],
            linestyle=":",
            linewidth=0.9,
            alpha=0.35,
        )

    title_case = "grouped prior" if case == "grouped" else "three-axis prior"
    ax.set_title(
        rf"Unrolled sparse-prior objective on the solution plane ({title_case})",
        pad=14,
    )
    ax.set_xlabel(r"$z_1$")
    ax.set_ylabel(r"$z_2$")
    ax.set_zlabel(r"$P(\mathbf{x}_0 + \mathbf{B}\mathbf{z})$")
    ax.set_xlim(*z1_bounds)
    ax.set_ylim(*z2_bounds)
    z_lower = contour_offset if show_contours else 0.0
    ax.set_zlim(z_lower, float(np.nanmax(data.objective)) * 1.05)
    ax.set_box_aspect(box_aspect)
    ax.view_init(elev=elev, azim=azim)
    ax.legend(loc="upper right", frameon=True, framealpha=0.94)
    return fig, data


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open an unrolled sparse-prior objective viewer."
    )
    parser.add_argument(
        "--case",
        choices=("grouped", "thesis"),
        default=DEFAULT_CASE,
        help="Which 3D sparse-prior example to visualize.",
    )
    parser.add_argument("--z1", nargs=2, type=float, default=DEFAULT_Z1_BOUNDS)
    parser.add_argument("--z2", nargs=2, type=float, default=DEFAULT_Z2_BOUNDS)
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument("--contours", type=int, default=DEFAULT_CONTOURS)
    parser.add_argument("--offset", type=float, default=DEFAULT_CONTOUR_OFFSET)
    parser.add_argument("--surface-alpha", type=float, default=DEFAULT_SURFACE_ALPHA)
    parser.add_argument(
        "--marker-lift",
        type=float,
        default=DEFAULT_MARKER_LIFT,
        help="Draw markers this much above the surface to avoid 3D occlusion.",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="Prior weights. Defaults depend on --case.",
    )
    parser.add_argument("--elev", type=float, default=DEFAULT_ELEV)
    parser.add_argument("--azim", type=float, default=DEFAULT_AZIM)
    parser.add_argument("--box-aspect", nargs=3, type=float, default=DEFAULT_BOX_ASPECT)
    parser.add_argument(
        "--hide-contours",
        action="store_false",
        dest="show_contours",
        help="Hide the projected contour floor.",
    )
    parser.add_argument(
        "--hide-guides",
        action="store_false",
        dest="show_guides",
        help="Hide vertical guide lines from markers to the contour floor.",
    )
    parser.add_argument("--save", type=Path)
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Build and optionally save the figure without opening a window.",
    )
    parser.set_defaults(
        show_contours=DEFAULT_SHOW_CONTOURS,
        show_guides=DEFAULT_SHOW_GUIDES,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    z1_bounds = _bounds(args.z1, "--z1")
    z2_bounds = _bounds(args.z2, "--z2")

    fig, _ = build_unrolled_view(
        case=args.case,
        weights=args.weights,
        z1_bounds=z1_bounds,
        z2_bounds=z2_bounds,
        grid_size=args.grid_size,
        contours=args.contours,
        contour_offset=args.offset,
        surface_alpha=args.surface_alpha,
        marker_lift=args.marker_lift,
        elev=args.elev,
        azim=args.azim,
        box_aspect=tuple(args.box_aspect),
        show_contours=args.show_contours,
        show_guides=args.show_guides,
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
