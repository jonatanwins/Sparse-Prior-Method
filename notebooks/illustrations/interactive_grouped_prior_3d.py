from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sparse_prior_visuals import (
    COLORS,
    DEFAULT_OUTPUT_DIR,
    SparsePriorExample,
    prior_objective,
    require_pyvista,
    validate_example,
)


def make_grouped_3d_example() -> SparsePriorExample:
    """3D toy problem with a two-component grouped sparse prior.

    The first prior component has high precision on x1 and x2 and low precision
    on x3, so it rewards points near the x3 axis. The second component has low
    precision on x1 and x2 and high precision on x3, so it rewards points near
    the x1-x2 plane.
    """
    x_true = np.array([0.0, 0.0, 3.0])
    A = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    y = np.array([3.0, 6.0])
    x0 = np.linalg.pinv(A) @ y
    B = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
    precisions = (
        np.diag([1.0, 1.0, 0.005]),
        np.diag([0.005, 0.005, 1.0]),
    )
    prior_weights = (1.0, 1.0)
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


def solution_plane_patch(example: SparsePriorExample, plane_reach: float):
    z_corners = np.array(
        [
            [-plane_reach, -plane_reach],
            [plane_reach, -plane_reach],
            [plane_reach, plane_reach],
            [-plane_reach, plane_reach],
        ],
        dtype=float,
    )
    plane_vertices = example.x0 + z_corners @ example.B.T
    plane_faces = np.hstack([[4, 0, 1, 2, 3]])
    return plane_vertices, plane_faces


def build_isosurface(
    example: SparsePriorExample, bounds: float, grid_size: int, iso_value: float
):
    pv = require_pyvista()
    lower, upper = -bounds, bounds
    axis = np.linspace(lower, upper, grid_size)
    X1, X2, X3 = np.meshgrid(axis, axis, axis, indexing="ij")
    values = prior_objective(
        np.stack([X1, X2, X3], axis=-1),
        example.precisions,
        example.prior_weights,
    )

    grid = pv.ImageData()
    grid.dimensions = values.shape
    grid.origin = (lower, lower, lower)
    spacing = (upper - lower) / (grid_size - 1)
    grid.spacing = (spacing, spacing, spacing)
    grid.point_data["P"] = values.ravel(order="F")
    return grid.contour([iso_value], scalars="P")


def open_interactive_view(
    bounds: float,
    grid_size: int,
    iso_value: float,
    plane_reach: float,
    surface_opacity: float,
    plane_opacity: float,
    show_bounds: bool,
    screenshot: Path | None,
) -> None:
    pv = require_pyvista()
    example = make_grouped_3d_example()
    validate_example(example)

    isosurface = build_isosurface(example, bounds, grid_size, iso_value)
    plane_vertices, plane_faces = solution_plane_patch(example, plane_reach)
    solution_plane = pv.PolyData(plane_vertices, plane_faces)

    plotter = pv.Plotter(window_size=(2600, 2200))
    plotter.set_background("white")
    plotter.add_mesh(
        isosurface,
        color="#0072B2",
        opacity=surface_opacity,
        smooth_shading=True,
    )
    plotter.add_mesh(
        solution_plane,
        color="#E69F00",
        opacity=plane_opacity,
        show_edges=True,
        edge_color="#222222",
    )

    points = np.array(list(example.sparse_points.values()), dtype=float)
    plotter.add_points(
        points,
        color=COLORS["true"],
        point_size=17,
        render_points_as_spheres=True,
    )
    plotter.add_points(
        example.x0.reshape(1, 3),
        color=COLORS["x0"],
        point_size=18,
        render_points_as_spheres=True,
    )

    if show_bounds:
        plotter.show_bounds(
            bounds=(-bounds, bounds, -bounds, bounds, -bounds, bounds),
            grid="front",
            all_edges=True,
            xtitle="x1",
            ytitle="x2",
            ztitle="x3",
        )

    plotter.camera_position = [
        (11.0, -13.0, 10.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    ]
    plotter.camera.zoom(0.85)

    if screenshot is not None:
        screenshot.parent.mkdir(parents=True, exist_ok=True)
        plotter.show(screenshot=str(screenshot))
    else:
        plotter.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive PyVista view for a grouped 3D sparse prior."
    )
    parser.add_argument(
        "--bounds", type=float, default=5.0, help="Show axes from -bounds to +bounds."
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=140,
        help="Regular grid resolution for the isosurface.",
    )
    parser.add_argument(
        "--iso-value",
        type=float,
        default=0.9,
        help="Objective level used for the isosurface.",
    )
    parser.add_argument(
        "--plane-reach",
        type=float,
        default=3.0,
        help="Reach of the rectangular solution-plane patch in z-coordinates.",
    )
    parser.add_argument("--surface-opacity", type=float, default=1)
    parser.add_argument("--plane-opacity", type=float, default=0.72)
    parser.add_argument(
        "--show-bounds", action="store_true", help="Show PyVista bounds/grid."
    )
    parser.add_argument(
        "--screenshot",
        type=Path,
        default=None,
        help="Optional path for saving a screenshot when the window closes.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    default_screenshot_dir = DEFAULT_OUTPUT_DIR
    _ = default_screenshot_dir  # Keeps the import visible for easy local edits.
    open_interactive_view(
        bounds=args.bounds,
        grid_size=args.grid_size,
        iso_value=args.iso_value,
        plane_reach=args.plane_reach,
        surface_opacity=args.surface_opacity,
        plane_opacity=args.plane_opacity,
        show_bounds=args.show_bounds,
        screenshot=args.screenshot,
    )
