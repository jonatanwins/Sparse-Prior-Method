from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import warnings

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "illustrations"
DEFAULT_2D_PRIOR_WEIGHTS = (1.0, 1.0, 0.35)

COLORS = {
    "objective": "#0072B2",
    "solution": "#FFC404",
    "x0": "#FF0000",
    "sparse": "#DD00FF",
    "true": "#009E73",
    "alt": "#CC79A7",
    "plane": "#FF6565",
    "text": "#222222",
    "grid": "#D8D8D8",
}


@dataclass(frozen=True)
class SparsePriorExample:
    x_true: np.ndarray
    A: np.ndarray
    y: np.ndarray
    x0: np.ndarray
    B: np.ndarray
    precisions: tuple[np.ndarray, ...]
    prior_weights: tuple[float, ...]
    sparse_points: dict[str, np.ndarray]
    sparse_z: dict[str, np.ndarray | float]


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 600,
            "font.family": "serif",
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.8,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.55,
            "grid.alpha": 0.8,
            "lines.linewidth": 1.8,
        }
    )


def quad_form(points: np.ndarray, precision: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    precision = np.asarray(precision, dtype=float)
    return np.einsum("...i,ij,...j->...", points, precision, points)


def prior_objective(
    points: np.ndarray,
    precisions: Iterable[np.ndarray],
    weights: Iterable[float] | None = None,
) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    precisions = tuple(precisions)
    if weights is None:
        weights = np.ones(len(precisions), dtype=float)
    weights = np.asarray(tuple(weights), dtype=float)
    if weights.shape != (len(precisions),):
        raise ValueError(
            f"Expected one prior weight per precision matrix, got {weights.size} "
            f"weights for {len(precisions)} precision matrices."
        )

    values = np.zeros(points.shape[:-1], dtype=float)
    for weight, precision in zip(weights, precisions):
        values += weight * np.exp(-quad_form(points, precision))
    return values


def make_2d_example(
    prior_weights: Iterable[float] = DEFAULT_2D_PRIOR_WEIGHTS,
) -> SparsePriorExample:
    x_true = np.array([0.0, 3.0])
    A = np.array([[1.0, 2.0]])
    y = np.array([6.0])
    x0 = np.linalg.pinv(A) @ y
    B = np.array([[-2.0], [1.0]])
    precisions = (
        np.diag([5.0, 0.01]),
        np.diag([0.01, 5.0]),
        np.diag([0.01, 0.01]),
    )
    prior_weights = tuple(float(weight) for weight in prior_weights)
    sparse_points = {
        r"$(0,3)^\top$": np.array([0.0, 3.0]),
        r"$(6,0)^\top$": np.array([6.0, 0.0]),
    }
    sparse_z = {
        r"$(0,3)^\top$": 0.6,
        r"$(6,0)^\top$": -2.4,
    }
    return SparsePriorExample(
        x_true, A, y, x0, B, precisions, prior_weights, sparse_points, sparse_z
    )


def make_3d_example(
    prior_weights: Iterable[float] = (1.0, 1.0, 1.0),
) -> SparsePriorExample:
    x_true = np.array([0.0, 3.0, 0.0])
    A = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    y = np.array([3.0, 6.0])
    x0 = np.linalg.pinv(A) @ y
    B = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
    precisions = (
        np.diag([1.0, 0.005, 0.005]),
        np.diag([0.005, 1.0, 0.005]),
        np.diag([0.005, 0.005, 1.0]),
    )
    prior_weights = tuple(float(weight) for weight in prior_weights)
    sparse_points = {
        r"$(3,0,0)^\top$": np.array([3.0, 0.0, 0.0]),
        r"$(0,3,0)^\top$": np.array([0.0, 3.0, 0.0]),
        r"$(0,0,3)^\top$": np.array([0.0, 0.0, 3.0]),
    }
    sparse_z = {
        r"$(3,0,0)^\top$": np.array([-1.0, -1.0]),
        r"$(0,3,0)^\top$": np.array([2.0, -1.0]),
        r"$(0,0,3)^\top$": np.array([-1.0, 2.0]),
    }
    return SparsePriorExample(
        x_true, A, y, x0, B, precisions, prior_weights, sparse_points, sparse_z
    )


def validate_example(example: SparsePriorExample) -> None:
    assert np.allclose(example.A @ example.x_true, example.y)
    assert np.allclose(example.A @ example.x0, example.y)
    assert np.allclose(example.A @ example.B, 0.0)

    for point in example.sparse_points.values():
        assert np.allclose(example.A @ point, example.y)

    objective_at_x0 = prior_objective(
        example.x0, example.precisions, example.prior_weights
    )
    for point in example.sparse_points.values():
        assert (
            prior_objective(point, example.precisions, example.prior_weights)
            > objective_at_x0
        )


def ensure_output_dir(output_dir: Path | str = DEFAULT_OUTPUT_DIR) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_matplotlib_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    output_dir = ensure_output_dir(output_dir)
    paths = []
    for suffix in (".png", ".pdf", ".svg"):
        path = output_dir / f"{stem}{suffix}"
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.04}
        if suffix == ".png":
            kwargs["dpi"] = 600
        fig.savefig(path, **kwargs)
        paths.append(path)
    return paths


def _axis_values(
    spec: tuple[float, float, int] | np.ndarray | list[float],
) -> np.ndarray:
    if isinstance(spec, tuple) and len(spec) == 3:
        start, stop, count = spec
        return np.linspace(float(start), float(stop), int(count))
    return np.asarray(spec, dtype=float)


def plot_2d_contour(
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    x1=(-0.6, 6.4, 560),
    x2=(-0.6, 3.8, 460),
    contour_scale: str = "line",
    prior_weights: Iterable[float] | None = None,
    contour_levels: Iterable[float] | None = None,
) -> tuple[plt.Figure, list[Path]]:
    configure_matplotlib()
    example = make_2d_example(
        prior_weights=(
            prior_weights if prior_weights is not None else DEFAULT_2D_PRIOR_WEIGHTS
        )
    )
    validate_example(example)
    output_dir = ensure_output_dir(output_dir)

    x1_values = _axis_values(x1)
    x2_values = _axis_values(x2)
    X1, X2 = np.meshgrid(x1_values, x2_values)
    points = np.stack([X1, X2], axis=-1)
    objective = prior_objective(points, example.precisions, example.prior_weights)

    fig, ax = plt.subplots(figsize=(6.2, 5.0), constrained_layout=True)
    max_objective = float(np.nanmax(objective))
    if contour_levels is not None:
        contour_levels = np.asarray(tuple(contour_levels), dtype=float)
        contour_levels = np.unique(np.sort(contour_levels))
        contour_levels = contour_levels[contour_levels < max_objective]
    elif contour_scale == "line":
        sparse_levels = [
            float(prior_objective(point, example.precisions, example.prior_weights))
            for point in example.sparse_points.values()
        ]
        contour_levels = np.array([0.02, *sparse_levels, 0.55, 1.6])
        contour_levels = np.unique(np.round(contour_levels, decimals=10))
        contour_levels = contour_levels[contour_levels < max_objective]
    elif contour_scale == "log":
        contour_levels = np.geomspace(
            max(max_objective * 1e-3, 1e-6), max_objective * 0.9, 7
        )
    elif contour_scale == "linear":
        contour_levels = np.linspace(max_objective * 0.08, max_objective * 0.9, 7)
    else:
        raise ValueError("contour_scale must be 'line', 'log', or 'linear'")
    if contour_levels.size == 0:
        raise ValueError("No contour levels remain below the maximum objective value.")
    cs = ax.contour(
        X1,
        X2,
        objective,
        levels=contour_levels,
        cmap="viridis",
        linewidths=0.8,
    )
    ax.clabel(cs, levels=contour_levels[::2], inline=True, fontsize=7, fmt="%.2g")

    z_candidates = []
    for dim, values in enumerate((x1_values, x2_values)):
        if not np.isclose(example.B[dim, 0], 0.0):
            z_candidates.extend(
                [
                    (float(values.min()) - example.x0[dim]) / example.B[dim, 0],
                    (float(values.max()) - example.x0[dim]) / example.B[dim, 0],
                ]
            )
    z_line = np.linspace(min(z_candidates), max(z_candidates), 420)
    line_points = example.x0[None, :] + z_line[:, None] * example.B[:, 0][None, :]
    ax.plot(
        line_points[:, 0],
        line_points[:, 1],
        color=COLORS["solution"],
        label=r"$\mathbf{x}_0 + \mathbf{B}\mathbf{z}$",
        zorder=4,
    )

    ax.scatter(
        example.x0[0],
        example.x0[1],
        s=24,
        marker="o",
        color=COLORS["x0"],
        edgecolor=COLORS["text"],
        linewidth=0.35,
        label=r"$\mathbf{x}_0$",
        zorder=6,
    )

    marker_styles = [
        ("sparse", "o", "sparse solutions"),
        ("sparse", "o", "_nolegend_"),
    ]
    for (label, point), (color_key, marker, legend_label) in zip(
        example.sparse_points.items(), marker_styles
    ):
        ax.scatter(
            point[0],
            point[1],
            s=28,
            marker=marker,
            color=COLORS[color_key],
            edgecolor=COLORS["text"],
            linewidth=0.35,
            label=legend_label,
            zorder=7,
        )

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title(
        r"Sum of two bivariate Gaussians with solution line for $\mathbf{y} = \mathbf{A}\mathbf{x}$",
        fontsize=12,
        pad=8,
    )
    ax.set_xlim(float(x1_values.min()), float(x1_values.max()))
    ax.set_ylim(float(x2_values.min()), float(x2_values.max()))
    ax.set_aspect("equal", adjustable="box")
    ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.94,
        fontsize=8,
        borderpad=0.35,
        labelspacing=0.35,
        handlelength=1.7,
        handletextpad=0.55,
    )

    paths = save_matplotlib_figure(fig, output_dir, "sparse_prior_2d_contour")
    return fig, paths


def plot_2d_profile(
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    prior_weights: Iterable[float] | None = None,
) -> tuple[plt.Figure, list[Path]]:
    configure_matplotlib()
    example = make_2d_example(
        prior_weights=(
            prior_weights if prior_weights is not None else DEFAULT_2D_PRIOR_WEIGHTS
        )
    )
    validate_example(example)
    output_dir = ensure_output_dir(output_dir)

    z = np.linspace(-3.0, 1.15, 700)
    line_points = example.x0[None, :] + z[:, None] * example.B[:, 0][None, :]
    objective = prior_objective(line_points, example.precisions, example.prior_weights)

    fig, ax = plt.subplots(figsize=(5.2, 2.75), constrained_layout=True)
    ax.plot(z, objective, color=COLORS["objective"])
    ax.fill_between(
        z, objective, 0.0, color=COLORS["objective"], alpha=0.12, linewidth=0
    )

    markers = [
        (0.0, r"$z=0$", COLORS["x0"], "o"),
        (example.sparse_z[r"$(0,3)^\top$"], r"$z=0.6$", COLORS["true"], "D"),
        (example.sparse_z[r"$(6,0)^\top$"], r"$z=-2.4$", COLORS["alt"], "s"),
    ]
    for z_value, label, color, marker in markers:
        value = prior_objective(
            example.x0 + example.B[:, 0] * z_value,
            example.precisions,
            example.prior_weights,
        )
        ax.scatter(
            z_value,
            value,
            s=64,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
        )
        ax.axvline(z_value, color=color, linewidth=0.9, linestyle="--", alpha=0.7)
        ax.annotate(label, (z_value, value), xytext=(6, 8), textcoords="offset points")

    ax.set_xlabel(r"$z$ in $x_0 + bz$")
    ax.set_ylabel(r"$P(x_0 + bz)$")
    ax.set_xlim(z.min(), z.max())
    ax.set_ylim(bottom=0)
    paths = save_matplotlib_figure(fig, output_dir, "sparse_prior_2d_profile")
    return fig, paths


def plot_3d_unrolled(
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> tuple[plt.Figure, list[Path]]:
    configure_matplotlib()
    example = make_3d_example()
    validate_example(example)
    output_dir = ensure_output_dir(output_dir)

    z1 = np.linspace(-1.8, 2.8, 520)
    z2 = np.linspace(-1.8, 2.8, 520)
    Z1, Z2 = np.meshgrid(z1, z2)
    z_grid = np.stack([Z1, Z2], axis=-1)
    plane_points = example.x0 + np.einsum("ij,...j->...i", example.B, z_grid)
    objective = prior_objective(plane_points, example.precisions, example.prior_weights)

    fig, ax = plt.subplots(figsize=(5.1, 4.25), constrained_layout=True)
    levels = np.linspace(0.65, 1.95, 18)
    cf = ax.contourf(Z1, Z2, objective, levels=levels, cmap="viridis", extend="both")
    ax.contour(
        Z1,
        Z2,
        objective,
        levels=[1.0, 1.25, 1.5, 1.75],
        colors="white",
        linewidths=0.65,
        alpha=0.85,
    )

    sparse_z = np.array(list(example.sparse_z.values()))
    ax.plot(
        [sparse_z[0, 0], sparse_z[1, 0], sparse_z[2, 0], sparse_z[0, 0]],
        [sparse_z[0, 1], sparse_z[1, 1], sparse_z[2, 1], sparse_z[0, 1]],
        color="#222222",
        linewidth=1.0,
        alpha=0.75,
        label=r"$x_1+x_2+x_3=3$ simplex",
    )

    for label, z_value in example.sparse_z.items():
        point = example.x0 + example.B @ z_value
        ax.scatter(
            z_value[0],
            z_value[1],
            s=76,
            marker="D" if np.allclose(point, example.x_true) else "s",
            color=(
                COLORS["true"] if np.allclose(point, example.x_true) else COLORS["alt"]
            ),
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
        )
        ax.annotate(label, z_value, xytext=(7, 7), textcoords="offset points")

    ax.scatter(
        0.0,
        0.0,
        s=62,
        marker="o",
        color=COLORS["x0"],
        edgecolor="white",
        linewidth=0.8,
        label=r"$x_0$",
        zorder=6,
    )
    ax.annotate(r"$x_0$", (0.0, 0.0), xytext=(8, 8), textcoords="offset points")
    ax.set_xlabel(r"$z_1$ along $(-1,1,0)^\top$")
    ax.set_ylabel(r"$z_2$ along $(-1,0,1)^\top$")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right", frameon=True, framealpha=0.94)
    colorbar = fig.colorbar(cf, ax=ax, pad=0.02, shrink=0.92)
    colorbar.set_label(r"$P(x_0 + Bz)$")

    paths = save_matplotlib_figure(fig, output_dir, "sparse_prior_3d_unrolled")
    return fig, paths


def require_pyvista():
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError(
            "PyVista is required for the isosurface figure. Install it with: python -m pip install pyvista"
        ) from exc
    return pv


def render_3d_isosurface(
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    iso_value: float = 1.5,
    grid_bounds: tuple[float, float] = (-5, 5),
    grid_size: int = 88,
) -> list[Path]:
    example = make_3d_example()
    validate_example(example)
    output_dir = ensure_output_dir(output_dir)
    pv = require_pyvista()
    pv.OFF_SCREEN = True
    pv.set_plot_theme("document")

    lower, upper = grid_bounds
    axis = np.linspace(lower, upper, grid_size)
    X1, X2, X3 = np.meshgrid(axis, axis, axis, indexing="ij")
    points = np.stack([X1, X2, X3], axis=-1)
    objective = prior_objective(points, example.precisions, example.prior_weights)

    grid = pv.ImageData()
    grid.dimensions = objective.shape
    grid.origin = (lower, lower, lower)
    spacing = (upper - lower) / (grid_size - 1)
    grid.spacing = (spacing, spacing, spacing)
    grid.point_data["P"] = objective.ravel(order="F")
    isosurface = grid.contour([iso_value], scalars="P")
    if isosurface.n_points == 0:
        raise ValueError(f"No isosurface found for iso_value={iso_value}.")

    simplex_vertices = np.array(
        [
            example.sparse_points[r"$(3,0,0)^\top$"],
            example.sparse_points[r"$(0,3,0)^\top$"],
            example.sparse_points[r"$(0,0,3)^\top$"],
        ],
        dtype=float,
    )
    simplex_faces = np.hstack([[3, 0, 1, 2]])
    simplex = pv.PolyData(simplex_vertices, simplex_faces)
    simplex_edges = simplex.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False,
    )

    if not pv.system_supports_plotting():
        warnings.warn(
            "PyVista extracted the isosurface, but this environment cannot create "
            "a PyVista render window. Falling back to a Matplotlib renderer."
        )
        return _render_pyvista_mesh_with_matplotlib(
            isosurface=isosurface,
            simplex_vertices=simplex_vertices,
            example=example,
            output_dir=output_dir,
            iso_value=iso_value,
            bounds=(lower, upper),
        )

    plotter = pv.Plotter(off_screen=True, window_size=(2400, 2000))
    plotter.set_background("white")
    plotter.add_mesh(
        isosurface,
        color=COLORS["objective"],
        opacity=0.34,
        smooth_shading=True,
        specular=0.18,
        label=rf"$P(x)={iso_value}$",
    )
    plotter.add_mesh(simplex, color=COLORS["plane"], opacity=0.28, show_edges=False)
    plotter.add_mesh(simplex_edges, color="#222222", line_width=5)

    vertex_points = np.vstack([simplex_vertices, example.x0])
    vertex_colors = [COLORS["alt"], COLORS["true"], COLORS["alt"], COLORS["x0"]]
    for point, color in zip(vertex_points, vertex_colors):
        plotter.add_mesh(
            pv.Sphere(
                radius=0.07, center=point, theta_resolution=32, phi_resolution=16
            ),
            color=color,
        )

    label_positions = np.array(
        [
            [3.18, 0.10, 0.10],
            [0.10, 3.18, 0.10],
            [0.10, 0.10, 3.18],
            [1.12, 1.12, 1.25],
        ]
    )
    labels = [r"(3,0,0)", r"(0,3,0)", r"(0,0,3)", r"x0"]
    plotter.add_point_labels(
        label_positions,
        labels,
        font_size=30,
        text_color=COLORS["text"],
        shape_opacity=0.0,
        point_size=0,
        always_visible=True,
    )

    plotter.add_axes(
        xlabel="x1",
        ylabel="x2",
        zlabel="x3",
        line_width=4,
        labels_off=False,
    )
    plotter.show_bounds(
        bounds=(lower, upper, lower, upper, lower, upper),
        location="outer",
        grid="front",
        all_edges=True,
        font_size=16,
        xtitle="x1",
        ytitle="x2",
        ztitle="x3",
    )
    plotter.add_text(
        rf"Prior isosurface $P(x)={iso_value}$ and solution plane $x_1+x_2+x_3=3$",
        position="upper_left",
        font_size=18,
        color=COLORS["text"],
    )
    plotter.camera_position = [
        (5.4, -6.7, 4.9),
        (1.15, 1.15, 1.12),
        (0.0, 0.0, 1.0),
    ]
    plotter.camera.zoom(1.08)

    png_path = output_dir / "sparse_prior_3d_isosurface.png"
    plotter.show(screenshot=str(png_path), auto_close=False)
    paths = [png_path]

    svg_path = output_dir / "sparse_prior_3d_isosurface.svg"
    try:
        plotter.save_graphic(str(svg_path))
        paths.append(svg_path)
    except Exception as exc:  # VTK vector export is backend-dependent.
        warnings.warn(f"Could not export PyVista SVG: {exc}")
    finally:
        plotter.close()

    return paths


def _triangular_faces(polydata) -> tuple[np.ndarray, np.ndarray]:
    mesh = polydata.triangulate()
    faces = mesh.faces.reshape(-1, 4)
    return mesh.points, faces[:, 1:4]


def _render_pyvista_mesh_with_matplotlib(
    isosurface,
    simplex_vertices: np.ndarray,
    example: SparsePriorExample,
    output_dir: Path,
    iso_value: float,
    bounds: tuple[float, float],
) -> list[Path]:
    configure_matplotlib()
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    vertices, faces = _triangular_faces(isosurface)
    if faces.shape[0] > 28_000:
        stride = int(np.ceil(faces.shape[0] / 28_000))
        faces = faces[::stride]

    lower, upper = bounds
    fig = plt.figure(figsize=(5.6, 4.8), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    surface_triangles = vertices[faces]
    surface = Poly3DCollection(
        surface_triangles,
        facecolor=COLORS["objective"],
        edgecolor="none",
        alpha=0.23,
        rasterized=True,
    )
    ax.add_collection3d(surface)

    plane = Poly3DCollection(
        [simplex_vertices],
        facecolor=COLORS["plane"],
        edgecolor="#222222",
        linewidth=1.0,
        alpha=0.34,
    )
    ax.add_collection3d(plane)

    closed_simplex = np.vstack([simplex_vertices, simplex_vertices[0]])
    ax.plot(
        closed_simplex[:, 0],
        closed_simplex[:, 1],
        closed_simplex[:, 2],
        color="#222222",
        linewidth=1.6,
    )

    vertex_points = np.vstack([simplex_vertices, example.x0])
    vertex_colors = [COLORS["alt"], COLORS["true"], COLORS["alt"], COLORS["x0"]]
    ax.scatter(
        vertex_points[:, 0],
        vertex_points[:, 1],
        vertex_points[:, 2],
        s=[42, 48, 42, 42],
        c=vertex_colors,
        depthshade=False,
        edgecolors="white",
        linewidths=0.6,
    )

    labels = [
        (r"$(3,0,0)^\top$", simplex_vertices[0] + np.array([0.08, 0.05, 0.05])),
        (r"$(0,3,0)^\top$", simplex_vertices[1] + np.array([0.05, 0.08, 0.05])),
        (r"$(0,0,3)^\top$", simplex_vertices[2] + np.array([0.05, 0.05, 0.08])),
        (r"$x_0$", example.x0 + np.array([0.07, 0.07, 0.16])),
    ]
    for label, point in labels:
        ax.text(point[0], point[1], point[2], label, color=COLORS["text"], fontsize=9)

    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_zlim(lower, upper)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_zlabel(r"$x_3$")
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=25, azim=-52)
    ax.grid(True)
    ax.text2D(
        0.03,
        0.96,
        rf"Prior isosurface $P(x)={iso_value}$ and solution plane",
        transform=ax.transAxes,
        fontsize=10,
        color=COLORS["text"],
    )

    paths = save_matplotlib_figure(fig, output_dir, "sparse_prior_3d_isosurface")
    plt.close(fig)
    return paths


def generate_all_figures(
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> dict[str, list[Path]]:
    output_dir = ensure_output_dir(output_dir)
    figures: dict[str, list[Path]] = {}
    fig, paths = plot_2d_contour(output_dir)
    plt.close(fig)
    figures["2d_contour"] = paths
    fig, paths = plot_2d_profile(output_dir)
    plt.close(fig)
    figures["2d_profile"] = paths
    figures["3d_isosurface"] = render_3d_isosurface(output_dir)
    fig, paths = plot_3d_unrolled(output_dir)
    plt.close(fig)
    figures["3d_unrolled"] = paths
    return figures


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interactive-3d",
        action="store_true",
        help="Open an interactive PyVista window for rotating the isosurface.",
    )
    args = parser.parse_args()

    if args.interactive_3d:
        example = make_3d_example()
        validate_example(example)
        pv = require_pyvista()
        pv.OFF_SCREEN = False

        lower, upper = -5, 5
        grid_size = 140
        iso_value = 1.8
        axis = np.linspace(lower, upper, grid_size)
        X1, X2, X3 = np.meshgrid(axis, axis, axis, indexing="ij")
        objective = prior_objective(
            np.stack([X1, X2, X3], axis=-1),
            example.precisions,
            example.prior_weights,
        )

        grid = pv.ImageData()
        grid.dimensions = objective.shape
        grid.origin = (lower, lower, lower)
        grid.spacing = tuple([(upper - lower) / (grid_size - 1)] * 3)
        grid.point_data["P"] = objective.ravel(order="F")

        isosurface = grid.contour([iso_value], scalars="P")
        # simplex_vertices = np.array(list(example.sparse_points.values()), dtype=float)
        # simplex = pv.PolyData(simplex_vertices, np.hstack([[3, 0, 1, 2]]))
        plane_reach = 2.5  # trial-and-error knob: larger means wider plane

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
        solution_plane = pv.PolyData(plane_vertices, plane_faces)

        plotter = pv.Plotter()
        plotter.add_mesh(
            isosurface,
            color="#00B277",  # blue: prior isosurface
            opacity=1,
            smooth_shading=False,
        )

        plotter.add_mesh(
            solution_plane,
            color=COLORS["plane"],  # red: solution plane
            opacity=0.6,
            show_edges=True,
            edge_color="#222222",
        )

        sparse_vertices = np.array(list(example.sparse_points.values()), dtype=float)

        plotter.add_points(
            sparse_vertices,
            color=COLORS["true"],
            point_size=25,
            render_points_as_spheres=True,
        )

        # plotter.add_mesh(
        #     simplex,
        #     color="#E69F00",  # orange: solution plane
        #     opacity=1,
        #     show_edges=True,
        #     edge_color="#222222",
        # )

        # plotter.add_points(
        #     simplex_vertices,
        #     color=COLORS["true"],
        #     point_size=16,
        #     render_points_as_spheres=True,
        # )
        plotter.add_points(
            example.x0.reshape(1, 3),
            color=COLORS["x0"],
            point_size=25,
            render_points_as_spheres=True,
        )
        # plotter.show_bounds(
        #     grid="front", all_edges=True, xtitle="x1", ytitle="x2", ztitle="x3"
        # )
        plotter.add_axes()

        # camera
        plotter.camera_position = [
            (11.0, -13.0, 10.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        ]
        # plotter.camera.zoom(0.85)
        plotter.camera.zoom(1)  # < 1 zooms out, > 1 zooms in

        plotter.show()
    else:
        generated = generate_all_figures()
        for name, paths in generated.items():
            print(f"{name}:")
            for path in paths:
                print(f"  {path}")

"""
Get 3D view with: python3 notebooks/illustrations/sparse_prior_visuals.py --interactive-3d
"""
