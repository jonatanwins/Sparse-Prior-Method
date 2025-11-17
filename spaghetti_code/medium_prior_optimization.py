# plot 3D distribution
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from skimage import measure

# Grid for 3D plot
n_points = 50
x_range = np.linspace(-7, 7, n_points)
y_range = np.linspace(-7, 7, n_points)
z_range = np.linspace(-7, 7, n_points)
X, Y, Z = np.meshgrid(x_range, y_range, z_range)
grid_points = np.stack([X, Y, Z], axis=-1)  # shape (n, n, n, 3)

# 3D Gaussian parameters
variance = 1.0
spread = 0.005
iso_value = 2

D1 = np.array([[variance, 0.0, 0.0], [0.0, spread, 0.0], [0.0, 0.0, spread]])
D2 = np.array([[spread, 0.0, 0.0], [0.0, variance, 0.0], [0.0, 0.0, spread]])
D3 = np.array([[spread, 0.0, 0.0], [0.0, spread, 0.0], [0.0, 0.0, variance]])
D4 = np.array([[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]])
D = [D1, D2, D3, D4]
D = [D1, D2, D3]


# Y(2x1)  = A(2x3) X(3x1)
Y = np.array([[3.0], [6.0]])  # 2x1
A = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])  # 2x3
X0 = np.linalg.pinv(A) @ Y  # initial guess for X


# B is the span of the null space of A
# find the null space of A
U, S, Vt = np.linalg.svd(A)
# compute the rank of A
rank = np.sum(S > 1e-10)
B = Vt[rank:].T  # 3x2


# Quadratic form x^T D x for each grid point
def quad_form(points, D_matrix):
    # This function handles both a single column vector (shape N, 1)
    # and a batch of row vectors (shape ..., N)
    if points.shape[-1] == 1 and points.ndim > 1:
        # It's a single column vector, use the mathematical formula
        return (points.T @ D_matrix @ points).item()
    else:
        # It's a batch of row vectors, use the efficient NumPy way
        return (points @ D_matrix * points).sum(axis=-1)


# Find the sparse solutions by optimizing along the plane defined by X0 and B
# z is a 2D vector
def negative_objective(z: np.ndarray[2]) -> float:
    # numpy broadcasting will make row vectors if z is 1D
    x = X0 + (B @ z.reshape(-1, 1))
    print("X0 + (B @ z):", x)
    print(
        "probability at current solution:",
        sum(np.exp(-quad_form(x.T, D_i)) for D_i in D),
    )

    return -sum(np.exp(-quad_form(x, D_i)) for D_i in D)


# optimize to find sparse solutions
res = optimize.minimize(negative_objective, np.array([1.0, 1.0]), method='BFGS')
print(res.message)
z_opt = res.x
print("Optimal z:", z_opt, "Objective value:", -res.fun)
x_opt = X0 + (B @ z_opt.reshape(-1, 1))
print("B @ z_opt:", B @ z_opt, "X0:", X0)
print("Optimal x:", x_opt)


# Objective function P = sum_i exp(-x^T D_i x) over the grid
P = sum(np.exp(-quad_form(grid_points, D_i)) for D_i in D)

# ----------------
# ----------------
# --- Plotting ---
# ----------------
# ----------------

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')


def plot_isosurface_and_plane(ax, X0, B, P, iso_value):
    # Plot the plane spanned by X0 and the null space of A (columns of B)
    z1_range = np.linspace(-3, 3, 10)
    z2_range = np.linspace(-3, 3, 10)
    Z1, Z2 = np.meshgrid(z1_range, z2_range)

    # Calculate the coordinates of the plane points
    # X_plane = X0 + B * [z1, z2]^T
    X_plane = X0.reshape(-1, 1) + B @ np.array([Z1.ravel(), Z2.ravel()])
    X_surf = X_plane[0, :].reshape(Z1.shape)
    Y_surf = X_plane[1, :].reshape(Z1.shape)
    Z_surf = X_plane[2, :].reshape(Z1.shape)

    ax.plot_surface(X_surf, Y_surf, Z_surf, alpha=0.5, label="Solution Plane")

    # Find and plot the isosurface

    ax.scatter(
        x_opt[0],
        x_opt[1],
        x_opt[2],
        s=150,
        marker="*",
        label="Optimum",
        color="magenta",
        zorder=10,
    )

    try:
        verts, faces, _, _ = measure.marching_cubes(
            P,
            level=iso_value,
            spacing=(
                x_range[1] - x_range[0],
                y_range[1] - y_range[0],
                z_range[1] - z_range[0],
            ),
        )
        # Adjust vertex coordinates to match the grid
        verts[:, 0] += x_range[0]
        verts[:, 1] += y_range[0]
        verts[:, 2] += z_range[0]
        ax.plot_trisurf(
            verts[:, 0], verts[:, 1], faces, verts[:, 2], cmap='viridis', lw=1
        )
    except (RuntimeError, ValueError) as e:
        print(f"Could not generate isosurface for value {iso_value}: {e}")
        print("The surface might not intersect the volume at this level.")

    # Set plot properties
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Isosurface of 3D Gaussian Sum (level={iso_value})', fontsize=16)
    ax.set_xlim(x_range.min(), x_range.max())
    ax.set_ylim(y_range.min(), y_range.max())
    ax.set_zlim(z_range.min(), z_range.max())


def plot_objective_topology(ax, X0, B):
    """
    Plots the objective function as a topological surface over the z1-z2 plane.
    """
    # 1. Create the z1, z2 grid
    z1_range = np.linspace(-4, 4, 100)
    z2_range = np.linspace(-4, 4, 100)
    Z1, Z2 = np.meshgrid(z1_range, z2_range)
    z_grid = np.stack([Z1.ravel(), Z2.ravel()], axis=0)

    # 2. Calculate the objective function P for all points on the z1-z2 grid
    X_plane_rows = (X0.reshape(-1, 1) + B @ z_grid).T
    P_values = sum(np.exp(-quad_form(X_plane_rows, D_i)) for D_i in D)
    P_surf = P_values.reshape(Z1.shape)

    # 3. Plot the topological surface
    ax.plot_surface(Z1, Z2, P_surf, cmap='viridis', alpha=0.8)

    # 4. Run optimizer and plot its path
    path = []

    def callback(xk):
        path.append(np.copy(xk))

    # Run from a non-symmetric starting point to see the path
    z_start = np.array([-3.0, -3.0])
    res = optimize.minimize(
        negative_objective, z_start, method='BFGS', callback=callback
    )
    path = np.array(path)

    # Calculate P values along the path
    path_x = (X0.reshape(-1, 1) + B @ path.T).T
    path_p = sum(np.exp(-quad_form(path_x, D_i)) for D_i in D)

    # Plot the path on the surface
    ax.plot(
        path[:, 0],
        path[:, 1],
        path_p,
        color='red',
        marker='.',
        markersize=4,
        label='Optimizer Path',
    )
    ax.scatter(
        path[0, 0],
        path[0, 1],
        path_p[0],
        color='orange',
        s=100,
        label='Start',
        zorder=10,
    )
    ax.scatter(
        path[-1, 0],
        path[-1, 1],
        path_p[-1],
        color='magenta',
        s=150,
        marker='*',
        label='Optimum',
        zorder=10,
    )

    # 5. Set labels and title
    ax.set_xlabel("z1 (Direction of 1st Null Space Vector)")
    ax.set_ylabel("z2 (Direction of 2nd Null Space Vector)")
    ax.set_zlabel("Objective Function (P)")
    ax.set_title("Topological Map of Objective Function on Solution Plane")
    ax.legend()


# --- Main Execution ---
# plot_isosurface_and_plane(ax, X0, B, P, iso_value=2)
plot_objective_topology(ax, X0, B)
plt.show()
