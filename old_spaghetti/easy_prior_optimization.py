# Plot the affine line x0 + b*z together with the sum of two "Gaussians" exp(-x^T D_i x)
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

# Parameters from the example
x0 = np.array([1.2, 2.4])
B = np.array([-2.0, 1.0])
D1 = np.array([[5.0, 0.0], [0.0, 0.01]])
D2 = np.array([[0.01, 0.0], [0.0, 5.0]])
D3 = np.array([[0.01, 0.0], [0.0, 0.01]])
D = [D1, D2, D3]

# Grid for contour plot
x1 = np.linspace(-7, 7, 450)
x2 = np.linspace(-7, 7, 450)
X1, X2 = np.meshgrid(x1, x2)
X = np.stack([X1, X2], axis=-1)  # shape (H, W, 2)


# Quadratic forms x^T D x for each grid point
def quad_form(X, D):
    # X: (..., 2)
    # returns: (...,)
    return (X @ D * X).sum(-1)


# compute P = sum_i exp(-x^T D_i x) over the grid
def negative_objective(z):
    x = x0 + B * z
    return -sum(np.exp(-quad_form(x, D_i)) for D_i in D)


P = sum(np.exp(-quad_form(X, D_i)) for D_i in D)

# optimize to find sparse solutions
res = scipy.optimize.minimize(negative_objective, 0.0, method='BFGS')
print(res.message)
z_opt = res.x
x_opt = x0 + B * z_opt
P_opt = sum(np.exp(-quad_form(x_opt, D_i)) for D_i in D)


# Plotting function
def plot_contour_with_line(ax, X1, X2, P, x0):
    z_vals = np.linspace(-5, 5, 400)
    line_pts = x0[None, :] + z_vals[:, None] * B[None, :]

    # plot line
    ax.plot(line_pts[:, 0], line_pts[:, 1], linewidth=2, label=r"$x_0 + B z$", zorder=1)

    # plot contour
    cs = ax.contour(X1, X2, P, levels=20)
    ax.clabel(cs, inline=True, fontsize=8)

    # plot x_0
    ax.scatter(
        [x0[0]], [x0[1]], s=40, marker="h", label=r"$x_0$", color="orange", zorder=2
    )

    # optimize to find sparse solutions
    res = scipy.optimize.minimize(negative_objective, 0.0, method='BFGS')
    print(res.message)
    z_opt = res.x
    x_opt = x0 + B * z_opt
    ax.scatter(
        [x_opt[0]], [x_opt[1]], s=50, marker="o", label="optimum", color="red", zorder=3
    )

    ax.axis("equal")
    ax.set_xlim(-2, 6)
    ax.set_ylim(-2, 6)
    ax.set_xticks(np.arange(-2, 6, 0.5))
    ax.set_xticklabels(np.arange(-2, 6, 0.5), fontsize=10)
    ax.set_yticks(np.arange(-2, 6, 0.5))
    ax.set_yticklabels(np.arange(-2, 6, 0.5), fontsize=10)
    ax.grid(True)
    ax.set_xlabel(r"$x_1$", fontsize=12)
    ax.set_ylabel(r"$x_2$", fontsize=12)
    ax.set_title(
        r"Sum of two bivariate Gaussians and solution line to $Y=AX$",
        fontsize=24,
    )
    ax.legend(fontsize=16)
    return ax


def plot_in_contour_and_surface(ax, point, P_point, offset, color="purple"):
    ax.scatter(
        [point[0]],
        [point[1]],
        [offset],
        s=40,
        marker="h",
        label=r"$x_0$",
        color=color,
    )
    ax.scatter(
        [point[0]],
        [point[1]],
        [P_point],
        s=40,
        marker="o",
        label="optimum",
        color=color,
    )


def plot_3D_objective_with_line(fig, X1, X2, P, x0, x_opt, P_opt, offset=-2.5):
    ax = fig.add_subplot(111, projection='3d')

    # plot plane in Z along the line
    z_vals = np.linspace(-3, 3, 400)
    line_pts = x0[None, :] + z_vals[:, None] * B[None, :]

    # plot plane with contour in the X1X2 plane
    ax.plot_surface(
        X1, X2, P, linewidth=0, antialiased=True, cmap="viridis", zorder=1, alpha=0.5
    )
    ax.contour(X1, X2, P, zdir='z', offset=offset, levels=10)

    # Plot the line together with the contours
    ax.plot(
        line_pts[:, 0],
        line_pts[:, 1],
        offset,  # z=-0.5 plane to match contour
        linewidth=2,
        color="red",
        label=r"$x_0 + B z$",
        zorder=2,
    )

    # plot intersection between plane and surface
    P_line = sum(np.exp(-quad_form(line_pts, D_i)) for D_i in D)
    ax.plot(
        line_pts[:, 0],
        line_pts[:, 1],
        P_line,
        linewidth=2,
        color="red",
        label="intersection",
        zorder=5,
    )

    # plot pseudoinverse x_0
    plot_in_contour_and_surface(
        ax, x0, sum(np.exp(-quad_form(x0, D_i)) for D_i in D), offset, color='orange'
    )

    plot_in_contour_and_surface(ax, x_opt, P_opt, offset, color='purple')


fig, ax = plt.subplots(figsize=(6, 6))


# plot objective function onto the line of solutions
# x0 is (2, ) B is (2, ) z is linspace (-5, 5, 400)
def plot_objective_on_line(ax, x0, B, z):
    line_pts = x0[None, :] + z[:, None] * B[None, :]
    P_line = sum(np.exp(-quad_form(line_pts, D_i)) for D_i in D)
    ax.plot(
        z,
        P_line,
        linewidth=2,
        color="red",
        label=r"$P(x_0 + zB)$",
    )
    ax.set_xlabel("z (parameter along the line)")
    ax.set_ylabel("Objective Function P")
    ax.set_title("Objective Function along the Solution Line")
    ax.grid(True)
    ax.legend()


# plot_contour_with_line(ax, X1, X2, P, x0)
plot_3D_objective_with_line(fig, X1, X2, P, x0, x_opt, P_opt, offset=-2.5)
ax.set_box_aspect(111)
# plot_objective_on_line(ax, x0, B, np.linspace(-5, 5, 400))

plt.show()
