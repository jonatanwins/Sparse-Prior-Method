import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# zero-mean components
mu = np.array([0.0, 0.0])
covs = [
    np.array([[10.0, 0.0], [0.0, 0.1]]),
    np.array([[0.02, 0.0], [0.0, 0.1]]),
    np.array([[0.1, 0.0], [0.0, 10.0]]),
    np.array([[0.01, 0.0], [0.0, 0.02]]),
]

# weights
w = np.array([10, 0, 10, 0])
# w = np.array([1, 1, 1, 1])
w = w / w.sum()  # remove this line if you want an unnormalized sum

# grid
zoom = 5
xs = np.linspace(-zoom, zoom, 220)
ys = np.linspace(-zoom, zoom, 220)
XX, YY = np.meshgrid(xs, ys)
pos = np.dstack((XX, YY))

# weighted sum of Gaussians
components = [multivariate_normal(mean=mu, cov=C) for C in covs]
Zs = np.stack([rv.pdf(pos) for rv in components], axis=0)  # shape (K, Ny, Nx)
Z = np.tensordot(w, Zs, axes=1)  # shape (Ny, Nx)

# plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(XX, YY, Z, linewidth=0, antialiased=True, cmap="viridis")
ax.set_box_aspect([1, 1, 0.2])
# ax.set_aspect('equal')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('weighted pdf')
plt.show()

# TODO make into callable function from beamforming_with_priors
