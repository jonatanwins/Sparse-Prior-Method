import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_elements = 8  # Number of array elements
d = 0.5  # Element spacing in wavelengths
theta = np.linspace(0, 2 * np.pi, 360)  # Angles from 0 to 360 degrees

# Element positions
x = np.arange(num_elements) * d
y = np.zeros_like(x)

# Initialize array factor
AF = np.zeros_like(theta, dtype=complex)

# Calculate array factor
for i in range(len(theta)):
    # Progressive phase shift
    phase = 2 * np.pi * d * np.cos(theta[i])
    # Sum contributions from all elements
    AF[i] = np.sum(np.exp(1j * np.arange(num_elements) * phase))

# Normalize array factor
AF = np.abs(AF) / np.max(np.abs(AF))

# Convert to dB
AF_db = 20 * np.log10(AF)

# Create single polar plot
plt.figure(figsize=(10, 10))
ax = plt.subplot(111, projection="polar")

# Plot radiation pattern
ax.plot(theta, AF)

# Convert microphone positions to polar coordinates for overlay
# Scale the positions to match the radiation pattern
scale_factor = 0.02  # Adjust this to change the size of the array visualization
r = np.ones_like(x) * scale_factor
phi = np.arctan2(y, x)
ax.scatter(phi, r, c="blue", marker="o", s=100, label="Array Elements")

ax.set_title("2D Phased Array Radiation Pattern with Array Elements")
ax.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.grid(True)
plt.legend()
plt.show()
