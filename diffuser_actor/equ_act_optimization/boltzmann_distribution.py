import numpy as np
import matplotlib.pyplot as plt

# Define constants
temperature = 0.01

# Define the distance range
r = np.linspace(-0.5, 0.5, 1000)  # Distance in arbitrary units

# Boltzmann distribution as a function of distance
def boltzmann_distribution(r, temperature):
    return np.exp(-np.abs(r) / temperature)

# Compute the Boltzmann distribution
P_r = boltzmann_distribution(r, temperature)

# Plot the Boltzmann distribution
plt.figure(figsize=(5, 5))
plt.plot(r, P_r, label=f'PDF')
plt.xlabel('Distance $||\hat{a}_{trans} - a_{trans}||$')
plt.ylabel(r'$p(a_{trans})$')
plt.title('Boltzmann Distribution with Respect to Distance')
plt.legend()
plt.grid(True)
plt.show()
