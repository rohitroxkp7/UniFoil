import os
import numpy as np

coefs = np.loadtxt('input/coefs.txt')    # Shape: (num_samples, num_modes)
modes = np.loadtxt('input/modes.txt')    # Shape: (num_modes, num_points)

# Print the first set of coefficients used for the first airfoil
print("First set of modal coefficients (used for first airfoil):")
print(coefs[0])

# Generate airfoils using modal reconstruction
airfy = np.dot(coefs, modes)

ns = airfy.shape[0]

# Save reconstructed airfoils to file
with open('airfoilys.txt', 'w') as f:
    for ins in range(ns):
        for i in range(airfy.shape[1]):
            f.write('%.15f ' % airfy[ins, i])
        f.write('\n')

# Plotting first 100 airfoils
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Define colors
coloraxis = '#ecf0f1'
colordoe = '#ecf0f1'
coloropt = '#2980b9'
colorlabel = '#2c3e50'
colorinfeasib = '#95a5a6'
colorfeasib = '#95a5a6'
colors = ['#34495e', '#2c3e50', '#c0392b']

xslice = np.loadtxt('input/xslice.txt')

ict = 0
fig = plt.figure(figsize=(10, 5))
gs = gridspec.GridSpec(10, 10)

for i in range(10):
    for j in range(10):
        ax = fig.add_subplot(gs[i, j])
        ax.axis('equal')
        ax.axis('off')
        ax.plot(xslice, airfy[ict, :], lw=0.4, color=colors[0], label='Baseline')
        ict += 1

fig.savefig('test.pdf', dpi=500, bbox_inches="tight")
