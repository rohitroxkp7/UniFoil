import numpy as np
import matplotlib.pyplot as plt

# Function to load .dat file containing x, y coordinates
def load_coordinates(filename):
    """
    Load a .dat file containing x, y coordinates.

    Parameters:
        filename (str): Path to the .dat file.

    Returns:
        tuple: Two numpy arrays, x and y, containing the coordinates.
    """
    data = np.loadtxt(filename)
    x = data[:, 1]
    y = data[:, 2]
    return x, y

# Function to plot x, y coordinates
def plot_coordinates(x, y):
    """
    Plot x, y coordinates.

    Parameters:
        x (array): Array of x coordinates.
        y (array): Array of y coordinates.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, marker='o', linestyle='-', label="Coordinates")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Plot of X vs Y Coordinates")
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
file_path = "airfoil_5th.dat"  # Replace with your .dat file path
x, y = load_coordinates(file_path)

# Plot the coordinates
plot_coordinates(x, y)
