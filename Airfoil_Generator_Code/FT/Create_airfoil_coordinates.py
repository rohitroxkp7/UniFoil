import numpy as np
import os

# Load basis and x-coordinates
basis = np.loadtxt('input/basis.txt')
xslice = basis[0, :]
modes = basis[1:, :]  # shape (14, npts)

# Create output folder if not exists
os.makedirs("airfoil_dat", exist_ok=True)

def process_airfoils(datafile, start_idx):
    data = np.loadtxt(datafile)
    for i, row in enumerate(data):
        coefs = row[:14]  # first 14 are modal coefficients
        yslice = np.dot(coefs, modes)

        # Combine x and y directly, no reordering or closure
        coords = np.column_stack((xslice, yslice))

        # Write to .dat file as-is
        file_index = start_idx + i + 1
        filename = f"airfoil_dat/airfoil_{file_index}.dat"
        with open(filename, "w") as f:
            for j, (x, y) in enumerate(coords):
                f.write(f"{j+1:03d} {x:.8e} {y:.8e}\n")

    return start_idx + data.shape[0]

# Process training and validation sets
idx_after_train = process_airfoils("input/training.dat", start_idx=0)
_ = process_airfoils("input/validating.dat", start_idx=idx_after_train)
