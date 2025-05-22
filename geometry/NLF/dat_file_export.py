import numpy as np
import os

# Load the data
airfoil_data = np.loadtxt('airfoilys.txt')  # Load the airfoil shapes
xslice = np.loadtxt('input/xslice.txt')    # Load the x-coordinates

# Ensure xslice and airfoil_data are compatible
if airfoil_data.shape[1] != len(xslice):
    print(f"Error: xslice length ({len(xslice)}) does not match airfoil data columns ({airfoil_data.shape[1]}).")
    exit()

# Create the folder to store .dat files
output_folder = 'airfoil_dat_files'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through all airfoils and save each one as a .dat file
# Loop through all airfoils and save each one as a .dat file
for n in range(airfoil_data.shape[0]):
    # Extract the nth airfoil's y-coordinates
    airfoil_y = airfoil_data[n, :]
    
    # Define the filename
    dat_filename = os.path.join(output_folder, f'airfoil_{n+1:03d}.dat')  # e.g., airfoil_001.dat, airfoil_002.dat

    # Prepare x-y coordinate pairs
    coords = list(zip(xslice, airfoil_y))

    # Check if the first and last points are the same
    if not (np.isclose(coords[0][0], coords[-1][0]) and np.isclose(coords[0][1], coords[-1][1])):
        coords.append(coords[0])  # Append the first point to close the airfoil
    
    # Open the .dat file for writing
    with open(dat_filename, 'w') as file:
        for i, (x, y) in enumerate(coords, start=1):
            file.write(f"{i:03d} {x:.8e} {y:.8e}\n")
    
    print(f"Airfoil {n+1} coordinates successfully exported to '{dat_filename}'.")

print(f"All airfoil files have been saved in the '{output_folder}' folder.")

