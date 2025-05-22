import numpy as np

# Read only the first row from the file
with open('training.dat', 'r') as f:
    first_line = f.readline()

# Convert the row into a numpy array of floats
data = np.fromstring(first_line, sep=' ')

# Segregate the data
x = data[:14]             # 14 mode coefficients
mach = data[14]           # Mach number
aoa = data[15]            # Angle of attack
cl = data[16]             # Lift coefficient
cd = data[17]             # Drag coefficient
cm = data[18]             # Moment coefficient
dcl_dx = data[19]         # Sensitivity of cl
dcd_dx = data[20]         # Sensitivity of cd
dcm_dx = data[21]         # Sensitivity of cm

# Print the values
print("x (14 modes):", x)
print("Mach:", mach)
print("AoA:", aoa)
print("CL:", cl)
print("CD:", cd)
print("CM:", cm)
print("dCL/dx:", dcl_dx)
print("dCD/dx:", dcd_dx)
print("dCM/dx:", dcm_dx)
