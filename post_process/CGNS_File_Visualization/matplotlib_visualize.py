import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# ----------------------------
# CONFIG
filename = "surf.cgns"
flag = 1  # 1 = Cp, 2 = Velocity magnitude, 3 = Mach
z_tolerance = 1e-6
grid_res = 1000  # Grid resolution for interpolation

# ----------------------------
# Load CGNS file
mesh = pv.read(filename)



# ----------------------------
# Access blocks by known structure
airfoil  = mesh[0][0][0]  # NSWallAdiabaticBCZone1
farfield = mesh[0][0][0] #mesh[0][1][0]  # FarFieldBCZone2
sym_z0   = mesh[0][0][0]#mesh[0][2][0]  # SymmetryBCZone3

# Extract airfoil coordinates from cell centers
af_coords = airfoil.cell_centers().points

# Keep only z=0 points and drop z dimension
af_2d = af_coords[np.abs(af_coords[:, 2]) < 1e-6][:, :2]  # shape: (n, 2)


# Combine the z=0 region
z0_blocks = [airfoil, farfield, sym_z0]
combined = pv.MultiBlock(z0_blocks).combine()

# ----------------------------
# Choose scalar field
if flag == 1:
    scalar_field = "CoefPressure"#"TurbulentSANuTilde"#"CoefPressure"
    cmap = "coolwarm"#"inferno"#"coolwarm"
elif flag == 2:
    velocity = combined.cell_data["Velocity"]
    velocity_mag = np.linalg.norm(velocity, axis=1)
    combined.cell_data["VelocityMagnitude"] = velocity_mag
    scalar_field = "VelocityMagnitude"
    cmap = "viridis"
elif flag == 3:
    scalar_field = "Mach"
    cmap = "inferno"
else:
    raise ValueError("Invalid flag. Must be 1 (Cp), 2 (Velocity magnitude), or 3 (Mach)")

# ----------------------------
# Get cell centers and scalar values
centers = combined.cell_centers().points
scalar_vals = combined.cell_data[scalar_field]
x, y = centers[:, 0], centers[:, 1]


import matplotlib.tri as tri
from matplotlib.patches import Polygon

# ----------------------------
# Load airfoil from coords.dat (cols 2 and 3 → x, y)
coords = np.loadtxt("coords.dat")
af_2d = coords[:, 1:3]

# Sort points counterclockwise to ensure proper polygon
# Assume airfoil points are ordered in coords.dat from LE -> TE (upper) and TE -> LE (lower)
af_2d_sorted = af_2d  # use as-is



# Triangulate and plot
triang = tri.Triangulation(x, y)
plt.figure(figsize=(10, 4))
contour = plt.tricontourf(triang, scalar_vals, levels=100, cmap=cmap)

# Overlay airfoil as white-filled patch with black edge
airfoil_poly = Polygon(af_2d_sorted, closed=True, facecolor='white', edgecolor='black', linewidth=1.5, zorder=10)

plt.gca().add_patch(airfoil_poly)

# Final styling
plt.gca().set_facecolor("white")
plt.axis("equal")
plt.xlim(-0.1, 1)
plt.ylim(-0.3, 0.3)
plt.xticks([])
plt.yticks([])
plt.axis('off')

#cbar = plt.colorbar(contour)
#cbar.set_label(scalar_field)

plt.tight_layout()
plt.show()
#plt.savefig("cp_transi.png", format="png",dpi=1000, bbox_inches="tight", pad_inches=0)

