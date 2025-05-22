import pyvista as pv

mesh = pv.read("airfoil_surf.cgns")

# Drill down if necessary
while isinstance(mesh, pv.MultiBlock):
    mesh = mesh[0]

# Confirm arrays
print("Available fields:", mesh.cell_data.keys())

# Example: plot Cp (Coefficient of Pressure)
plotter = pv.Plotter()
plotter.add_mesh(mesh, scalars="CoefPressure", cmap="coolwarm", show_edges=False)
plotter.add_axes()
plotter.show()
