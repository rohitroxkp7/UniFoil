import pyvista as pv

def explore_block(block, prefix="Root", level=0):
    indent = "  " * level

    if isinstance(block, pv.MultiBlock):
        print(f"{indent}📦 {prefix} — MultiBlock with {len(block)} sub-blocks")
        for i, sub_block in enumerate(block):
            name = block.get_block_name(i)
            label = f"{prefix}[{i}]"
            if name:
                label += f" (name: {name})"
            explore_block(sub_block, label, level + 1)
    elif block is None:
        print(f"{indent}❌ {prefix} — Empty block")
    else:
        print(f"{indent}✅ {prefix} — {type(block).__name__}")
        print(f"{indent}   ↳ n_points: {block.n_points}")
        print(f"{indent}   ↳ n_cells : {block.n_cells}")
        print(f"{indent}   ↳ bounds  : {block.bounds}")
        print(f"{indent}   ↳ point_data: {list(block.point_data.keys())}")
        print(f"{indent}   ↳ cell_data : {list(block.cell_data.keys())}")

# ----------------------------
# Replace with your CGNS file
filename = "vol.cgns"
mesh = pv.read(filename)

# ----------------------------
print(f"\n🔍 Exploring CGNS file: {filename}")
explore_block(mesh)

# Example plotting and visualization
'''
Airfoil SUrface
'''
airfoil = mesh[0][0][0]  # BaseSurfaceSol[0] → NSWallAdiabaticBCZone1[0]

plotter = pv.Plotter()
plotter.add_mesh(airfoil, scalars="SkinFrictionMagnitude", cmap="inferno", show_edges=False)
plotter.show()

'''
full z=0 field (Symmetry + Wall + Farfield)
'''

z0_blocks = [mesh[0][0][0], mesh[0][1][0], mesh[0][2][0]]  # airfoil + farfield + z=0 symmetry
combined = pv.MultiBlock(z0_blocks).combine()

plotter = pv.Plotter()
plotter.add_mesh(combined, scalars="CoefPressure", cmap="coolwarm", show_edges=False)
plotter.show()

'''
z=0 slice
'''

mesh = pv.read("vol.cgns")
sym_z0 = mesh[0][2][0]  # BaseSurfaceSol[2] → SymmetryBCZone3[0]

plotter = pv.Plotter()
plotter.add_mesh(sym_z0, scalars="CoefPressure", cmap="coolwarm", show_edges=False)
plotter.add_axes()
plotter.show()
