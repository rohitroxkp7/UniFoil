from unifoil.extract_data import ExtractData
import matplotlib.pyplot as plt

# ===========================================
#   Initialize the ExtractData class
# ===========================================
ed = ExtractData()

# -------------------------------------------
# Choose which airfoil/case to work with
# -------------------------------------------
AIRFOIL = 5
CASE    = 2
BLOCK   = 2  # your mid-plane block

# Axis limits for plots
XLIM = (-1, 1)
YLIM = (-1, 1)

# ===========================================
# 1) Display CGNS structure hierarchy
# ===========================================
print("\n=== [1] Display CGNS File Structure ===")
ed.surf_turb(
    airfoil_number=AIRFOIL,
    case_number=CASE,
    action="display_structure"
)

# ===========================================
# 2) Plot scalar fields (Cp, Mach)
# ===========================================
print("\n=== [2] Plot Coefficient of Pressure (Cp) ===")
ed.surf_turb(
    airfoil_number=AIRFOIL,
    case_number=CASE,
    field_name="CoefPressure",
    action="plot_field",
    block_index=BLOCK,
    xlim=XLIM, ylim=YLIM,
    levels=200, cmap="RdBu",
    overlay_airfoil=True
)

print("\n=== [3] Plot Mach ===")
ed.surf_turb(
    airfoil_number=AIRFOIL,
    case_number=CASE,
    field_name="Mach",
    action="plot_field",
    block_index=BLOCK,
    xlim=XLIM, ylim=YLIM,
    levels=200, cmap="viridis",
    overlay_airfoil=True
)

# ===========================================
# 3) Plot velocity magnitude and components
# ===========================================
print("\n=== [4] Plot Velocity Magnitude (|u|) ===")
ed.surf_turb(
    airfoil_number=AIRFOIL,
    case_number=CASE,
    field_name="Velocity",
    vel_component='a',   # |u|
    action="plot_field",
    block_index=BLOCK,
    xlim=XLIM, ylim=YLIM,
    levels=200, cmap="viridis",
    overlay_airfoil=True
)

print("\n=== [5] Plot Velocity X-component (u_x) ===")
ed.surf_turb(
    airfoil_number=AIRFOIL,
    case_number=CASE,
    field_name="Velocity",
    vel_component='b',   # u_x
    action="plot_field",
    block_index=BLOCK,
    xlim=XLIM, ylim=YLIM,
    levels=200, cmap="viridis",
    overlay_airfoil=True
)

print("\n=== [6] Plot Velocity Y-component (u_y) ===")
ed.surf_turb(
    airfoil_number=AIRFOIL,
    case_number=CASE,
    field_name="Velocity",
    vel_component='c',   # u_y
    action="plot_field",
    block_index=BLOCK,
    xlim=XLIM, ylim=YLIM,
    levels=200, cmap="viridis",
    overlay_airfoil=True
)

# ===========================================
# 4) Extract field values (x, y, q) on z-plane
# ===========================================
print("\n=== [7] Extract Cp field (CoefPressure) and save ===")
res = ed.surf_turb(
    airfoil_number=AIRFOIL,
    case_number=CASE,
    field_name="CoefPressure",
    action="extract_xy_quantity",
    block_index=BLOCK,
    save_path=f"cp_airfoil{AIRFOIL}_case{CASE}.npz"
)

if res:
    x, y, q = res
    print(f"Extracted {len(x)} points. Example values:")
    print("x[0:5] =", x[:5])
    print("y[0:5] =", y[:5])
    print("q[0:5] =", q[:5])

    # Quick scatter plot of extracted Cp
    plt.figure(figsize=(7, 5))
    sc = plt.scatter(x, y, c=q, cmap="coolwarm")
    plt.colorbar(sc, label=r"$C_p$")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$", rotation=0)
    plt.xlim(XLIM); plt.ylim(YLIM)
    plt.title(f"Extracted $C_p$ (Airfoil {AIRFOIL}, Case {CASE})")
    plt.show()

# ===========================================
# 5) (Optional) Nearest AVAILABLE case by (M, AoA, Re)
#     — uses the fallback logic you asked for
# ===========================================
print("\n=== [8] Plot |u| for nearest AVAILABLE to (M=0.60, AoA=5.3°, Re=2.745068e6) ===")
ed.surf_turb(
    airfoil_number=3,
    case_number=None,               # trigger nearest-available search
    Mach=0.60, AoA=5.3, Re=2.745068e6,
    field_name="Velocity", vel_component='a',
    block_index=BLOCK,
    action="plot_field",
    xlim=XLIM, ylim=YLIM,
    levels=200, cmap="viridis",
    overlay_airfoil=True
)

print("\n✅ All demo steps completed successfully.")
