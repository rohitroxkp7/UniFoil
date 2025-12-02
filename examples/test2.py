from unifoil.extract_data import ExtractData
import matplotlib.pyplot as plt

# =====================================================
#  Initialize ExtractData class
# =====================================================
ed = ExtractData()

# =====================================================
#  Example: identify case by Mach, AoA, and Re
# =====================================================

# Suppose the user knows approximate flow conditions
airfoil_number = 3
Mach = 0.6
AoA = 5.3
Re = 2745068

BLOCK_INDEX = 2  # mid-plane block in your surface CGNSs

# Step 1 — Find the closest AVAILABLE case automatically
# and display its CGNS structure (uses the fallback if needed)
print("\n=== [1] Automatically match case from flow conditions ===")
ed.surf_turb(
    airfoil_number=airfoil_number,
    Mach=Mach,
    AoA=AoA,
    Re=Re,
    action="display_structure"
)

# Step 2 — Plot Cp field for that matched case (cell-centered @ auto z-plane)
print("\n=== [2] Plot Cp field for automatically selected case ===")
ed.surf_turb(
    airfoil_number=airfoil_number,
    Mach=Mach,
    AoA=AoA,
    Re=Re,
    field_name="CoefPressure",
    block_index=BLOCK_INDEX,
    action="plot_field",
    xlim=(-1, 1), ylim=(-1, 1),
    levels=200, cmap="viridis",
    overlay_airfoil=True
)

# Step 3 — Extract Mach field values on the dominant z-plane and save
print("\n=== [3] Extract Mach field data and save ===")
result = ed.surf_turb(
    airfoil_number=airfoil_number,
    Mach=Mach,
    AoA=AoA,
    Re=Re,
    field_name="Mach",
    block_index=BLOCK_INDEX,
    action="extract_xy_quantity",
    save_path=f"mach_airfoil{airfoil_number}_Mach{Mach:.2f}_AoA{AoA:.1f}_Re{int(Re)}.npz"
)

# Optional quick visualization of extracted Mach (x, y, q)
if result:
    x, y, q = result
    plt.figure(figsize=(7, 5))
    sc = plt.scatter(x, y, c=q, cmap="viridis")
    plt.colorbar(sc, label="Mach")
    plt.xlabel("x")
    plt.ylabel("y", rotation=0)
    plt.xlim((-1, 1)); plt.ylim((-1, 1))
    plt.title(f"Airfoil {airfoil_number} (Matched Mach={Mach:.2f}, AoA={AoA:.1f}, Re={Re:.1e})")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

print("\n✅ Completed automatic Mach–AoA–Re match demo.")
