from unifoil.extract_data import ExtractData

ed = ExtractData()
dat = ed.get_aero_coeffs_turb(airfoil_number=5801, case_number=1)

data = ed.load_convergence_data_turb(airfoil_number=3, case_number=1, print_flag=True)

# Access specific convergence fields
if data:
    print("Available keys:", list(data.keys()))
    print("Example CFL values:", data.get("CFL", [])[:5])
