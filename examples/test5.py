from unifoil.extract_data import ExtractData

ed = ExtractData()
dat = ed.get_aero_coeffs_lam(airfoil_number=4136, case_number=5)

data = ed.load_convergence_data_lam(airfoil_number=4136, case_number=5, print_flag=True)

# Access specific convergence fields
if data:
    print("Available keys:", list(data.keys()))
    print("Example CFL values:", data.get("CFL", [])[:5])
