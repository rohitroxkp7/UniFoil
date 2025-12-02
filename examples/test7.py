from unifoil.extract_data import ExtractData

ed = ExtractData()
dat = ed.get_aero_coeffs_transi(airfoil_number=1, case_number=1)

data = ed.load_convergence_data_transi(airfoil_number=1, case_number=9, print_flag=True)

# Access specific convergence fields
if data:
    print("Available keys:", list(data.keys()))
    print("Example CFL values:", data.get("CFL", [])[:5])

