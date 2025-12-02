from unifoil.extract_data import ExtractData

# ===========================================
#   Initialize the ExtractData class
# ===========================================
ed = ExtractData()


data = ed.get_supplement_transi(airfoil_number=1, case_number=7,plot_flag=True)
