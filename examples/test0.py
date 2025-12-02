import unifoil
unifoil.gen_ft()

unifoil.gen_nlf()

from unifoil.extract_data import ExtractData
ed = ExtractData()
x, y = ed.extract_airfoil_coords(airfoil_number=1, source="turb", plot_flag=True)

x, y = ed.extract_airfoil_coords(airfoil_number=1510, source="translam", plot_flag=True)

# Total number of airfoils in FT             - 30,000
# Total number of airfoils in NLF and Transi - 4,800

