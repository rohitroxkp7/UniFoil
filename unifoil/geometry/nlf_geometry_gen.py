import numpy as np
import os

class AirfoilGenerator:
    def __init__(self, coefs_file, modes_file, xslice_file, output_folder='airfoil_nlf_geom'):
        """
        Parameters
        ----------
        coefs_file : str
            Path to coefficients file (coefs.txt).
        modes_file : str
            Path to modes file (modes.txt).
        xslice_file : str
            Path to x-coordinates file (xslice.txt).
        output_folder : str, optional
            Folder where generated .dat files will be stored.
        """
        self.coefs_file = coefs_file
        self.modes_file = modes_file
        self.xslice_file = xslice_file
        self.output_folder = output_folder
        self.generated_airfoilys = "airfoilys.txt"  # local generated file name

    # ------------------------------------------------------------------
    def _generate_airfoilys(self):
        """Generate airfoilys.txt from coefs.txt and modes.txt."""
        print("[unifoil] 🔧 Generating airfoilys.txt from coefs and modes ...")

        coefs = np.loadtxt(self.coefs_file)
        modes = np.loadtxt(self.modes_file)

        # Modal reconstruction
        airfy = np.dot(coefs, modes)
        ns = airfy.shape[0]

        with open(self.generated_airfoilys, "w") as f:
            for ins in range(ns):
                np.savetxt(f, airfy[ins, :][None, :], fmt="%.15f", delimiter=" ")
        print(f"[unifoil] ✅ Generated airfoilys.txt with {ns} airfoils.")

        return self.generated_airfoilys

    # ------------------------------------------------------------------
    def generate(self):
        """Main entry: build airfoilys if missing and generate .dat geometries."""
        # Step 1 — Generate airfoilys.txt locally
        airfoil_file = self._generate_airfoilys()
        success = False
        try:
            # Step 2 — Load data and xslice
            airfoil_data = np.loadtxt(airfoil_file)
            xslice = np.loadtxt(self.xslice_file)

            if airfoil_data.shape[1] != len(xslice):
                print(f"[unifoil] ❌ Error: xslice length ({len(xslice)}) "
                      f"does not match airfoil data columns ({airfoil_data.shape[1]}).")
                return

            os.makedirs(self.output_folder, exist_ok=True)

            # Step 3 — Loop through airfoils and write .dat files
            for n in range(airfoil_data.shape[0]):
                airfoil_y = airfoil_data[n, :]
                dat_filename = os.path.join(self.output_folder, f"airfoil_{n+1:03d}.dat")

                coords = list(zip(xslice, airfoil_y))
                if not (np.isclose(coords[0][0], coords[-1][0]) and np.isclose(coords[0][1], coords[-1][1])):
                    coords.append(coords[0])  # close loop

                with open(dat_filename, "w") as file:
                    for i, (x, y) in enumerate(coords, start=1):
                        file.write(f"{i:03d} {x:.8e} {y:.8e}\n")

            success = True
            print(f"[unifoil] ✅ All airfoil geometries saved to '{self.output_folder}'.")
        finally:
            if success and os.path.exists(airfoil_file):
                try:
                    os.remove(airfoil_file)
                    print(f"[unifoil] 🗑️ Removed temporary file '{airfoil_file}'.")
                except OSError as err:
                    print(f"[unifoil] ⚠️ Could not delete '{airfoil_file}': {err}")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    gen = AirfoilGenerator(
        coefs_file="input_nlf/coefs.txt",
        modes_file="input_nlf/modes.txt",
        xslice_file="input_nlf/xslice.txt",
        output_folder="airfoil_nlf_geom"
    )
    gen.generate()
    print("[unifoil] ✅ Finished generating NLF airfoil geometries.")
