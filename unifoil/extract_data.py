import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import niceplots
import pickle
import re

from unifoil.surfCGNSPostProcessor import surfCGNSPostProcessor
from unifoil.post_processor_class_transi import sup_transi


plt.style.use(niceplots.get_style())

class ExtractData:
    """
    Class for extracting airfoil geometry and case information
    from the current working directory.

    Expected working directory contents:
      - Airfoil_Case_Data_turb.csv
      - Airfoil_Case_Data_Trans_Lam.csv
      - airfoil_ft_geom/ (contains airfoil_###.dat)
      - airfoil_nlf_geom/ (contains airfoil_###.dat)
    """

    def __init__(self):
        # Paths in the current working directory
        self.cwd = os.getcwd()
        self.ft_geom_path = os.path.join(self.cwd, "airfoil_ft_geom")
        self.nlf_geom_path = os.path.join(self.cwd, "airfoil_nlf_geom")

        # CSV paths
        self.turb_csv = os.path.join(self.cwd, "Airfoil_Case_Data_turb.csv")
        self.translam_csv = os.path.join(self.cwd, "Airfoil_Case_Data_Trans_Lam.csv")
        

        # Check folders
        if not os.path.exists(self.ft_geom_path):
            print(f"Warning: '{self.ft_geom_path}' not found.")
        if not os.path.exists(self.nlf_geom_path):
            print(f"Warning: '{self.nlf_geom_path}' not found.")

    '''
    def extract_airfoil_coords(self, airfoil_number, source="turb", plot_flag=False):
        """
        Extracts x and y coordinates for the airfoil corresponding to the given number.
    
        Parameters
        ----------
        airfoil_number : int
            The airfoil number (as in the 'airfoil' column of the CSV).
        source : str, optional
            Which dataset to use ("turb" or "translam").
        plot_flag : bool, optional
            Whether to plot the airfoil coordinates.
    
        Returns
        -------
        tuple of np.ndarray
            (x, y) coordinates of the specified airfoil.
        """
        # Select source file and geometry folder
        if source.lower() == "turb":
            csv_path = self.turb_csv
            geom_path = self.ft_geom_path
            ft_style = True
        else:
            csv_path = self.translam_csv
            geom_path = self.nlf_geom_path
            ft_style = False
    
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
        # Read CSV
        df = pd.read_csv(csv_path)
        if "airfoil" not in df.columns:
            raise ValueError("CSV must have an 'airfoil' column.")
    
        # Locate row
        row = df.loc[df["airfoil"] == airfoil_number]
        if row.empty:
            raise ValueError(f"Airfoil {airfoil_number} not found in {os.path.basename(csv_path)}")
    
        airfoil_idx = int(row["airfoil"].values[0])
    
        # Smart filename pattern based on dataset type
        if ft_style:
            # e.g., airfoil_1.dat, airfoil_2.dat
            dat_filename = os.path.join(geom_path, f"airfoil_{airfoil_idx}.dat")
        else:
            # e.g., airfoil_001.dat, airfoil_002.dat
            dat_filename = os.path.join(geom_path, f"airfoil_{airfoil_idx:03d}.dat")
    
        # Check existence
        if not os.path.exists(dat_filename):
            raise FileNotFoundError(f"Geometry file not found: {dat_filename}")
    
        # Load coordinates
        data = np.loadtxt(dat_filename)
        x, y = data[:, 1], data[:, 2]
    
        # Plot if requested
        if plot_flag:
            plt.figure(figsize=(7, 5))
            plt.plot(x, y, "-")
            plt.gca().set_aspect("equal", adjustable="box")
            plt.title(f"Airfoil #{airfoil_idx} ({source})")
            plt.xlabel("x")
            plt.ylabel("y",rotation=0,labelpad=10)
            plt.show()
    
        return x, y
    '''
    
    def extract_airfoil_coords(self, airfoil_number, source="turb", plot_flag=False):
        """
        Extracts x and y coordinates for the airfoil corresponding to the given number.
    
        Parameters
        ----------
        airfoil_number : int
            The airfoil number (as in the 'airfoil' column of the CSV).
        source : str, optional
            Which dataset to use ("turb" or "translam").
        plot_flag : bool, optional
            Whether to plot the airfoil coordinates.
    
        Returns
        -------
        tuple of np.ndarray
            (x, y) coordinates of the specified airfoil.
        """
    
        # ---------------------------------------------------------------
        # Select dataset type
        # ---------------------------------------------------------------
        if source.lower() == "turb":
            csv_path = self.turb_csv
            geom_path = self.ft_geom_path
            ft_style = True
        else:
            csv_path = self.translam_csv
            geom_path = self.nlf_geom_path
            ft_style = False
    
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
        # ---------------------------------------------------------------
        # Read CSV and locate airfoil
        # ---------------------------------------------------------------
        df = pd.read_csv(csv_path)
        if "airfoil" not in df.columns:
            raise ValueError("CSV must have an 'airfoil' column.")
    
        row = df.loc[df["airfoil"] == airfoil_number]
        if row.empty:
            raise ValueError(f"Airfoil {airfoil_number} not found in {os.path.basename(csv_path)}")
    
        airfoil_idx = int(row["airfoil"].values[0])
    
        # ---------------------------------------------------------------
        # Handle NLF mapping logic (translam only)
        # ---------------------------------------------------------------
        if not ft_style:
            matched_csv_path = os.path.join(self.cwd, "matched_files.csv")
    
            if not os.path.exists(matched_csv_path):
                print(f"[unifoil] ⚠️ matched_files.csv not found at {matched_csv_path}")
                mapped_airfoil = airfoil_idx
            else:
                df_match = pd.read_csv(matched_csv_path)
    
                # Extract airfoil numbers from filenames
                def extract_number(name):
                    match = re.search(r"airfoil_(\d+)_G2", name)
                    return int(match.group(1)) if match else None
    
                df_match["old_airfoil"] = df_match["Matched_Grid_File"].apply(extract_number)
                df_match["new_airfoil"] = df_match["Extracted_File"].apply(extract_number)
    
                # Reverse mapping: new_airfoil → old_airfoil
                reverse_map = dict(zip(df_match["new_airfoil"], df_match["old_airfoil"]))
                mapped_airfoil = reverse_map.get(airfoil_idx, airfoil_idx)
    
                if mapped_airfoil != airfoil_idx:
                    print(f"[unifoil] Using mapped airfoil {airfoil_idx} → {mapped_airfoil}")
                else:
                    print(f"[unifoil] No mapping found, using {airfoil_idx} directly.")
        else:
            mapped_airfoil = airfoil_idx
    
        # ---------------------------------------------------------------
        # Build filename pattern
        # ---------------------------------------------------------------
        if ft_style:
            dat_filename = os.path.join(geom_path, f"airfoil_{mapped_airfoil}.dat")
        else:
            dat_filename = os.path.join(geom_path, f"airfoil_{mapped_airfoil:03d}.dat")
    
        if not os.path.exists(dat_filename):
            raise FileNotFoundError(f"Geometry file not found: {dat_filename}")
    
        # ---------------------------------------------------------------
        # Load coordinates
        # ---------------------------------------------------------------
        data = np.loadtxt(dat_filename)
        x, y = data[:, 1], data[:, 2]
    
        # ---------------------------------------------------------------
        # Plot if requested
        # ---------------------------------------------------------------
        if plot_flag:
            plt.figure(figsize=(7, 5))
            plt.plot(x, y, "-")
            plt.gca().set_aspect("equal", adjustable="box")
            plt.title(f"Airfoil #{airfoil_idx} ({source}) → Geometry #{mapped_airfoil}")
            plt.xlabel("x")
            plt.ylabel("y", rotation=0, labelpad=10)
            plt.show()
    
        return x, y

    
    # FT turb cases:

    def surf_turb(
        self,
        airfoil_number,
        case_number=None,
        Mach=None,
        AoA=None,
        Re=None,
        field_name="Velocity",
        vel_component='a',
        block_index=2,
        action="plot_field",
        **kwargs
    ):
        """
        Locate and process CGNS surface data for a given airfoil and case.
        """
    
        # Step 1: Load CSV
        if not os.path.exists(self.turb_csv):
            print(f"[unifoil] ❌ CSV file not found: {self.turb_csv}")
            return None
    
        df = pd.read_csv(self.turb_csv)
    
        # Determine case number
        if case_number is None:
            if Mach is None or AoA is None or Re is None:
                print("[unifoil] ❌ Must specify either (airfoil, case_number) or (airfoil, Mach, AoA, Re).")
                return None
    
            subset = df[df["airfoil"] == airfoil_number]
            if subset.empty:
                print(f"[unifoil] ❌ Airfoil {airfoil_number} not found in CSV.")
                return None
    
            dist = np.sqrt(
                (subset["Mach"] - Mach) ** 2 +
                (subset["AoA"] - AoA) ** 2 +
                ((subset["Re"] - Re) / subset["Re"].max()) ** 2
            )
            best_idx = dist.idxmin()
            row = subset.loc[best_idx]
            case_number = int(row["case"])
            Mach, AoA, Re = row["Mach"], row["AoA"], row["Re"]
            print(f"[unifoil] Closest match → Airfoil {airfoil_number}, Case {case_number} "
                  f"(Mach={Mach:.3f}, AoA={AoA:.3f}, Re={Re:.2e})")
        else:
            row = df[(df["airfoil"] == airfoil_number) & (df["case"] == case_number)]
            if row.empty:
                print(f"[unifoil] ❌ No entry found for Airfoil {airfoil_number}, Case {case_number}.")
                return None
            Mach, AoA, Re = row["Mach"].values[0], row["AoA"].values[0], row["Re"].values[0]
    
        # Step 2: Find CGNS file
        case_index = case_number - 1  # ✅ 1-indexed → 0-indexed
        filename_pattern = f"airfoil_{airfoil_number}_G2_A_L0_case_{case_index}_000_surf_turb.cgns"
        found_path = None
    
        for i in range(1, 7):
            cutout_folder = os.path.join(self.cwd, f"Turb_Cutout_{i}")
            if not os.path.isdir(cutout_folder):
                continue
            candidate_path = os.path.join(cutout_folder, filename_pattern)
            if os.path.exists(candidate_path):
                found_path = candidate_path
                break
    
        if not found_path:
            print(f"[unifoil] ⚠️ CGNS file not found for Airfoil {airfoil_number}, Case {case_number}. "
                  f"(possibly missing or unconverged simulation)")
            return None
    
        print(f"[unifoil] ✅ Found CGNS file: {found_path}")
    
        # Step 3: Geometry file
        airfoil_file = None
        ft_file_candidate = os.path.join(self.ft_geom_path, f"airfoil_{airfoil_number}.dat")
    
        if os.path.exists(ft_file_candidate):
            airfoil_file = ft_file_candidate
        else:
            print(f"[unifoil] ⚠️ Airfoil geometry file not found for Airfoil {airfoil_number}.")
    
        # Step 4: Execute requested action
        try:
            post = surfCGNSPostProcessor(found_path, airfoil_file=airfoil_file)
            if action == "plot_field":
                post.plot_field(field_name=field_name, block_index=block_index, vel_component=vel_component)
                return None
            elif hasattr(post, action):
                func = getattr(post, action)
                if callable(func):
                    # Always pass field_name and block_index if the function expects them
                    if "field_name" in func.__code__.co_varnames:
                        kwargs.setdefault("field_name", field_name)
                    if "block_index" in func.__code__.co_varnames:
                        kwargs.setdefault("block_index", block_index)
                    if "vel_component" in func.__code__.co_varnames:
                        kwargs.setdefault("vel_component", vel_component)
            
                    result = func(**kwargs)
                    return result

                else:
                    print(f"[unifoil] ⚠️ '{action}' exists but is not callable.")
                    return None
            else:
                print(f"[unifoil] ⚠️ Unknown action '{action}'. Available: "
                      f"{[m for m in dir(post) if not m.startswith('_')]}")
                return None
        except Exception as e:
            print(f"[unifoil] ❌ Error running surfCGNSPostProcessor.{action}: {e}")
            return None

    def get_aero_coeffs_turb(self, airfoil_number, case_number, print_flag=True):
        """
        Extract CL and CD from airfoil_<num>_G2_A_L0_case_<num>_analysis.csv.
        Handles both 7-row and 8-row formats, converts text to floats,
        and converts user-provided 1-indexed case_number to 0-indexed filenames.
    
        Parameters
        ----------
        airfoil_number : int
            Airfoil number.
        case_number : int
            Case number (1-indexed, as in CSV).
        print_flag : bool, optional
            If True, prints Cl and Cd to console.
    
        Returns
        -------
        tuple : (Cl, Cd) or (None, None) if file not found.
        """
        import pandas as pd
        import numpy as np
        import os
    
        # Convert 1-based case number to 0-based for filename
        case_index = case_number - 1
        filename = f"airfoil_{airfoil_number}_G2_A_L0_case_{case_index}_analysis.csv"
    
        # Search both datasets
        found_path = None
        for folder in ["airfoil_data_from_simulations_turb_set1", "airfoil_data_from_simulations_turb_set2"]:
            candidate = os.path.join(self.cwd, folder, filename)
            if os.path.exists(candidate):
                found_path = candidate
                break
    
        if not found_path:
            print(f"[unifoil] ⚠️ Analysis file not found for Airfoil {airfoil_number}, Case {case_number} (→ file case {case_index}).")
            return None, None
    
        try:
            df = pd.read_csv(found_path, header=None)
    
            # Locate CL/CD header
            header_row = None
            for i in range(len(df)):
                vals = [str(v).strip().upper() for v in df.iloc[i, :2].values]
                if "CL" in vals and "CD" in vals:
                    header_row = i
                    break
    
            # Extract row below CL/CD header or fallback to last row
            if header_row is not None and header_row + 1 < len(df):
                Cl_raw, Cd_raw = df.iloc[header_row + 1, 0], df.iloc[header_row + 1, 1]
            else:
                Cl_raw, Cd_raw = df.iloc[-1, 0], df.iloc[-1, 1]
    
            # Convert to float robustly
            try:
                Cl = float(Cl_raw)
                Cd = float(Cd_raw)
            except Exception:
                Cl, Cd = pd.to_numeric([Cl_raw, Cd_raw], errors="coerce")
    
            if np.isnan(Cl) or np.isnan(Cd):
                raise ValueError(f"Invalid numeric values: Cl={Cl_raw}, Cd={Cd_raw}")
    
            if print_flag:
                print(f"[unifoil] ✅ Airfoil {airfoil_number}, Case {case_number} → File Case {case_index}: Cl = {Cl:.6f}, Cd = {Cd:.6f}")
                print(f"[unifoil]    File: {found_path}")
    
            return Cl, Cd
    
        except Exception as e:
            print(f"[unifoil] ❌ Error reading {found_path}: {e}")
            return None, None

    def load_convergence_data_turb(self, airfoil_number, case_number, print_flag=False):
        """
        Load convergence pickle data for a given airfoil and case.
        Automatically converts user-provided 1-indexed case_number to 0-indexed filenames.
    
        Parameters
        ----------
        airfoil_number : int
            Airfoil number.
        case_number : int
            Case number (1-indexed, as seen in CSV).
        print_flag : bool, optional
            If True, prints a summary of the loaded contents.
    
        Returns
        -------
        dict or None
            Dictionary loaded from pickle, or None if file not found or unreadable.
        """
        import os
        import pickle
        import numpy as np
    
        # Convert 1-based to 0-based case index
        case_index = case_number - 1
        filename = f"airfoil_{airfoil_number}_G2_A_L0_case_{case_index}_convergence.pkl"
    
        # Search both dataset folders
        found_path = None
        for folder in ["airfoil_data_from_simulations_turb_set1", "airfoil_data_from_simulations_turb_set2"]:
            candidate = os.path.join(self.cwd, folder, filename)
            if os.path.exists(candidate):
                found_path = candidate
                break
    
        if not found_path:
            print(f"[unifoil] ⚠️ Convergence file not found for Airfoil {airfoil_number}, Case {case_number} (→ file case {case_index}).")
            return None
    
        try:
            with open(found_path, "rb") as f:
                data = pickle.load(f)
    
            if print_flag:
                print(f"[unifoil] ✅ Loaded convergence data for Airfoil {airfoil_number}, Case {case_number} (→ file case {case_index}):")
                for key, val in data.items():
                    if isinstance(val, (list, tuple, np.ndarray)):
                        print(f"  {key}: array with {len(val)} elements")
                    else:
                        print(f"  {key}: {val}")
    
            return data

        except Exception as e:
            print(f"[unifoil] ❌ Error reading {found_path}: {e}")
            return None

    def _build_nlf_reverse_map(self):
        """
        Returns mapping from new_airfoil -> old_airfoil using matched_files.csv.
        """
        matched_csv_path = os.path.join(self.cwd, "matched_files.csv")
        if not os.path.exists(matched_csv_path):
            print(f"[unifoil] ⚠️ matched_files.csv not found at {matched_csv_path}")
            return {}

        df_match = pd.read_csv(matched_csv_path)

        def extract_number(name):
            match = re.search(r"airfoil_(\d+)_G2", name)
            return int(match.group(1)) if match else None

        df_match["old_airfoil"] = df_match["Matched_Grid_File"].apply(extract_number)
        df_match["new_airfoil"] = df_match["Extracted_File"].apply(extract_number)
        reverse_map = dict(zip(df_match["new_airfoil"], df_match["old_airfoil"]))
        return reverse_map

    def export_ft_modal_case_table(
        self,
        airfoil_range,
        case_range,
        output_csv="ft_modal_case_data.csv",
        print_progress=True,
    ):
        """
        Export modal coefficients + flow/aero data for FT (turbulent) airfoils.

        Parameters
        ----------
        airfoil_range : tuple(int, int)
            Inclusive (min_airfoil, max_airfoil) IDs to include.
        case_range : tuple(int, int)
            Inclusive (min_case, max_case) case IDs to include.
        output_csv : str, optional
            Output CSV file name (written in current working directory unless an
            absolute path is provided).
        print_progress : bool, optional
            If True, logs the number of exported rows.

        Returns
        -------
        str
            Absolute path to the written CSV file.
        """
        if not os.path.exists(self.turb_csv):
            raise FileNotFoundError(f"CSV file not found: {self.turb_csv}")

        airfoil_min, airfoil_max = airfoil_range
        case_min, case_max = case_range

        train_path = os.path.join(self.cwd, "input_ft", "training.dat")
        valid_path = os.path.join(self.cwd, "input_ft", "validating.dat")
        basis_path = os.path.join(self.cwd, "input_ft", "basis.txt")
        if not all(os.path.exists(p) for p in (train_path, valid_path, basis_path)):
            raise FileNotFoundError(
                "Missing one of input_ft/basis.txt, input_ft/training.dat, or input_ft/validating.dat in the current directory."
            )

        train_data = np.loadtxt(train_path)
        valid_data = np.loadtxt(valid_path)
        basis = np.loadtxt(basis_path)
        num_ft_modes = max(basis.shape[0] - 1, 1)
        n_train = train_data.shape[0]
        n_valid = valid_data.shape[0]

        def modal_coeffs_for_airfoil(airfoil_id):
            if airfoil_id < 1 or airfoil_id > (n_train + n_valid):
                raise ValueError(f"Airfoil ID {airfoil_id} outside available FT geometries.")
            zero_idx = airfoil_id - 1
            if zero_idx < n_train:
                return train_data[zero_idx, :num_ft_modes]
            zero_idx -= n_train
            return valid_data[zero_idx, :num_ft_modes]

        df_turb = pd.read_csv(self.turb_csv)
        mask = (
            (df_turb["airfoil"] >= airfoil_min)
            & (df_turb["airfoil"] <= airfoil_max)
            & (df_turb["case"] >= case_min)
            & (df_turb["case"] <= case_max)
        )
        filtered = df_turb.loc[mask].copy()
        if filtered.empty:
            raise ValueError("No FT cases found for the specified airfoil/case ranges.")

        records = []
        for _, row in filtered.iterrows():
            airfoil_id = int(row["airfoil"])
            case_id = int(row["case"])
            try:
                coeffs = modal_coeffs_for_airfoil(airfoil_id)
            except Exception as exc:
                print(f"[unifoil] ⚠️ Skipping airfoil {airfoil_id}: {exc}")
                continue

            Cl, Cd = self.get_aero_coeffs_turb(airfoil_id, case_id, print_flag=False)
            if Cl is None or Cd is None:
                print(f"[unifoil] ⚠️ Skipping airfoil {airfoil_id}, case {case_id}: missing Cl/Cd file.")
                continue

            record = {
                "airfoil": airfoil_id,
                "case": case_id,
                "Mach": row.get("Mach"),
                "AoA": row.get("AoA"),
                "Re": row.get("Re"),
                "Cl": Cl,
                "Cd": Cd,
            }
            for idx, value in enumerate(coeffs, start=1):
                record[f"mode_{idx}"] = value
            records.append(record)

        if not records:
            raise ValueError("No rows exported (all candidates skipped).")

        export_df = pd.DataFrame(records)
        mode_cols = [c for c in export_df.columns if c.startswith("mode_")]
        ordered_cols = ["airfoil", "case", "Mach", "AoA", "Re", "Cl", "Cd"] + sorted(
            mode_cols, key=lambda name: int(name.split("_")[1])
        )
        export_df = export_df[ordered_cols]

        output_path = output_csv
        if not os.path.isabs(output_path):
            output_path = os.path.join(self.cwd, output_csv)
        export_df.to_csv(output_path, index=False)

        if print_progress:
            print(f"[unifoil] ✅ Exported {len(export_df)} FT rows to {output_path}")

        return output_path

    def _export_nlf_modal_case_table_common(
        self,
        airfoil_range,
        case_range,
        output_csv,
        clcd_fetcher,
        dataset_label,
        print_progress=True,
    ):
        if not os.path.exists(self.translam_csv):
            raise FileNotFoundError(f"CSV file not found: {self.translam_csv}")

        coefs_path = os.path.join(self.cwd, "input_nlf", "coefs.txt")
        if not os.path.exists(coefs_path):
            raise FileNotFoundError(f"Missing NLF coefficients file: {coefs_path}")

        nlf_coefs = np.loadtxt(coefs_path)
        total_airfoils = nlf_coefs.shape[0]
        num_nlf_modes = nlf_coefs.shape[1]
        reverse_map = self._build_nlf_reverse_map()

        airfoil_min, airfoil_max = airfoil_range
        case_min, case_max = case_range

        df = pd.read_csv(self.translam_csv)
        mask = (
            (df["airfoil"] >= airfoil_min)
            & (df["airfoil"] <= airfoil_max)
            & (df["case"] >= case_min)
            & (df["case"] <= case_max)
        )
        subset = df.loc[mask].copy()
        if subset.empty:
            raise ValueError(f"No {dataset_label} cases found for specified ranges.")

        records = []
        for _, row in subset.iterrows():
            airfoil_id = int(row["airfoil"])
            case_id = int(row["case"])
            mapped_id = reverse_map.get(airfoil_id, airfoil_id)

            if mapped_id < 1 or mapped_id > total_airfoils:
                print(f"[unifoil] ⚠️ Skipping airfoil {airfoil_id}: mapped ID {mapped_id} invalid.")
                continue

            coeffs = nlf_coefs[mapped_id - 1, :num_nlf_modes]
            Cl, Cd = clcd_fetcher(airfoil_id, case_id, print_flag=False)
            if Cl is None or Cd is None:
                print(f"[unifoil] ⚠️ Skipping airfoil {airfoil_id}, case {case_id}: missing Cl/Cd file.")
                continue

            record = {
                "airfoil": airfoil_id,
                "case": case_id,
                "Mach": row.get("Mach"),
                "AoA": row.get("AoA"),
                "Re": row.get("Re"),
                "Cl": Cl,
                "Cd": Cd,
            }
            for idx, value in enumerate(coeffs, start=1):
                record[f"mode_{idx}"] = value
            records.append(record)

        if not records:
            raise ValueError(f"No rows exported for {dataset_label} (all skipped).")

        export_df = pd.DataFrame(records)
        mode_cols = [c for c in export_df.columns if c.startswith("mode_")]
        ordered_cols = [
            "airfoil",
            "case",
            "Mach",
            "AoA",
            "Re",
            "Cl",
            "Cd",
        ] + sorted(mode_cols, key=lambda name: int(name.split("_")[1]))
        export_df = export_df[ordered_cols]

        output_path = output_csv
        if not os.path.isabs(output_path):
            output_path = os.path.join(self.cwd, output_csv)
        export_df.to_csv(output_path, index=False)

        if print_progress:
            print(f"[unifoil] ✅ Exported {len(export_df)} {dataset_label} rows to {output_path}")

        return output_path

    def export_nlf_turb_modal_case_table(
        self,
        airfoil_range,
        case_range,
        output_csv="nlf_turb_modal_case_data.csv",
        print_progress=True,
    ):
        """
        Export modal coefficients + flow/aero data for NLF airfoils (fully turbulent sims).
        """
        return self._export_nlf_modal_case_table_common(
            airfoil_range,
            case_range,
            output_csv,
            clcd_fetcher=self.get_aero_coeffs_lam,
            dataset_label="NLF turb",
            print_progress=print_progress,
        )

    def export_nlf_transi_modal_case_table(
        self,
        airfoil_range,
        case_range,
        output_csv="nlf_transi_modal_case_data.csv",
        print_progress=True,
    ):
        """
        Export modal coefficients + flow/aero data for NLF airfoils (transition sims).
        """
        return self._export_nlf_modal_case_table_common(
            airfoil_range,
            case_range,
            output_csv,
            clcd_fetcher=self.get_aero_coeffs_transi,
            dataset_label="NLF transi",
            print_progress=print_progress,
        )

    # NLF turb cases:

    def surf_lam(
            self,
            airfoil_number,
            case_number=None,
            Mach=None,
            AoA=None,
            Re=None,
            field_name="Velocity",
            vel_component='a',
            block_index=2,
            action="plot_field",
            **kwargs
        ):
            """
            Locate and process CGNS surface data for a given airfoil and case.
            """
        
            # Step 1: Load CSV
            if not os.path.exists(self.translam_csv):
                print(f"[unifoil] ❌ CSV file not found: {self.translam_csv}")
                return None
        
            df = pd.read_csv(self.translam_csv)
        
            # Determine case number
            if case_number is None:
                if Mach is None or AoA is None or Re is None:
                    print("[unifoil] ❌ Must specify either (airfoil, case_number) or (airfoil, Mach, AoA, Re).")
                    return None
        
                subset = df[df["airfoil"] == airfoil_number]
                if subset.empty:
                    print(f"[unifoil] ❌ Airfoil {airfoil_number} not found in CSV.")
                    return None
        
                dist = np.sqrt(
                    (subset["Mach"] - Mach) ** 2 +
                    (subset["AoA"] - AoA) ** 2 +
                    ((subset["Re"] - Re) / subset["Re"].max()) ** 2
                )
                best_idx = dist.idxmin()
                row = subset.loc[best_idx]
                case_number = int(row["case"])
                Mach, AoA, Re = row["Mach"], row["AoA"], row["Re"]
                print(f"[unifoil] Closest match → Airfoil {airfoil_number}, Case {case_number} "
                    f"(Mach={Mach:.3f}, AoA={AoA:.3f}, Re={Re:.2e})")
            else:
                row = df[(df["airfoil"] == airfoil_number) & (df["case"] == case_number)]
                if row.empty:
                    print(f"[unifoil] ❌ No entry found for Airfoil {airfoil_number}, Case {case_number}.")
                    return None
                Mach, AoA, Re = row["Mach"].values[0], row["AoA"].values[0], row["Re"].values[0]
        
            # Step 2: Find CGNS file
            case_index = case_number - 1  # ✅ 1-indexed → 0-indexed
            filename_pattern = f"airfoil_{airfoil_number}_G2_A_L0_case_{case_index}_000_surf_lam.cgns"
            found_path = None

            #for i in range(1, 7):
            #    cutout_folder = os.path.join(self.cwd, f"Turb_Cutout_{i}")
            #    if not os.path.isdir(cutout_folder):
            #        continue
            #    candidate_path = os.path.join(cutout_folder, filename_pattern)
            #    if os.path.exists(candidate_path):
            #        found_path = candidate_path
            #        break
            cutout_folder = os.path.join(self.cwd, "NLF_Airfoils_Fully_Turbulent")
            candidate_path = os.path.join(cutout_folder, filename_pattern)
            if os.path.exists(candidate_path):
                found_path = candidate_path


            if not found_path:
                print(f"[unifoil] ⚠️ CGNS file not found for Airfoil {airfoil_number}, Case {case_number}. "
                    f"(possibly missing or unconverged simulation)")
                return None
        
            print(f"[unifoil] ✅ Found CGNS file: {found_path}")
            #############################################################################################################
            '''
            # Step 3: Geometry file

            airfoil_file = None
            nlf_file_candidate = os.path.join(self.nlf_geom_path, f"airfoil_{airfoil_number:03d}.dat")
        
            if os.path.exists(nlf_file_candidate):
                airfoil_file = nlf_file_candidate
            else:
                print(f"[unifoil] ⚠️ Airfoil geometry file not found for Airfoil {airfoil_number}.")
        
            
            '''
            #############################################################################################################
            '''
            Ideally we need to do it in the way it is in the commented block just above. But there was a filename mismatch.
            We will fix this in v2.
            '''

            airfoil_file = None
            matched_csv_path = os.path.join(self.cwd, "matched_files.csv")

            # ----------------------------------------------------------
            # Load mapping from matched_files.csv
            # ----------------------------------------------------------
            if not os.path.exists(matched_csv_path):
                print(f"[unifoil] ⚠️ matched_files.csv not found at {matched_csv_path}")
                mapped_airfoil = airfoil_number
            else:
                df_match = pd.read_csv(matched_csv_path)

                # Extract airfoil numbers from filenames
                def extract_number(name):
                    match = re.search(r"airfoil_(\d+)_G2", name)
                    return int(match.group(1)) if match else None

                df_match["old_airfoil"] = df_match["Matched_Grid_File"].apply(extract_number)
                df_match["new_airfoil"] = df_match["Extracted_File"].apply(extract_number)

                # Reverse mapping: new_airfoil → old_airfoil
                reverse_map = dict(zip(df_match["new_airfoil"], df_match["old_airfoil"]))
                mapped_airfoil = reverse_map.get(airfoil_number, airfoil_number)

            # ----------------------------------------------------------
            # Locate geometry file for the mapped airfoil
            # ----------------------------------------------------------
            nlf_file_candidate = os.path.join(self.nlf_geom_path, f"airfoil_{mapped_airfoil:03d}.dat")

            if os.path.exists(nlf_file_candidate):
                airfoil_file = nlf_file_candidate
            else:
                print(f"[unifoil] ⚠️ Airfoil geometry file not found for mapped Airfoil {mapped_airfoil} "
                    f"(expected: {nlf_file_candidate})")
            
            # Step 4: Execute requested action

            try:
                post = surfCGNSPostProcessor(found_path, airfoil_file=airfoil_file)
                if action == "plot_field":
                    post.plot_field(field_name=field_name, block_index=block_index, vel_component=vel_component)
                    return None
                elif hasattr(post, action):
                    func = getattr(post, action)
                    if callable(func):
                        # Always pass field_name and block_index if the function expects them
                        if "field_name" in func.__code__.co_varnames:
                            kwargs.setdefault("field_name", field_name)
                        if "block_index" in func.__code__.co_varnames:
                            kwargs.setdefault("block_index", block_index)
                        if "vel_component" in func.__code__.co_varnames:
                            kwargs.setdefault("vel_component", vel_component)
                
                        result = func(**kwargs)
                        return result

                    else:
                        print(f"[unifoil] ⚠️ '{action}' exists but is not callable.")
                        return None
                else:
                    print(f"[unifoil] ⚠️ Unknown action '{action}'. Available: "
                        f"{[m for m in dir(post) if not m.startswith('_')]}")
                    return None
            except Exception as e:
                print(f"[unifoil] ❌ Error running surfCGNSPostProcessor.{action}: {e}")
                return None

    def get_aero_coeffs_lam(self, airfoil_number, case_number, print_flag=True):
            """
            Extract CL and CD from airfoil_<num>_G2_A_L0_case_<num>_analysis.csv.
            Handles both 7-row and 8-row formats, converts text to floats,
            and converts user-provided 1-indexed case_number to 0-indexed filenames.
        
            Parameters
            ----------
            airfoil_number : int
                Airfoil number.
            case_number : int
                Case number (1-indexed, as in CSV).
            print_flag : bool, optional
                If True, prints Cl and Cd to console.
        
            Returns
            -------
            tuple : (Cl, Cd) or (None, None) if file not found.
            """
            import pandas as pd
            import numpy as np
            import os
        
            # Convert 1-based case number to 0-based for filename
            case_index = case_number - 1
            filename = f"airfoil_{airfoil_number}_G2_A_L0_case_{case_index}_analysis.csv"
        
            # Search both datasets
            found_path = None
            for folder in ["airfoil_data_from_simulations_lam"]:
                candidate = os.path.join(self.cwd, folder, filename)
                if os.path.exists(candidate):
                    found_path = candidate
                    break
        
            if not found_path:
                print(f"[unifoil] ⚠️ Analysis file not found for Airfoil {airfoil_number}, Case {case_number} (→ file case {case_index}).")
                return None, None
        
            try:
                df = pd.read_csv(found_path, header=None)
        
                # Locate CL/CD header
                header_row = None
                for i in range(len(df)):
                    vals = [str(v).strip().upper() for v in df.iloc[i, :2].values]
                    if "CL" in vals and "CD" in vals:
                        header_row = i
                        break
        
                # Extract row below CL/CD header or fallback to last row
                if header_row is not None and header_row + 1 < len(df):
                    Cl_raw, Cd_raw = df.iloc[header_row + 1, 0], df.iloc[header_row + 1, 1]
                else:
                    Cl_raw, Cd_raw = df.iloc[-1, 0], df.iloc[-1, 1]
        
                # Convert to float robustly
                try:
                    Cl = float(Cl_raw)
                    Cd = float(Cd_raw)
                except Exception:
                    Cl, Cd = pd.to_numeric([Cl_raw, Cd_raw], errors="coerce")
        
                if np.isnan(Cl) or np.isnan(Cd):
                    raise ValueError(f"Invalid numeric values: Cl={Cl_raw}, Cd={Cd_raw}")
        
                if print_flag:
                    print(f"[unifoil] ✅ Airfoil {airfoil_number}, Case {case_number} → File Case {case_index}: Cl = {Cl:.6f}, Cd = {Cd:.6f}")
                    print(f"[unifoil]    File: {found_path}")
        
                return Cl, Cd
        
            except Exception as e:
                print(f"[unifoil] ❌ Error reading {found_path}: {e}")
                return None, None
    
    def load_convergence_data_lam(self, airfoil_number, case_number, print_flag=False):
        """
        Load convergence pickle data for a given airfoil and case.
        Automatically converts user-provided 1-indexed case_number to 0-indexed filenames.

        Parameters
        ----------
        airfoil_number : int
            Airfoil number.
        case_number : int
            Case number (1-indexed, as seen in CSV).
        print_flag : bool, optional
            If True, prints a summary of the loaded contents.

        Returns
        -------
        dict or None
            Dictionary loaded from pickle, or None if file not found or unreadable.
        """
        import os
        import pickle
        import numpy as np

        # Convert 1-based to 0-based case index
        case_index = case_number - 1
        filename = f"airfoil_{airfoil_number}_G2_A_L0_case_{case_index}_convergence.pkl"

        # Search both dataset folders
        found_path = None
        for folder in ["airfoil_data_from_simulations_lam"]:
            candidate = os.path.join(self.cwd, folder, filename)
            if os.path.exists(candidate):
                found_path = candidate
                break

        if not found_path:
            print(f"[unifoil] ⚠️ Convergence file not found for Airfoil {airfoil_number}, Case {case_number} (→ file case {case_index}).")
            return None

        try:
            with open(found_path, "rb") as f:
                data = pickle.load(f)

            if print_flag:
                print(f"[unifoil] ✅ Loaded convergence data for Airfoil {airfoil_number}, Case {case_number} (→ file case {case_index}):")
                for key, val in data.items():
                    if isinstance(val, (list, tuple, np.ndarray)):
                        print(f"  {key}: array with {len(val)} elements")
                    else:
                        print(f"  {key}: {val}")

            return data

        except Exception as e:
            print(f"[unifoil] ❌ Error reading {found_path}: {e}")
            return None
    
    # NLF transi cases:

    def surf_transi(
            self,
            airfoil_number,
            case_number=None,
            Mach=None,
            AoA=None,
            Re=None,
            field_name="Velocity",
            vel_component='a',
            block_index=2,
            action="plot_field",
            **kwargs
        ):
            """
            Locate and process CGNS surface data for a given airfoil and case.
            """
        
            # Step 1: Load CSV
            if not os.path.exists(self.translam_csv):
                print(f"[unifoil] ❌ CSV file not found: {self.translam_csv}")
                return None
        
            df = pd.read_csv(self.translam_csv)
        
            # Determine case number
            if case_number is None:
                if Mach is None or AoA is None or Re is None:
                    print("[unifoil] ❌ Must specify either (airfoil, case_number) or (airfoil, Mach, AoA, Re).")
                    return None
        
                subset = df[df["airfoil"] == airfoil_number]
                if subset.empty:
                    print(f"[unifoil] ❌ Airfoil {airfoil_number} not found in CSV.")
                    return None
        
                dist = np.sqrt(
                    (subset["Mach"] - Mach) ** 2 +
                    (subset["AoA"] - AoA) ** 2 +
                    ((subset["Re"] - Re) / subset["Re"].max()) ** 2
                )
                best_idx = dist.idxmin()
                row = subset.loc[best_idx]
                case_number = int(row["case"])
                Mach, AoA, Re = row["Mach"], row["AoA"], row["Re"]
                print(f"[unifoil] Closest match → Airfoil {airfoil_number}, Case {case_number} "
                    f"(Mach={Mach:.3f}, AoA={AoA:.3f}, Re={Re:.2e})")
            else:
                row = df[(df["airfoil"] == airfoil_number) & (df["case"] == case_number)]
                if row.empty:
                    print(f"[unifoil] ❌ No entry found for Airfoil {airfoil_number}, Case {case_number}.")
                    return None
                Mach, AoA, Re = row["Mach"].values[0], row["AoA"].values[0], row["Re"].values[0]
        
            # Step 2: Find CGNS file
            case_index = case_number - 1  # ✅ 1-indexed → 0-indexed
            foldername = f"airfoil_{airfoil_number}_G2_A_L0_case_{case_index}"
            filename_pattern = f"{foldername}_000_surf.cgns"
            found_path = None

            for i in range(1, 5):
                cutout_folder = os.path.join(self.cwd, f"Transi_Cutout_{i}")
                if not os.path.isdir(cutout_folder):
                    continue
                candidate_path = os.path.join(cutout_folder, foldername, filename_pattern)
                if os.path.exists(candidate_path):
                    found_path = candidate_path
                    break

            if not found_path:
                print(f"[unifoil] ⚠️ CGNS file not found for Airfoil {airfoil_number}, Case {case_number}. "
                    f"(possibly missing or unconverged simulation)")
                return None
        
            print(f"[unifoil] ✅ Found CGNS file: {found_path}")
            #############################################################################################################
            '''
            # Step 3: Geometry file

            airfoil_file = None
            nlf_file_candidate = os.path.join(self.nlf_geom_path, f"airfoil_{airfoil_number:03d}.dat")
        
            if os.path.exists(nlf_file_candidate):
                airfoil_file = nlf_file_candidate
            else:
                print(f"[unifoil] ⚠️ Airfoil geometry file not found for Airfoil {airfoil_number}.")
        
            
            '''
            #############################################################################################################
            '''
            Ideally we need to do it in the way it is in the commented block just above. But there was a filename mismatch.
            We will fix this in v2.
            '''

            airfoil_file = None
            matched_csv_path = os.path.join(self.cwd, "matched_files.csv")

            # ----------------------------------------------------------
            # Load mapping from matched_files.csv
            # ----------------------------------------------------------
            if not os.path.exists(matched_csv_path):
                print(f"[unifoil] ⚠️ matched_files.csv not found at {matched_csv_path}")
                mapped_airfoil = airfoil_number
            else:
                df_match = pd.read_csv(matched_csv_path)

                # Extract airfoil numbers from filenames
                def extract_number(name):
                    match = re.search(r"airfoil_(\d+)_G2", name)
                    return int(match.group(1)) if match else None

                df_match["old_airfoil"] = df_match["Matched_Grid_File"].apply(extract_number)
                df_match["new_airfoil"] = df_match["Extracted_File"].apply(extract_number)

                # Reverse mapping: new_airfoil → old_airfoil
                reverse_map = dict(zip(df_match["new_airfoil"], df_match["old_airfoil"]))
                mapped_airfoil = reverse_map.get(airfoil_number, airfoil_number)

            # ----------------------------------------------------------
            # Locate geometry file for the mapped airfoil
            # ----------------------------------------------------------
            nlf_file_candidate = os.path.join(self.nlf_geom_path, f"airfoil_{mapped_airfoil:03d}.dat")

            if os.path.exists(nlf_file_candidate):
                airfoil_file = nlf_file_candidate
            else:
                print(f"[unifoil] ⚠️ Airfoil geometry file not found for mapped Airfoil {mapped_airfoil} "
                    f"(expected: {nlf_file_candidate})")
            
            # Step 4: Execute requested action

            try:
                post = surfCGNSPostProcessor(found_path, airfoil_file=airfoil_file)
                if action == "plot_field":
                    post.plot_field(field_name=field_name, block_index=block_index, vel_component=vel_component)
                    return None
                elif hasattr(post, action):
                    func = getattr(post, action)
                    if callable(func):
                        # Always pass field_name and block_index if the function expects them
                        if "field_name" in func.__code__.co_varnames:
                            kwargs.setdefault("field_name", field_name)
                        if "block_index" in func.__code__.co_varnames:
                            kwargs.setdefault("block_index", block_index)
                        if "vel_component" in func.__code__.co_varnames:
                            kwargs.setdefault("vel_component", vel_component)
                
                        result = func(**kwargs)
                        return result

                    else:
                        print(f"[unifoil] ⚠️ '{action}' exists but is not callable.")
                        return None
                else:
                    print(f"[unifoil] ⚠️ Unknown action '{action}'. Available: "
                        f"{[m for m in dir(post) if not m.startswith('_')]}")
                    return None
            except Exception as e:
                print(f"[unifoil] ❌ Error running surfCGNSPostProcessor.{action}: {e}")
                return None
    
    def get_aero_coeffs_transi(self, airfoil_number, case_number, print_flag=True):
            """
            Extract CL and CD from airfoil_<num>_G2_A_L0_case_<num>_analysis.csv.
            Handles both 7-row and 8-row formats, converts text to floats,
            and converts user-provided 1-indexed case_number to 0-indexed filenames.
        
            Parameters
            ----------
            airfoil_number : int
                Airfoil number.
            case_number : int
                Case number (1-indexed, as in CSV).
            print_flag : bool, optional
                If True, prints Cl and Cd to console.
        
            Returns
            -------
            tuple : (Cl, Cd) or (None, None) if file not found.
            """
        
            # Convert 1-based case number to 0-based for filename
            case_index = case_number - 1
            filename = f"{airfoil_number}_case_{case_index}_analysis.csv"
        
            # Search both datasets
            found_path = None
            for folder in ["airfoil_data_from_simulations_transi"]:
                candidate = os.path.join(self.cwd, folder, filename)
                if os.path.exists(candidate):
                    found_path = candidate
                    break
        
            if not found_path:
                print(f"[unifoil] ⚠️ Analysis file not found for Airfoil {airfoil_number}, Case {case_number} (→ file case {case_index}).")
                return None, None
        
            try:
                df = pd.read_csv(found_path, header=None)
        
                # Locate CL/CD header
                header_row = None
                for i in range(len(df)):
                    vals = [str(v).strip().upper() for v in df.iloc[i, :2].values]
                    if "CL" in vals and "CD" in vals:
                        header_row = i
                        break
        
                # Extract row below CL/CD header or fallback to last row
                if header_row is not None and header_row + 1 < len(df):
                    Cl_raw, Cd_raw = df.iloc[header_row + 1, 0], df.iloc[header_row + 1, 1]
                else:
                    Cl_raw, Cd_raw = df.iloc[-1, 0], df.iloc[-1, 1]
        
                # Convert to float robustly
                try:
                    Cl = float(Cl_raw)
                    Cd = float(Cd_raw)
                except Exception:
                    Cl, Cd = pd.to_numeric([Cl_raw, Cd_raw], errors="coerce")
        
                if np.isnan(Cl) or np.isnan(Cd):
                    raise ValueError(f"Invalid numeric values: Cl={Cl_raw}, Cd={Cd_raw}")
        
                if print_flag:
                    print(f"[unifoil] ✅ Airfoil {airfoil_number}, Case {case_number} → File Case {case_index}: Cl = {Cl:.6f}, Cd = {Cd:.6f}")
                    print(f"[unifoil]    File: {found_path}")
        
                return Cl, Cd
        
            except Exception as e:
                print(f"[unifoil] ❌ Error reading {found_path}: {e}")
                return None, None
            
    def load_convergence_data_transi(self, airfoil_number, case_number, print_flag=False):
        """
        Load convergence pickle data for a given airfoil and case.
        Automatically converts user-provided 1-indexed case_number to 0-indexed filenames.

        Parameters
        ----------
        airfoil_number : int
            Airfoil number.
        case_number : int
            Case number (1-indexed, as seen in CSV).
        print_flag : bool, optional
            If True, prints a summary of the loaded contents.

        Returns
        -------
        dict or None
            Dictionary loaded from pickle, or None if file not found or unreadable.
        """

        # Convert 1-based to 0-based case index
        case_index = case_number - 1
        filename = f"{airfoil_number}_case_{case_index}_convergence.pkl"

        # Search both dataset folders
        found_path = None
        for folder in ["airfoil_data_from_simulations_transi"]:
            candidate = os.path.join(self.cwd, folder, filename)
            if os.path.exists(candidate):
                found_path = candidate
                break

        if not found_path:
            print(f"[unifoil] ⚠️ Convergence file not found for Airfoil {airfoil_number}, Case {case_number} (→ file case {case_index}).")
            return None

        try:
            with open(found_path, "rb") as f:
                data = pickle.load(f)

            if print_flag:
                print(f"[unifoil] ✅ Loaded convergence data for Airfoil {airfoil_number}, Case {case_number} (→ file case {case_index}):")
                for key, val in data.items():
                    if isinstance(val, (list, tuple, np.ndarray)):
                        print(f"  {key}: array with {len(val)} elements")
                    else:
                        print(f"  {key}: {val}")

            return data

        except Exception as e:
            print(f"[unifoil] ❌ Error reading {found_path}: {e}")
            return None        
    
    def get_supplement_transi(
        self,
        airfoil_number,
        case_number=None,
        Mach=None,
        AoA=None,
        Re=None,
        plot_flag=True
        ):
        """
        Locate and process supplementary transition data (nfactor_ts.dat, transiLoc.dat)
        for a given airfoil and case using sup_transi.
    
        Searches folders:
            Transi_sup_data_Cutout_i/airfoil_<num>_G2_A_L0_case_<case>
    
        Parameters
        ----------
        airfoil_number : int
            Airfoil number.
        case_number : int, optional
            Case number (1-indexed as in CSV).
        Mach, AoA, Re : optional
            Used to find the closest case if case_number is not given.
        """

    
        # Step 1 — Load CSV to locate the case
        if not os.path.exists(self.translam_csv):
            print(f"[unifoil] ❌ CSV file not found: {self.translam_csv}")
            return None
    
        df = pd.read_csv(self.translam_csv)
    
        if case_number is None:
            if Mach is None or AoA is None or Re is None:
                print("[unifoil] ❌ Must specify either (airfoil, case_number) or (airfoil, Mach, AoA, Re).")
                return None
    
            subset = df[df["airfoil"] == airfoil_number]
            if subset.empty:
                print(f"[unifoil] ❌ Airfoil {airfoil_number} not found in CSV.")
                return None
    
            dist = np.sqrt(
                (subset["Mach"] - Mach) ** 2 +
                (subset["AoA"] - AoA) ** 2 +
                ((subset["Re"] - Re) / subset["Re"].max()) ** 2
            )
            best_idx = dist.idxmin()
            row = subset.loc[best_idx]
            case_number = int(row["case"])
            Mach, AoA, Re = row["Mach"], row["AoA"], row["Re"]
            print(f"[unifoil] Closest match → Airfoil {airfoil_number}, Case {case_number} "
                  f"(Mach={Mach:.3f}, AoA={AoA:.3f}, Re={Re:.2e})")
    
        case_index = case_number - 1
        foldername = f"airfoil_{airfoil_number}_G2_A_L0_case_{case_index}"
    
        # Step 2 — Map airfoil number (NLF geometry mapping)
        airfoil_file = None
        matched_csv_path = os.path.join(self.cwd, "matched_files.csv")
    
        if not os.path.exists(matched_csv_path):
            print(f"[unifoil] ⚠️ matched_files.csv not found at {matched_csv_path}")
            mapped_airfoil = airfoil_number
        else:
            df_match = pd.read_csv(matched_csv_path)
    
            def extract_number(name):
                match = re.search(r"airfoil_(\d+)_G2", name)
                return int(match.group(1)) if match else None
    
            df_match["old_airfoil"] = df_match["Matched_Grid_File"].apply(extract_number)
            df_match["new_airfoil"] = df_match["Extracted_File"].apply(extract_number)
            reverse_map = dict(zip(df_match["new_airfoil"], df_match["old_airfoil"]))
            mapped_airfoil = reverse_map.get(airfoil_number, airfoil_number)
    
        nlf_file_candidate = os.path.join(self.nlf_geom_path, f"airfoil_{mapped_airfoil:03d}.dat")
        if os.path.exists(nlf_file_candidate):
            airfoil_file = nlf_file_candidate
        else:
            print(f"[unifoil] ⚠️ Geometry file not found for mapped Airfoil {mapped_airfoil} "
                  f"(expected: {nlf_file_candidate})")
    
        # Step 3 — Locate supplementary data (Transi_sup_data_Cutout_i)
        found_sup = False
        for i in range(1, 5):
            sup_folder = os.path.join(self.cwd, f"Transi_sup_data_Cutout_{i}", foldername)
            if not os.path.isdir(sup_folder):
                continue
    
            transiLoc_path = os.path.join(sup_folder, "transiLoc.dat")
            nfactor_path = os.path.join(sup_folder, "nfactor_ts.dat")
    
            if os.path.exists(transiLoc_path) or os.path.exists(nfactor_path):
                found_sup = True
                print(f"[unifoil] ✅ Found supplementary data in: {sup_folder}")
    
                transi_obj = sup_transi(
                    nfactor_file=nfactor_path if os.path.exists(nfactor_path) else "./missing_nfactor.dat",
                    airfoil_file=airfoil_file,
                    transiloc_file=transiLoc_path if os.path.exists(transiLoc_path) else "./missing_transiLoc.dat"
                )
                
                nfactor_data, transi_data = transi_obj.get_data()
    
                # Call available functions based on found files
                if plot_flag==True:
                    if os.path.exists(nfactor_path):
                        print("[unifoil] → Plotting N-factor data...")
                        transi_obj.plot_nfactor()
                    if os.path.exists(transiLoc_path):
                        print("[unifoil] → Plotting transition locations...")
                        transi_obj.plot_airfoil_with_transition()
                break
    
        if not found_sup:
            print(f"[unifoil] ⚠️ No supplementary transition data found for Airfoil {airfoil_number}, Case {case_number}.")
        
        return [nfactor_data, transi_data]
