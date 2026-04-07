import csv
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional
import urllib.request
import urllib.error
from zipfile import ZipFile
import pandas as pd


@dataclass(frozen=True)
class DownloadSpec:
    """Describes a single remote file that should be fetched."""

    url: str
    filename: str
    description: str
    convert_tsv_to_csv: bool = False


class GetData:
    """
    Helper to download and stage UniFoil data assets.

    The class focuses on automation tasks such as fetching CSV metadata files
    that are mandatory for the rest of the tooling to run.
    """

    # Minimal manifest - expand as we identify additional compulsory assets.
    COMPULSORY_FILES: List[DownloadSpec] = [
        DownloadSpec(
            url="https://dataverse.harvard.edu/api/access/datafile/11478347",
            filename="Airfoil_Case_Data_Trans_Lam.csv",
            description="Transition/laminar case metadata",
            convert_tsv_to_csv=True,
        ),
        DownloadSpec(
            url="https://dataverse.harvard.edu/api/access/datafile/11478352",
            filename="Airfoil_Case_Data_turb.csv",
            description="Fully turbulent case metadata",
            convert_tsv_to_csv=True,
        ),
        DownloadSpec(
            url="https://dataverse.harvard.edu/api/access/datafile/13185746",
            filename="airfoil_ft_geom.zip",
            description="FT airfoil geometry archive",
        ),
        DownloadSpec(
            url="https://dataverse.harvard.edu/api/access/datafile/13185748",
            filename="airfoil_nlf_geom.zip",
            description="NLF airfoil geometry archive",
        ),
        DownloadSpec(
            url="https://dataverse.harvard.edu/api/access/datafile/13185745",
            filename="input_ft.zip",
            description="FT generator inputs (basis/training/validation)",
        ),
        DownloadSpec(
            url="https://dataverse.harvard.edu/api/access/datafile/13185744",
            filename="input_nlf.zip",
            description="NLF generator inputs (coefs/modes/xslice)",
        ),
        DownloadSpec(
            url="https://huggingface.co/datasets/rkanchi/UniFoil/resolve/main/sample_cutout_turb.zip?download=true",
            filename="sample_cutout_turb.zip",
            description="Sample turbulent surface cutout CGNS files",
        ),
    ]
    SAMPLE_FILES: List[DownloadSpec] = [
        DownloadSpec(
            url="https://dataverse.harvard.edu/api/access/datafile/13197104",
            filename="Turb_Cutout_1.zip",
            description="Sample turbulent cutout",
        ),
        DownloadSpec(
            url="https://dataverse.harvard.edu/api/access/datafile/13197103",
            filename="Transi_sup_data_Cutout_1.zip",
            description="Sample transition supplemental data",
        ),
        DownloadSpec(
            url="https://dataverse.harvard.edu/api/access/datafile/13197101",
            filename="Transi_Cutout_1.zip",
            description="Sample transitional cutout",
        ),
        DownloadSpec(
            url="https://dataverse.harvard.edu/api/access/datafile/13197102",
            filename="NLF_Airfoils_Fully_Turbulent.zip",
            description="Sample NLF airfoils (fully turbulent)",
        ),
        DownloadSpec(
            url="https://dataverse.harvard.edu/api/access/datafile/13197107",
            filename="airfoil_data_from_simulations_lam.zip",
            description="Sample laminar analysis files",
        ),
        DownloadSpec(
            url="https://dataverse.harvard.edu/api/access/datafile/13197106",
            filename="airfoil_data_from_simulations_transi.zip",
            description="Sample transitional analysis files",
        ),
        DownloadSpec(
            url="https://dataverse.harvard.edu/api/access/datafile/13197105",
            filename="airfoil_data_from_simulations_turb_set1.zip",
            description="Sample turbulent analysis files",
        ),
    ]

    def __init__(self, data_root: Optional[str] = None):
        self.data_root = Path(data_root) if data_root else Path(os.getcwd())
        self.data_root.mkdir(parents=True, exist_ok=True)
        self._global_index_files = {
            "Airfoil_Case_Data_Trans_Lam.csv",
            "Airfoil_Case_Data_turb.csv",
        }

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def getdata(
        self,
        flag: str = "compulsory",
        overwrite: bool = False,
        extra_items: Optional[Iterable[DownloadSpec]] = None,
    ) -> List[Path]:
        """
        Download UniFoil assets according to the requested flag.

        Parameters
        ----------
        flag : {"compulsory", "sample", "full"}
            Controls which built-in manifest to use.
        overwrite : bool
            If True, existing files will be replaced.
        extra_items : Iterable[DownloadSpec], optional
            Additional assets to download alongside the built-in set.

        Returns
        -------
        list[pathlib.Path]
            Paths to the downloaded (or already existing) files.
        """
        flag_normalized = (flag or "compulsory").lower()
        if flag_normalized == "compulsory":
            manifest = list(self.COMPULSORY_FILES)
        elif flag_normalized == "sample":
            manifest = list(self.COMPULSORY_FILES) + list(self.SAMPLE_FILES)
        elif flag_normalized == "full":
            raise NotImplementedError("[getdata] flag='full' is not implemented yet.")
        else:
            raise ValueError(f"[getdata] Unknown flag '{flag}'. Use 'compulsory', 'sample', or 'full'.")

        if extra_items:
            manifest.extend(extra_items)

        saved_paths: List[Path] = []
        start_time = time.perf_counter()
        for spec in manifest:
            target = self.data_root / spec.filename
            is_zip = target.suffix.lower() == ".zip"
            extracted_dir = target.with_suffix("") if is_zip else None

            if target.exists() and not overwrite:
                print(f"[getdata] Skipping '{target.name}' (already exists).")
                if target.name in self._global_index_files:
                    self._ensure_global_index(target)
                saved_paths.append(target)
                continue
            if is_zip and extracted_dir and extracted_dir.exists() and not overwrite:
                print(f"[getdata] Skipping '{target.name}' download (found extracted folder '{extracted_dir.name}').")
                saved_paths.append(extracted_dir)
                continue

            try:
                downloaded_path = self._download_to_temp(spec.url, spec.description)
            except Exception as exc:  # pragma: no cover - network failure path
                print(f"[getdata] Failed downloading '{spec.description}': {exc}")
                continue

            final_path = target
            try:
                if spec.convert_tsv_to_csv:
                    final_path = self._convert_tsv_to_csv(downloaded_path, target)
                else:
                    self._move_into_place(downloaded_path, target)
                print(f"[getdata] Saved '{final_path.name}'.")

                if final_path.name in self._global_index_files:
                    self._ensure_global_index(final_path)

                if final_path.suffix.lower() == ".zip":
                    extracted_path = self._extract_zip(final_path)
                    saved_paths.append(extracted_path)
                    print(f"[getdata] Extracted '{final_path.name}' to '{extracted_path.name}'.")
                    final_path.unlink(missing_ok=True)
                else:
                    saved_paths.append(final_path)
            finally:
                if downloaded_path.exists():
                    downloaded_path.unlink(missing_ok=True)

        elapsed = time.perf_counter() - start_time
        print(f"[getdata] Completed downloads in {elapsed:.1f} seconds.")
        return saved_paths

    def get_compulsory(
        self,
        overwrite: bool = False,
        extra_items: Optional[Iterable[DownloadSpec]] = None,
    ) -> List[Path]:
        """
        Backwards-compatible alias for getdata(flag="compulsory").
        """
        print("[getdata] get_compulsory() is deprecated; use getdata(flag='compulsory').")
        return self.getdata(flag="compulsory", overwrite=overwrite, extra_items=extra_items)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _download_to_temp(self, url: str, label: str) -> Path:
        """
        Stream the remote asset into a temporary file.
        """
        tmp_fd, tmp_path = tempfile.mkstemp(prefix="unifoil_", suffix=".download")
        tmp_file = Path(tmp_path)
        os.close(tmp_fd)

        print(f"[getdata] Downloading {label} ...")
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; UniFoil-GetData/1.0)",
                "Accept": "*/*",
            },
        )
        try:
            with urllib.request.urlopen(request) as response, open(tmp_file, "wb") as dst:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
        except urllib.error.URLError as exc:
            tmp_file.unlink(missing_ok=True)
            raise RuntimeError(f"Download error: {exc}") from exc

        return tmp_file

    def _convert_tsv_to_csv(self, source_path: Path, target_path: Path) -> Path:
        """
        Convert a TSV file to CSV format.
        """
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(source_path, "r", encoding="utf-8") as src, open(target_path, "w", newline="", encoding="utf-8") as dst:
            reader = csv.reader(src, delimiter="\t")
            writer = csv.writer(dst)
            for row in reader:
                writer.writerow(row)
        return target_path

    @staticmethod
    def _move_into_place(source_path: Path, target_path: Path) -> None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.replace(target_path)

    @staticmethod
    def _extract_zip(zip_path: Path) -> Path:
        destination = zip_path.with_suffix("")
        destination.mkdir(parents=True, exist_ok=True)
        with ZipFile(zip_path, "r") as archive:
            archive.extractall(destination)
        entries = list(destination.iterdir())
        if len(entries) == 1 and entries[0].is_dir():
            nested_dir = entries[0]
            for item in nested_dir.iterdir():
                shutil.move(str(item), destination)
            nested_dir.rmdir()
        return destination

    @staticmethod
    def _ensure_global_index(csv_path: Path) -> None:
        """
        Ensure a global_index column exists with 1..N in row order.
        """
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:  # pragma: no cover - IO/parsing errors
            print(f"[getdata] ⚠️ Could not read {csv_path.name} to add global_index: {exc}")
            return

        if "global_index" in df.columns:
            return

        df["global_index"] = range(1, len(df) + 1)
        try:
            df.to_csv(csv_path, index=False)
            print(f"[getdata] Added global_index to '{csv_path.name}'.")
        except Exception as exc:  # pragma: no cover - IO errors
            print(f"[getdata] ⚠️ Could not write {csv_path.name} with global_index: {exc}")
