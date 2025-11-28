#!/usr/bin/env python3
"""
Convenience wrapper to run the FT geometry generator from the repo root.
"""

import argparse
import os
import sys

from unifoil.geometry.ft_geometry_gen import AirfoilFTGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="Generate FT airfoil geometry .dat files.")
    parser.add_argument(
        "--basis",
        default="input_ft/basis.txt",
        help="Path to basis.txt (default: input_ft/basis.txt)",
    )
    parser.add_argument(
        "--training",
        default="input_ft/training.dat",
        help="Path to training.dat coefficients (default: input_ft/training.dat)",
    )
    parser.add_argument(
        "--validating",
        default="input_ft/validating.dat",
        help="Path to validating.dat coefficients (default: input_ft/validating.dat)",
    )
    parser.add_argument(
        "--output",
        default="airfoil_ft_geom",
        help="Output directory for generated .dat files (default: airfoil_ft_geom)",
    )
    return parser.parse_args()


def _require(path, description):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{description} not found: {path}")
    return os.path.abspath(path)


def main():
    args = parse_args()
    try:
        generator = AirfoilFTGenerator(
            basis_file=_require(args.basis, "basis file"),
            train_file=_require(args.training, "training coefficients"),
            valid_file=_require(args.validating, "validating coefficients"),
            output_folder=os.path.abspath(args.output),
        )
    except FileNotFoundError as exc:
        print(f"[run_ft_generator] ❌ {exc}")
        sys.exit(1)

    generator.generate()
    print(f"[run_ft_generator] ✅ Finished. Files saved to {generator.output_folder}")


if __name__ == "__main__":
    main()
