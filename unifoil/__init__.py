import os
import runpy

from .getdata import GetData

def gen_ft():
    """
    Generate airfoil_ft_geom folder in the current working directory.
    Calls ft_geometry_gen.py from the geometry module.
    """
    script_path = os.path.join(os.path.dirname(__file__), "geometry", "ft_geometry_gen.py")
    print(f"[unifoil] Running FT geometry generator from: {script_path}")
    runpy.run_path(script_path, run_name="__main__")


def gen_nlf():
    """
    Generate airfoil_nlf_geom folder in the current working directory.
    Calls nlf_geometry_gen.py from the geometry module.
    """
    script_path = os.path.join(os.path.dirname(__file__), "geometry", "nlf_geometry_gen.py")
    print(f"[unifoil] Running NLF geometry generator from: {script_path}")
    runpy.run_path(script_path, run_name="__main__")


__all__ = ["gen_ft", "gen_nlf", "GetData"]
