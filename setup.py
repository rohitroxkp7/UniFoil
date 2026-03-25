from setuptools import setup, find_packages

setup(
    name="unifoil",
    version="0.2.0",
    description="Unified airfoil geometry generators (FT & NLF)",
    author="Rohit Kanchi",
    packages=find_packages(include=["unifoil", "unifoil.*"]),
    include_package_data=True,
    package_data={"unifoil": ["geometry/*.py"]},
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "matplotlib",
        "pandas",
        "pyvista",
        "niceplots",
        "mdolab-baseclasses",
        "docker",
    ],
)
