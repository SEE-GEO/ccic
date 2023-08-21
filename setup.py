from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="ccic",
    version="0.0",
    description="Chalmers Cloud Ice Climatology",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/see-geo/ccic",
    author="Simon Pfreundschuh",
    author_email="simon.pfreundschuh@chalmers.se",
    install_requires=[
        "metpy",
        "numpy",
        "quantnn>=0.0.5",
        "torch",
        "pytorch-lightning",
        "pansat",
        "xarray",
        "zarr",
        "netCDF4",
        "dask",
        "artssat @ git+https://github.com/simonpf/artssat.git"
    ],
    packages=find_packages(),
    python_requires=">=3.8",
    project_urls={
        "Source": "https://github.com/see-geo/ccic/",
    },
    include_package_data=True,
    package_data={
        "ccic" : [
            "files/ccic.mplstyle"
        ]
    },
    entry_points = {
        'console_scripts': ['ccic=ccic.bin:ccic'],
    },
)
