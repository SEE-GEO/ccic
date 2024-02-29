import codecs
from pathlib import Path
from setuptools import setup, find_packages

here = Path(__file__).parent.absolute()
with open(here / "README.md", encoding="utf-8") as f:
    long_description = f.read()


def read(path):
    """
    Read file from a path relative to this file.

    Args:
        path: Path of a file specified relative to the directory containing
            this setup.py.

    Return:
        A string containing the content of the file.
    """
    with codecs.open(here / path) as fp:
        return fp.read()


def get_version(path):
    """
    Extract version value from a given source file.

    Args:
        path: Relative path of the file from which to read the __version__
            attribute.

    Return:
        A string representing the version value.
    """
    for line in read(path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    return None


setup(
    name="ccic",
    version=get_version("ccic/__init__.py"),
    description="Chalmers Cloud Ice Climatology",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/see-geo/ccic",
    author="Simon Pfreundschuh",
    author_email="simon.pfreundschuh@chalmers.se",
    install_requires=[
        "zarr",
        "netCDF4",
        "dask",
        "beautifulsoup4",
        "lxml",
        "xarray",
    ],
    extras_require={
        "complete": [
            "metpy",
            "numpy",
            "quantnn>=0.0.5",
            "torch==1.13.1",
            "torchvision==0.14.1",
            "pytorch-lightning",
            "pyresample",
            "h5py",
            "pansat",
            "xarray",
            "netCDF4",
            "dask",
            "beautifulsoup4",
            "lxml",
            "jupyter-book",
        ]
    },
    packages=find_packages(),
    python_requires=">=3.8",
    project_urls={
        "Source": "https://github.com/see-geo/ccic/",
    },
    include_package_data=True,
    package_data={"ccic": ["files/ccic.mplstyle"]},
    entry_points={
        "console_scripts": ["ccic=ccic.bin:ccic"],
    },
)
