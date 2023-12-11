# Getting started

> **TL;DR:** Reading CCIC ``.zarr`` files requires the ``ccic`` package to be installed and imported prior to opening the file.

## Reading CCIC data

Reading CCIC files in ``.zarr`` format requires the ``ccic`` Python package to be installed and imported. The recommended way to install ``ccic`` is using ``pip``:

```bash
pip install ccic
```

Then, CCIC data files can be read using xarray. 

```
import ccic # Required pripr to reading CCIC .zarr files.
import xarray as xr

data = xr.open_zarr("ccic_gridsat_xxxxxxxxxxxx.zarr")
```

## CCIC processing and development

Advanced use cases of CCIC include running retrievals or extending the ``ccic`` package. Both of these use cases have additional requirements. These can be install using

```bash
pip install ccic[complete]
```
