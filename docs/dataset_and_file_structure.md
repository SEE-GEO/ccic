# Dataset and file structure

The dataset has been generated with the code available at https://github.com/SEE-GEO/ccic. In particular, only the vertically-integrated variables listed in the table below have been generated.

| Variable name     | Units             | Range | Description                                                           |
|-------------------|:-----------------:|:-----:|-----------------------------------------------------------------------|
| `tiwp`            | kg m<sup>-2</sup> | ≥ 0   | Vertically-integrated concentration of frozen hydrometeors            |
| `tiwp_ci`         | kg m<sup>-2</sup> | ≥ 0   | 90% confidence interval for the retrieved TIWP                        |
| `p_tiwp`          |                   | [0, 1]| Probability that `tiwp` exceeds 10<sup>-3</sup> kg m<sup>-2</sup>     |
| `cloud_prob_2d`   |                   | [0, 1]| Probability of presence of a cloud anywhere in the atmosphere         |
| `inpainted`       |                   | {0, 1}| Input pixel was NaN; the retrieval can be a numeric value (inpainted) |

These variables are gridded on the coordinates from the table below, where the spatial grid and the time resolution are constant for each input product.

| Coordinate name   | Units                                                                                                                                     | Range                                     | Description                       |
|-------------------|:-----------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------:|-----------------------------------|
| `latitude`        | degrees north                                                                                                                             | CPCIR: (-60, +60)<br/>GridSat: [-70, +70) | Latitude                          |
| `longitude`       | degrees east                                                                                                                              | [-180, +180)                              | Longitude                         |
| `time`            | [numpy's `datetime64[ns]`](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.datetime64)<br/>(nanoseconds since Unix epoch)|                                           | Timestamp of the cloud properties |

The main difference between the input products CPCIR and GridSat resides in the spatiotemporal resolution and their time coverage. Knapp et al. ([2011](https://doi.org/10.1175/2011BAMS3039.1)) mention that both products "attempt to reduce intersatellite differences by intersatellite normalization; however, GridSat also performs temporal normalization". The table, adapted from the [CCIC article](https://doi.org/10.5194/egusphere-2023-1953), displays the main characteristics of the CCIC data as a function of each product.

|                       | CPCIR                 | GridSat       |
|-----------------------|:---------------------:|:-------------:|
| Spatial resolution    | 0.036°                | 0.07°         |
| Temporal resolution   | 30 min                | 180 min       |
| Temporal coverage     | 2000 – 2023[^cpcir]   | 1980 – 2023   |
| Spatial coverage      | 60° S – 60° N         | 70° S – 70° N |

[^cpcir]: Note that CPCIR is not complete at the start of 2000.


Consequentally, the CCIC data available is organized by input product in the following structure:
```
.
├── cpcir
│   ├── 2000
│   ├── 2001
┆   ┆
│   ├── 2022
│   └── 2023
└── gridsat
    ├── 1980
    ├── 1981
    ┆
    ├── 2022
    └── 2023
```

Each folder contains the retrievals in the given year. The files follow this name pattern: `ccic_{product}_{YYYYmmddHH}00.zarr`, where `{product}` is either `cpcir` or `gridsat`, and `YYYYmmddHH` is the timestamp. Each GridSat file contains the retrieval for one timestamp, and each CPCIR file contains the retrieval for two timestamps, those contained in the hour of the filename: this matches how each input data product files are distributed. Consequently, GridSat files will have `HH` = 00, 03, 06, 09, 12, 15, 18, and 21, while CPCIR files will have `HH` = 00, 01, ..., 22, 23.

The files are stored as [Zarr](https://zarr.readthedocs.io/) files following the CF conventions (Eaton et al., [2022](https://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.html)). The Zarr format allows to store the data arrays in a compressed format suitable for distributed environments and cloud computing. In practice, this means that you do not need to download the full file to access a subset of the data, if you are accessing it remotely. The CCIC data uses a custom compression codec to encode the numeric data, because it helps to save an enormous amount of disk space. Therefore, to load the data, you will need to register this codec, otherwise you should get the exception `ValueError: codec not available: 'log_bins'`. To register the necessary codec, please import the CCIC Python package `ccic` in every script where you load CCIC data files even if you will not explicitly use the `ccic` package (if you have not installed it, see [here](https://github.com/SEE-GEO/ccic#installation) for install instructions). Here is an example:

```python
import ccic
import xarray as xr
ds = xr.open_zarr('/path/to/gridsat/2020/ccic_gridsat_202001010000.zarr')
```

The code above also shows how we recommend opening the CCIC Zarr files, which is using the [xarray](https://docs.xarray.dev) Python library. Xarray will do a lazy loading of the CCIC file into an [xarray.Dataset](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html). Using lazy loading helps to work with large datasets, since several operations can be executed without loading the full dataset, thus saves memory and network traffic, if accessed remotely. There are multiple guides online on how to work with xarray objects.

The xarray guide [shows how to access Zarr files stored in cloud storage buckets](https://docs.xarray.dev/en/stable/user-guide/io.html#cloud-storage-buckets). For instance, we have tested that the following works with a private Amazon S3 bucket (with user-bucket policies and AWS CLI and properly configured):

```python
import ccic
import xarray as xr
# Lazy load a CCIC file
ds = xr.open_zarr('s3://<private-bucket>/gridsat/2020/ccic_gridsat_202001010000.zarr')
# Load `cloud_prob_2d` into memory
ds.cloud_prob_2d.load()
# Do stuff with ds.cloud_prob_2d...
```
