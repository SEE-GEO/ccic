# Getting started

## Data access

The CCIC data record is available at the Registry of Open Data on AWS: https://registry.opendata.aws/ccic/.

## Reading CCIC data


Reading CCIC files in ``.zarr`` format requires the ``ccic`` Python package to be installed and imported. The recommended way to install ``ccic`` is using ``pip``:

```bash
pip install ccic
```

Then, CCIC data files can be read using xarray. 

```
import ccic # Required prior to reading CCIC .zarr files.
import xarray as xr

data = xr.open_zarr("ccic_gridsat_xxxxxxxxxxxx.zarr")
```


The Zarr format allows to store the data arrays in a compressed format suitable for distributed environments and cloud computing. In practice, this means that you do not need to download the full file to access a subset of the data, if you are accessing it remotely.

The CCIC data uses a custom compression codec to encode the numeric data, because it helps to save an enormous amount of disk space. This is why importing the ``ccic`` package is required prior to loading CCIC data in Zarr format.

```{note}
If you forget to import ``ccic`` prior to reading a file in Zarr format you will encounter the exception `ValueError: codec not available: 'log_bins'`. To register the necessary codec, please import the CCIC Python package `ccic` in every script where you load CCIC data files.
```

The code above also shows how we recommend opening the CCIC Zarr files, which is
using the [xarray](https://docs.xarray.dev) Python library. Xarray will do a
lazy loading of the CCIC file into an
[xarray.Dataset](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html).
Using lazy loading helps to work with large datasets, since several operations
can be executed without loading the full dataset, thus saves memory and network
traffic, if accessed remotely. There are multiple guides online on how to work
with xarray objects.

With xarray [you can also access Zarr files stored in cloud storage buckets](https://docs.xarray.dev/en/stable/user-guide/io.html#cloud-storage-buckets). For instance, you can load data from the [CCIC S3 bucket](https://registry.opendata.aws/ccic) with:

```python
import ccic
import xarray as xr

ds = xr.open_zarr(
    's3://chalmerscloudiceclimatology/record/gridsat/2021/ccic_gridsat_202101010000.zarr',
    storage_options={"anon": True},
    consolidated=True
)
# Load `tiwp` into memory
ds.tiwp.load()
# Do stuff with ds.tiwp...
```

## CCIC processing and development

Advanced use cases of CCIC include running retrievals or extending the ``ccic``
package. Both of these use cases have additional requirements. These can be
install using

```bash
pip install ccic[complete]
```

### Running retrievals

In order to run CCIC retrievals, you will first need to download the retrieval
 model from
 [Zenodo](https://zenodo.org/record/8277983/files/ccic.pckl?download=1).
 
#### Processing GridSat B1 input

Running retrievals on GridSat B1 data requires no further configuration. The command below demonstrates how to run retrievals  for the 1 January 2020.

``` shell
ccic process ccic.pckl gridsat results 2020-01-01T00:00:00 2020-01-02T00:00:00 --targets tiwp tiwc cloud_prob_2d cloud_prob_3d 
```

Additional options available for the ``ccic process`` command can be listed using
``` shell
ccic process --help
```

#### Processing CPCIR data

Processing CPCIR input data requires setting up the ``pansat`` package to
autmatically download the required input data. This requires providing
``pansat`` with your credentials for the [NASA GES DISC server](https://disc2.gesdisc.eosdis.nasa.gov/data/MERGED_IR/GPM_MERGIR.1/).


> **NOTE:** We recommend creating a new account with throw-away credential for this purpose.

After setting up your account you can configure ``pansat`` as follows:

``` shell
pansat --add "GES DISC" <username>
```

Pansat will then first prompt you to setup a password for ``pansat``. ``pansat`` will use this password to encrypt the log-in data it stores for different data portals. Following this, it will query your password for the NASA GES DISC server. 

After setting up ``pansat`` like this, you should be able to run the ``ccic process`` command for CPCIR input data by simply replacing the ``cpcir`` argument with ``gridsat``:

``` shell
ccic process ccic.pckl gridsat results 2020-01-01T00:00:00 2020-01-02T00:00:00 --targets tiwp tiwc cloud_prob_2d cloud_prob_3d 
```

To avoid having to enter your pansat password every time when you want to run a retrieval, you can set the ``PANSAT_PASSWORD`` environment variable to your password.

### Reproducing validation retrievals

The ``ccic.validation`` sub-module contains the implementation of the radar retrieval used to validate the CCIC retrievals. The validation code relies on the ``artssat`` package, which will need to be installed to use this functionality. The required package can be installed using

``` shell
pip install pip install git+ssh://git@github.com/simonpf/artssat
```
