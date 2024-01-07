# Getting started

## Data access

  Because of the large data volumes of the CCIC data record, we are currently
  still searching for ways to distribute the data. In the mean time, one year of
  CCIC results can be accessed [through
  globus](https://app.globus.org/file-manager?origin_id=3e5cb4b1-25d1-4d3a-882b-8e06b631a92f&origin_path=%2F).


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

After setting up ``pansat`` like this, you should be able to run the ``ccic process`` command for CPCIR input data by simply replacing the ``gridsat`` argument with ``cpcir``:

``` shell
ccic process ccic.pckl gridsat results 2020-01-01T00:00:00 2020-01-02T00:00:00 --targets tiwp tiwc cloud_prob_2d cloud_prob_3d 
```
