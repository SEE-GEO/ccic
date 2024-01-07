# The Chalmers Cloud Ice Climatology

This repository contains the all source code for the training and
processing of the retrieval underlying Chalmers
Cloud Ice Climatology (CCIC) retrieval and data records.

If you only need to install the `ccic` package to read CCIC data follow the minimal installation below.

## Installation

Currently it is recommended to install the ccic package using `pip`.

### Minimal installation (to read CCIC data)

```shell
pip install git+https://github.com/SEE-GEO/ccic.git
```

### Advanced installation (to run retrievals)

1. Clone the source code

    ``` shell
    git clone https://github.com/SEE-GEO/ccic
    cd ccic
    ```

2. Create a conda environment using [``ccic.yml``](ccic.yml)

   ```shell
   conda env create -f ccic.yml
   conda activate ccic
   ```

3. Install ``ccic`` with the `[complete]` option.

    ``` shell
    pip install .[complete]
    ```

    If you want to be able to use any updates without re-installing ``ccic`` or you are planning to contribute, use instead `pip install -e .[complete]`.

## Running retrievals

In addition to installing the ``ccic`` package, running retrievals requires downloading the neural network model and setting up the ``pansat`` package.

### Download the neural network model

Download the CCIC neural network model from [Zenodo](https://zenodo.org/record/8277983/files/ccic.pckl?download=1).

### Running the retrieval

The following command can then be used to run the retrieval for all available GridSat files from 2020-01-01 00:00:00 UTC to 2020-01-01 01:00:00 UTC.

``` shell
ccic process ccic.pckl gridsat results 2020-01-01T00:00:00 2020-01-01T00:00:00 --targets tiwp tiwc cloud_prob_2d cloud_prob_3d 
```

Additional options available for the ``ccic process`` can be listed using


``` shell
ccic process --help
```

### Setup ``pansat``

``ccic`` relies on the [``pansat``](https://github.com/SEE-GEO/pansat) package to automatically download required input data. To automatically download the CPCIR input data, you need to provide ``pansat`` with your credentials for the NASA GES DISC server.


> NOTE: We recommend creating a new account with throw-away credential for this purpose.

After setting up your account you can configure ``pansat`` as follows:

``` shell
pansat --add "GES DISC" <username>
```

Pansat will then first prompt you to setup a password for ``pansat``. ``pansat`` will use this password to encrypt the log-in data it stores for different data portals. Following this, it will query your password for the NASA GES DISC server. After setting up ``pansat`` like this, you should be able to run the ``ccic process`` command for CPCIR input data by simply replacing the ``gridsat`` argument with ``cpcir``.

GridSat data does not require any credentials.
