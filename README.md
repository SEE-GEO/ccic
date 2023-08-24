# The Chalmers Cloud Ice Climatology

This repository contains the all source code for the training and
processing of the retrieval underlying Chalmers
Cloud Ice Climatology (CCIC) retrieval and data records.

## Installation

The currently recommended way to install CCIC is by cloning the source code,
creating a CCIC conda environment and the installing the package using ``pip``.

### Obtain the source code

Clone the git repository to obtain the CCIC source code.

``` shell
git clone https://github.com/see-geo/ccic
cd ccic
```

### Create conda environment

Create the ``ccic`` conda environment defined in ``ccic.yml``. The conda environment contains all of CCIC's depencies.

``` shell
conda env create -f ccic.yml
conda activate ccic
```

### Install package

Finally, install the ``ccic`` Python package using ``pip``. This will also install the ``ccic`` command, which can be used to run the retrieval.

``` shell
pip install .
```

## Running retrievals

In addition to installing the ``ccic`` package, running retrievals requires downloading the neural network model and setting up the ``pansat`` package.

### Download the neural network model

Download the CCIC neural network model from [Zenodo](https://zenodo.org/record/8277983/files/ccic.pckl?download=1).

### Running the retrieval

The following command can then be used to run the retrieval for all available input files from 2020-01-01 00:00:00 UTC to 2020-01-01 01:00:00.

``` shell
ccic process ccic.pckl gridsat results 2020-01-01T00:00:00 2020-01-01T00:00:00 --targets tiwp tiwc cloud_prob_2d cloud_prob_3d 
```

Additional options available for the ``ccic process`` can be listed using


``` shell
ccic process --help
```

### Setup ``pansat``

``ccic`` relies on the ``pansat`` package to outmatically download reuqired input data. To automatically download the CPCIR input data, you need to provide ``pansat`` with your credentials for the NASA GES DISC server.


> NOTE: We recommend creating a new account with throw-away credential for this purpose.

After setting up your account you can configure ``pansat`` as follows:

``` shell
pansat --add "GES DISC" <username>
```

Pansat will then first prompt you to setup a password for ``pansat``. ``pansat`` will use this password to encrypt the loging data it stores for different data portals. Following this, it will query your password for the NASA GES DISC server. After setting up ``pansat`` like this, you should be able to run the ``ccic process`` command for CPCIR input data by simply replacing the ``gridsat`` argument with ``cpcir``.


