# Dataset and file structure

## CPCIR and GridSat

CCIC is derived from two satellite-observations datasets: The GridSat-B1 dataset
{cite:p}`Knapp_2014_CDR` and the NCEP/CPC Merged IR dataset
{cite:p}`Janowiak_2017_CPCIR`. In the following, these two datasets will be
referred to as **GridSat** and **CPCIR**, respectively.

CCIC provides estimates for both the GridSat and the CPCIR datasets. The
estimates are provided on the same grid as the observations and thus inherit
their temporospatial resolution and coverage. With temporospatial resolution
of 3h @ 0.07 degree, the GridSat-based data offers lower resolution than
the CPCIR data, which has 30 min @ 0.036 degree resolution. However, GridSat
is available from 1980, whereas CPCIR only from 2000. The temporospatial coverage
and resolution is summarized in table {numref}`table %s <resolution_and_coverage>`.

```{list-table} Temporospatial coverage and resolution of the GridSat and CPCIR variants of CCIC
:header-rows: 1
:name: resolution_and_coverage

* - Dataset
  - Coverage
  - Spatial resolution
  - Temporal resolution
* - GridSat
  - 1980 - present
  - 0.07 degree
  - 3 h
* - CPCIR
  - 2000 - present
  - 0.04 degree
  - 30 min
``` 

## Data organization

The CCIC record is organized  into results derived from GridSat input data results derived from CPCIR input data. Below that files are organized into folder by year.

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

The files follow the naming pattern

```
ccic_{product}_{YYYYmmddHH}00.zarr
```
 where `{product}` is either `cpcir` or `gridsat`, and `YYYYmmddHH` is the timestamp. Each GridSat file contains the retrieval for one timestamp, and each CPCIR file contains the retrieval for two timestamps, the full hour and 30 minutes after the full hour. For each day, GridSat are available at hours  `HH` = 00, 03, 06, 09, 12, 15, 18, and 21, while CPCIR files are available at hours `HH` = 00, 01, ..., 22, 23.

## Variables

The CCIC climate data record provides esimtates of the total ice water path (TIWP) and a 2D cloud probability. The data files follow  CF conventions. The variables and their meaning are listed in {numref}`table %s <variables>`. 

```{list-table} CCIC Variables and their significance 
:header-rows: 1
:name: variables

* - Variable name 
  - Units
  - Range
  - Description
* - `tiwp`            
  - kg m<sup>-2</sup> 
  - ≥ 0 
  - Vertically-integrated concentration of frozen hydrometeors
* - `tiwp_ci`        
  - kg m<sup>-2</sup> 
  - ≥ 0  
  - 90% confidence interval for the retrieved TIWP
* - `p_tiwp`
  - -                  
  - [0, 1]
  - Probability that `tiwp` exceeds 10<sup>-3</sup> kg m<sup>-2</sup>
* - `cloud_prob_2d` 
  - -
  - [0, 1]
  - Probability of presence of a cloud anywhere in the atmosphere
* - `inpainted`
  - -
  - {0, 1}
  - Input pixel was NaN; the retrieval can be a numeric value (inpainted) 
```


These variables are gridded on the coordinates from the table below, where the spatial grid and the time resolution are constant for each input product.


```{list-table} Coordinate conventions used by CCIC
:header-rows: 1
:name: coordinates
* - Coordinate name
  - Units
  - Range                                     
  - Description 
* - `latitude`
  - Degrees North
  - CPCIR: (-60, +60)<br/>GridSat: [-70, +70) 
  - Latitude
* - `longitude`
  - Degrees East
  - [-180, +180)
  - Longitude
* - `time` 
  - ns
  - [numpy's `datetime64[ns]`](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.datetime64)<br/>(nanoseconds since Unix epoch)|
  -  Nominal retrieval time
```

