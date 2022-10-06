"""
Tests for the ccic.data.cloudsat module.
"""
from pathlib import Path

import dask.array as da
import numpy as np
from pyresample.bucket import BucketResampler

from ccic.data.gpm_ir import GPMIR, GPM_IR_GRID
from ccic.data.cloudsat import CloudSat2CIce, get_sample_indices

TEST_DATA = Path("/home/simonpf/data_3/ccic/test")

def test_random_resampling():

    gpm_file = "merg_2008020110_4km-pixel.nc4"
    gpm_data = GPMIR(TEST_DATA / gpm_file).to_xarray_dataset()
    cs_file = "2008032025505_09375_CS_2C-ICE_GRANULE_P1_R05_E02_F00.hdf"
    cs_data = CloudSat2CIce(TEST_DATA / cs_file).to_xarray_dataset()
    target_grid = GPM_IR_GRID

    # Setup resampler
    source_lons = da.from_array(cs_data.longitude.data)
    source_lats = da.from_array(cs_data.latitude.data)
    resampler = BucketResampler(
        target_grid, source_lons=source_lons, source_lats=source_lats
    )

    target_inds, source_inds = get_sample_indices(resampler)
    target_lons, target_lats = GPM_IR_GRID.get_lonlats()

    lons_gpm = target_lons.ravel()[target_inds]
    lons_cs = cs_data.longitude.data[source_inds]
    lats_gpm = target_lats.ravel()[target_inds]
    lats_cs = cs_data.latitude.data[source_inds]

    assert np.all(np.abs(lons_gpm - lons_cs) < 0.5)
    assert np.all(np.abs(lats_gpm - lats_cs) < 0.5)





