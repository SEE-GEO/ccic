"""
Tests for the ccic.data.cloudsat module.
"""
from pathlib import Path

import dask.array as da
import numpy as np
from pyresample.bucket import BucketResampler

from ccic.data.gpm_ir import GPMIR, GPM_IR_GRID
from ccic.data.cloudsat import (
    CloudSat2CIce,
    get_sample_indices,
    remap_iwc,
    subsample_iwc_and_height,
    resample_data
)

TEST_DATA = Path("/home/simonpf/data_3/ccic/test")
CS_FILE = "2008032011612_09374_CS_2C-ICE_GRANULE_P1_R05_E02_F00.hdf"
GPMIR_FILE = "merg_2008020101_4km-pixel.nc4"


def test_subsample_iwc_and_height():
    """
    Test downsampling of IWC profiles by ensuring that the total IWP is
    conserved.
    """
    cs_data = CloudSat2CIce(TEST_DATA / CS_FILE).to_xarray_dataset()

    iwc = cs_data.iwc.data
    height = cs_data.height
    iwc_s, height_s = subsample_iwc_and_height(iwc, height)

    # IWP in kg/m^2
    iwp = np.trapz(iwc, x=height, axis=-1) * 1e-3
    iwp_s = np.trapz(iwc_s, x=height_s, axis=-1) * 1e-3

    # Ensure that subsampling has negligible effect for columns
    # with non-negligible IWP.
    notable_iwp = iwp > 1e-4
    assert np.all(np.isclose(iwp[notable_iwp], iwp_s[notable_iwp], rtol=1e-3))


def test_remap_iwc():
    """
    Test remapping of IWC by ensuring that total IWP is conserved.

    """
    cs_data = CloudSat2CIce(TEST_DATA / CS_FILE).to_xarray_dataset()

    iwc = cs_data.iwc.data
    height = cs_data.height.data
    surface_altitude = cs_data.surface_elevation.data

    target_altitudes = (np.arange(20) + 0.5) * 1e3
    iwc_s = remap_iwc(iwc, height, surface_altitude, target_altitudes)

    # IWP in kg/m^2
    iwp = np.trapz(iwc, x=height, axis=-1) * 1e-3
    iwp_s = (iwc_s * 1e3).sum(axis=-1) * 1e-3

    # Ensure that subsampling has negligible effect for columns
    # with non-negligible IWP.
    notable_iwp = iwp > 1e-4
    assert np.all(np.isclose(iwp[notable_iwp], iwp_s[notable_iwp], rtol=1e-3))


def test_random_resampling():
    """
    Test random resampling of profiles by ensuring that the error
    from resampling of longitude and latitudes is of the same order
    as the grid resolution.
    """
    gpm_data = GPMIR(TEST_DATA / GPMIR_FILE).to_xarray_dataset()
    cs_data = CloudSat2CIce(TEST_DATA / CS_FILE).to_xarray_dataset()
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

    assert np.all(np.abs(lons_gpm - lons_cs) < 0.05)
    assert np.all(np.abs(lats_gpm - lats_cs) < 0.05)


def test_resampling_gpmir():
    """
    Test resampling of cloudsat data to GPM IR data.
    """
    gpm_data = GPMIR(TEST_DATA / GPMIR_FILE).to_xarray_dataset()
    gpm_data = gpm_data[{"time": 0}]
    cs_data = CloudSat2CIce(TEST_DATA / CS_FILE).to_xarray_dataset()

    data_resampled = resample_data(gpm_data, GPM_IR_GRID, cs_data)

    iwp_r = gpm_data.iwp.data
    valid = np.isfinite(iwp_r)
    assert np.any(valid)

    iwp = cs_data.iwp.data
    assert (iwp_r[valid] < iwp.max()).all()

    iwc = cs_data.iwc.data
    iwc_r = gpm_data.iwc.data
    valid = np.isfinite(iwc_r)
    assert (iwc_r[valid] < iwc.max()).all()

