"""
Tests for the ccic.data.cloudsat module.
"""
import os
from pathlib import Path

import dask.array as da
import numpy as np
import pytest
from pyresample.bucket import BucketResampler

from ccic.data.gpmir import GPMIR, GPMIR_GRID
from ccic.data.cloudsat import (
    CloudSat2CIce,
    CloudSat2BCLDCLASS,
    get_sample_indices,
    remap_iwc,
    subsample_iwc_and_height,
    resample_data,
    remap_cloud_classes,
    get_available_granules
)


TEST_DATA = os.environ.get("CCIC_TEST_DATA", None)
if TEST_DATA is not None:
    TEST_DATA = Path(TEST_DATA)
NEEDS_TEST_DATA = pytest.mark.skipif(
    TEST_DATA is None, reason="Needs 'CCIC_TEST_DATA'."
)
CS_2CICE_FILE = "2008032011612_09374_CS_2C-ICE_GRANULE_P1_R05_E02_F00.hdf"
CS_2BCLDCLASS_FILE = "2008032011612_09374_CS_2B-CLDCLASS_GRANULE_P1_R05_E02_F00.hdf"
GPMIR_FILE = "merg_2008020101_4km-pixel.nc4"


def test_available_files():
    """
    Test that available files for 2008-02-01 are found.
    """
    available_files = CloudSat2CIce.get_available_files("2008-02-01T00:00:00")
    assert len(available_files) > 10
    available_files = CloudSat2BCLDCLASS.get_available_files("2008-02-01T00:00:00")
    assert len(available_files) > 10


def test_available_granules():
    """
    Test extraction of available granules is consistent with available files.
    """
    available_files = CloudSat2CIce.get_available_files("2008-02-01T00:00:00")
    available_files = CloudSat2BCLDCLASS.get_available_files("2008-02-01T00:00:00")
    available_granules = get_available_granules("2008-02-01T00:00:00")
    assert len(available_granules) == len(available_files)

@NEEDS_TEST_DATA
def test_subsample_iwc_and_height():
    """
    Test downsampling of IWC profiles by ensuring that the total IWP is
    conserved.
    """
    cs_data = CloudSat2CIce(TEST_DATA / CS_2CICE_FILE).to_xarray_dataset()

    iwc = cs_data.iwc.data[..., ::-1]
    height = cs_data.height[..., ::-1]
    iwc_s, height_s = subsample_iwc_and_height(iwc, height)

    # IWP in kg/m^2
    iwp = np.trapz(iwc, x=height, axis=-1) * 1e-3
    iwp_s = np.trapz(iwc_s, x=height_s, axis=-1) * 1e-3
    if np.any(iwp > 1e-4):
        assert np.any(iwp_s > 1e-4)

    notable_iwp = iwp > 1e-3
    iwp = iwp[notable_iwp]
    iwp_s = iwp_s[notable_iwp]
    assert np.all(np.isclose(iwp, iwp_s, rtol=1e-2))


@NEEDS_TEST_DATA
def test_remap_cloud_classes():
    """
    Test downsampling of cloud labels by ensuring that all returned labels
    are valid.
    """
    cs_data = CloudSat2BCLDCLASS(
        TEST_DATA / CS_2BCLDCLASS_FILE
    ).to_xarray_dataset()

    labels = cs_data.cloud_class.data
    height = cs_data.height
    surface_altitude = cs_data.surface_elevation.data
    target_altitudes = (np.arange(20) + 0.5) * 1e3

    labels = remap_cloud_classes(
        labels,
        height,
        surface_altitude,
        target_altitudes
    )
    assert ((labels <= 8) * (labels >= 0)).all()


@NEEDS_TEST_DATA
def test_remap_iwc():
    """
    Test remapping of IWC by ensuring that total IWP is conserved.
    """
    cs_data = CloudSat2CIce(TEST_DATA / CS_2CICE_FILE).to_xarray_dataset()

    iwc = cs_data.iwc.data[..., ::-1]
    height = cs_data.height.data[..., ::-1]
    surface_altitude = np.maximum(cs_data.surface_elevation.data, 0.0)
    below_surface = height < surface_altitude[..., None]
    iwc[below_surface] = 0.0
    iwc_s, height_s = subsample_iwc_and_height(iwc, height)

    target_altitudes = (np.linspace(0, 20, 21)) * 1e3
    iwc_r = remap_iwc(iwc_s, height_s, surface_altitude, target_altitudes)

    # IWP in kg/m^2
    iwp_s = np.trapz(iwc_s, x=height_s, axis=-1) * 1e-3
    iwp_r = np.trapz(iwc_r, x=target_altitudes, axis=-1) * 1e-3

    notable_iwp = iwp_s > 1e-3
    iwp_r = iwp_r[notable_iwp]
    iwp_s = iwp_s[notable_iwp]
    assert np.all(np.isclose(iwp_r, iwp_s, rtol=1e-2))


@NEEDS_TEST_DATA
def test_random_resampling():
    """
    Test random resampling of profiles by ensuring that the error
    from resampling of longitude and latitudes is of the same order
    as the grid resolution.
    """
    cs_data = CloudSat2CIce(TEST_DATA / CS_2CICE_FILE).to_xarray_dataset()
    target_grid = GPMIR_GRID

    # Setup resampler
    source_lons = da.from_array(cs_data.longitude.data)
    source_lats = da.from_array(cs_data.latitude.data)
    resampler = BucketResampler(
        target_grid, source_lons=source_lons, source_lats=source_lats
    )

    target_inds, source_inds = get_sample_indices(resampler)
    target_lons, target_lats = GPMIR_GRID.get_lonlats()

    lons_gpm = target_lons.ravel()[target_inds]
    lons_cs = cs_data.longitude.data[source_inds]
    lats_gpm = target_lats.ravel()[target_inds]
    lats_cs = cs_data.latitude.data[source_inds]

    assert np.all(np.abs(lons_gpm - lons_cs) < 0.05)
    assert np.all(np.abs(lats_gpm - lats_cs) < 0.05)


@NEEDS_TEST_DATA
def test_resampling_gpmir():
    """
    Test resampling of cloudsat data to GPM IR data.
    """
    gpm_data = GPMIR(TEST_DATA / GPMIR_FILE).to_xarray_dataset()
    gpm_data = gpm_data[{"time": 0}]
    cs_2cice_data = CloudSat2CIce(
        TEST_DATA / CS_2CICE_FILE
    ).to_xarray_dataset()

    cloudsat_files = [
        CloudSat2CIce(TEST_DATA / CS_2CICE_FILE),
        CloudSat2BCLDCLASS(TEST_DATA / CS_2BCLDCLASS_FILE),
    ]

    resample_data(
        gpm_data,
        GPMIR_GRID,
        cloudsat_files
    )

    # Make sure collocations are found.
    iwp_r = gpm_data.tiwp_fpavg.data
    valid = np.isfinite(iwp_r)
    assert np.any(valid)

    # Make sure average and random resampling map to the same
    # locations.
    iwp_rand_r = gpm_data.tiwp.data
    valid_rand = np.isfinite(iwp_rand_r)
    assert (valid == valid_rand).all()

    iwp = cs_2cice_data.iwp.data
    assert (iwp_r[valid] < iwp.max()).all()
    iwc = cs_2cice_data.iwc.data
    iwc_r = gpm_data.tiwc.data
    valid = np.isfinite(iwc_r)
    assert (iwc_r[valid] < iwc.max()).all()

    # Make sure that no cloud classes are consistent with
    # cloud mask.
    cm_r = gpm_data.cloud_mask.data
    clear = cm_r == 0
    cloud_classes = gpm_data.cloud_class.data
    assert cloud_classes[clear].max() == 0
