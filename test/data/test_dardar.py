"""
Tests for the ccic.data.dardar module
"""
from pathlib import Path
import os

import dask.array as da
import numpy as np
from pyresample.bucket import BucketResampler
import pytest

from ccic.data.cpcir import CPCIR, CPCIR_GRID
from ccic.data.cloudsat import (
    get_sample_indices,
    remap_cloud_classes,
    remap_iwc,
    resample_data,
    subsample_iwc_and_height,
    ALTITUDE_LEVELS
)
from ccic.data.dardar import DardarFile


TEST_DATA = os.environ.get("CCIC_TEST_DATA", None)
if TEST_DATA is not None:
    TEST_DATA = Path(TEST_DATA)
NEEDS_TEST_DATA = pytest.mark.skipif(
    TEST_DATA is None, reason="Needs 'CCIC_TEST_DATA'."
)

# Work with dardar-cloud 3.00
# DARDAR_FILE = "DARDAR-CLOUD_2010017014344_19801_V3-00.nc"
# Work with dardar-cloud 3.10
DARDAR_FILE = "DARDAR-CLOUD_2010017014344_19801_V3-10.nc"
CPCIR_FILE = "merg_2008020101_4km-pixel.nc4"

@NEEDS_TEST_DATA
def test_granule_parsing():
    """
    Ensure that parsin of the granule works.
    """
    dardar_file = DardarFile(DARDAR_FILE)
    assert dardar_file.granule == 19801

@NEEDS_TEST_DATA
def test_subsample_iwc_and_height():
    """
    Test downsampling of IWC profiles by ensuring that the total IWP is
    conserved.
    """
    dardar_file = DardarFile(TEST_DATA / DARDAR_FILE)
    dardar_data = dardar_file.to_xarray_dataset()

    # IWC transform to g/m3 for consistency
    iwc = dardar_data.iwc.data[..., ::-1] * 1e3
    height = np.tile(dardar_data.height[..., ::-1], (iwc.shape[0], 1))
    iwc_s, height_s = subsample_iwc_and_height(iwc, height)

    # Get IWPs (both in kg/m2)
    iwp = dardar_file.get_iwp(dardar_data) * 1e-3
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
    dardar_file = DardarFile(TEST_DATA / DARDAR_FILE)
    dardar_data = dardar_file.to_xarray_dataset()

    labels = dardar_data.DARMASK_Simplified_Categorization.data[..., ::-1]
    height = np.tile(dardar_data.height[..., ::-1], (labels.shape[0], 1))
    # Compute the surface altitude for each profile and flip it for consistency
    surface_mask = dardar_file.get_surface_mask(dardar_data)[..., ::-1]
    # Compute the surface elevation for each profile
    surface_altitude = np.max(surface_mask * height, axis=1)

    labels = remap_cloud_classes(
        labels,
        height,
        surface_altitude,
        ALTITUDE_LEVELS
    )
    assert (np.logical_or((labels <= 15) * (labels >= 0), labels==-2)).all()

@NEEDS_TEST_DATA
def test_remap_iwc():
    """
    Test remapping of IWC by ensuring that total IWP is conserved.
    """
    dardar_file = DardarFile(TEST_DATA / DARDAR_FILE)
    dardar_data = dardar_file.to_xarray_dataset()

    # IWC in g/m3
    iwc = dardar_data.iwc.data[..., ::-1] * 1e3
    height = np.tile(dardar_data.height[..., ::-1], (iwc.shape[0], 1))
    # Compute the surface altitude for each profile and flip it for consistency
    surface_mask = dardar_file.get_surface_mask(dardar_data)[..., ::-1]
    # Compute the surface elevation for each profile
    surface_altitude = np.max(surface_mask * height, axis=1)
    below_surface = height < surface_altitude[..., None]
    iwc[below_surface] = 0.0
    iwc_s, height_s = subsample_iwc_and_height(iwc, height)

    iwc_r = remap_iwc(iwc_s, height_s, surface_altitude, ALTITUDE_LEVELS)

    # IWP in kg/m^2
    iwp_s = np.trapz(iwc_s, x=height_s, axis=-1) * 1e-3
    iwp_r = np.trapz(iwc_r, x=ALTITUDE_LEVELS, axis=-1) * 1e-3

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
    cs_data = DardarFile(TEST_DATA / DARDAR_FILE).to_xarray_dataset()
    target_grid = CPCIR_GRID

    # Setup resampler
    source_lons = da.from_array(cs_data.longitude.data)
    source_lats = da.from_array(cs_data.latitude.data)
    resampler = BucketResampler(
        target_grid, source_lons=source_lons, source_lats=source_lats
    )

    target_inds, source_inds = get_sample_indices(resampler)
    target_lons, target_lats = CPCIR_GRID.get_lonlats()

    lons_cpc = target_lons.ravel()[target_inds]
    lons_cs = cs_data.longitude.data[source_inds]
    lats_cpc = target_lats.ravel()[target_inds]
    lats_cs = cs_data.latitude.data[source_inds]

    assert np.all(np.abs(lons_cpc - lons_cs) < 0.05)
    assert np.all(np.abs(lats_cpc - lats_cs) < 0.05)

@NEEDS_TEST_DATA
def test_resampling_cpcir():
    """
    Test resampling of cloudsat data to CPC IR data.
    """
    cpc_data = CPCIR(TEST_DATA / CPCIR_FILE).to_xarray_dataset()
    cpc_data = cpc_data[{"time": 0}].rename({
        "lon": "longitude",
        "lat": "latitude"
    })

    dardar_file = DardarFile(TEST_DATA / DARDAR_FILE)
    dardar_data = dardar_file.to_xarray_dataset()

    resample_data(
        cpc_data,
        CPCIR_GRID,
        [dardar_file]
    )

    # Make sure collocations are found.
    iwp_r = cpc_data.tiwp_fpavg.data
    valid = np.isfinite(iwp_r)
    assert np.any(valid)

    # Make sure average and random resampling map to the same
    # locations.
    iwp_rand_r = cpc_data.tiwp.data
    valid_rand = np.isfinite(iwp_rand_r)
    assert (valid == valid_rand).all()

    iwp = dardar_file.get_iwp(dardar_data) * 1e3
    assert (iwp_r[valid] < iwp.max()).all()

    iwc = dardar_data.iwc.data * 1e3
    iwc_r = cpc_data.tiwc.data
    valid = np.isfinite(iwc_r)
    assert (iwc_r[valid] < iwc.max()).all()

    # Make sure that no cloud classes are consistent with
    # cloud mask.
    cm_r = cpc_data.cloud_mask.data
    clear = cm_r == 0
    cloud_classes = cpc_data.cloud_class.data
    assert cloud_classes[clear].max() == 0
