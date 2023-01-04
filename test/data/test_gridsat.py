"""
Tests for the ccic.data.gridsat module.
"""
import os
from pathlib import Path

import numpy as np
import pytest

from ccic.data.cloudsat import CloudSat2CIce, CloudSat2BCLDCLASS
from ccic.data.gridsat import GridSatB1


TEST_DATA = os.environ.get("CCIC_TEST_DATA", None)
if TEST_DATA is not None:
    TEST_DATA = Path(TEST_DATA)
NEEDS_TEST_DATA = pytest.mark.skipif(
    TEST_DATA is None, reason="Needs 'CCIC_TEST_DATA'."
)
CS_2CICE_FILE = "2008032011612_09374_CS_2C-ICE_GRANULE_P1_R05_E02_F00.hdf"
CS_2BCLDCLASS_FILE = "2008032011612_09374_CS_2B-CLDCLASS_GRANULE_P1_R05_E02_F00.hdf"
GRIDSAT_FILE = "GRIDSAT-B1.2008.02.01.03.v02r01.nc"

def test_get_times():
    """
    Assert that the correct time are returned for a given day.
    """
    times = GridSatB1.get_available_files("2016-01-01")
    assert len(times) == 8


@NEEDS_TEST_DATA
def test_to_xarray_dataset():
    """
    Assert that data is loaded with decreasing latitudes.
    """
    gridsat = GridSatB1(TEST_DATA / GRIDSAT_FILE)
    data = gridsat.to_xarray_dataset()
    assert (np.diff(data.lat.data) < 0.0).all()


@NEEDS_TEST_DATA
def test_matches():
    """
    Make sure that matches are found for files that overlap in time.
    """
    gridsat = GridSatB1(TEST_DATA / GRIDSAT_FILE)
    cloudsat_files = [
        CloudSat2CIce(TEST_DATA / CS_2CICE_FILE),
        CloudSat2BCLDCLASS(TEST_DATA / CS_2BCLDCLASS_FILE),
    ]
    scenes = gridsat.get_matches(cloudsat_files)
    print(len(scenes))
    assert len(scenes) > 0

    assert "tiwp" in scenes[0].variables
    assert "cloud_mask" in scenes[0].variables

    # Make sure observations and output are co-located.
    for scene in scenes:
        lats_cs = scene.latitude_cloudsat.data
        lons_cs = scene.longitude_cloudsat.data

        rows, cols = np.where(np.isfinite(lats_cs))
        assert np.all(np.isclose(
            lats_cs[rows, cols], scene.latitude.data[rows], atol=0.1
        ))
        assert np.all(np.isclose(
            lons_cs[rows, cols], scene.longitude.data[cols], atol=0.1
        ))
