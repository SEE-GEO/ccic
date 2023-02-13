"""
Tests for the ccic.data.gridsat module.
"""
import os
from pathlib import Path

import numpy as np
import pytest

from ccic.data.cloudsat import CloudSat2CIce, CloudSat2BCLDCLASS
from ccic.data.gridsat import GridSat


TEST_DATA = os.environ.get("CCIC_TEST_DATA", None)
if TEST_DATA is not None:
    TEST_DATA = Path(TEST_DATA)
NEEDS_TEST_DATA = pytest.mark.skipif(
    TEST_DATA is None, reason="Needs 'CCIC_TEST_DATA'."
)
CS_2CICE_FILE = "2008032011612_09374_CS_2C-ICE_GRANULE_P1_R05_E02_F00.hdf"
CS_2BCLDCLASS_FILE = "2008032011612_09374_CS_2B-CLDCLASS_GRANULE_P1_R05_E02_F00.hdf"
GRIDSAT_FILE = "GRIDSAT-B1.2008.02.01.03.v02r01.nc"


def test_find_files():
    """
    Ensure that all three files in test data folder are found.
    """
    files = GridSat.find_files(TEST_DATA)
    assert len(files) == 3

    start_time = "2008-02-01T01:00:00"
    files = GridSat.find_files(TEST_DATA, start_time=start_time)
    assert len(files) == 1

    end_time = "2008-02-01T01:00:00"
    files = GridSat.find_files(TEST_DATA, end_time=end_time)
    assert len(files) == 2

    files = GridSat.find_files(TEST_DATA, start_time=start_time, end_time=end_time)
    assert len(files) == 0


def test_get_times():
    """
    Assert that the correct time are returned for a given day.
    """
    start_time = "2016-01-01T00:00:00"
    times = GridSat.get_available_files(start_time)
    assert len(times) == 8

    end_time = "2016-01-01T11:59:00"
    times = GridSat.get_available_files(
        start_time=start_time,
        end_time=end_time
    )
    assert len(times) == 4


@NEEDS_TEST_DATA
def test_get_input_file_attributes():
    """
    Assert that data is loaded with decreasing latitudes.
    """
    input_file = GridSat(TEST_DATA / GRIDSAT_FILE)
    attrs = input_file.get_input_file_attributes()
    assert isinstance(attrs, dict)

@NEEDS_TEST_DATA
def test_get_retrieval_input():
    """
    Assert that data is loaded with decreasing latitudes.
    """
    input_file = GridSat(TEST_DATA / GRIDSAT_FILE)
    x = input_file.get_retrieval_input()
    assert x.ndim == 4
    assert x.shape[0] == 1
    assert x.shape[1] == 1
    assert (x >= -1.5).all()
    assert (x <= 1.0).all()

    x = input_file.get_retrieval_input(roi=(0, 0, 1, 1))
    assert x.ndim == 4
    assert x.shape[2] == 256
    assert x.shape[3] == 256
    assert (x >= -1.5).all()
    assert (x <= 1.0).all()

@NEEDS_TEST_DATA
def test_to_xarray_dataset():
    """
    Assert that data is loaded with decreasing latitudes.
    """
    gridsat = GridSat(TEST_DATA / GRIDSAT_FILE)
    data = gridsat.to_xarray_dataset()
    assert (np.diff(data.lat.data) < 0.0).all()


@NEEDS_TEST_DATA
def test_matches():
    """
    Make sure that matches are found for files that overlap in time.
    """
    rng = np.random.default_rng(111)
    gridsat = GridSat(TEST_DATA / GRIDSAT_FILE)
    cloudsat_files = [
        CloudSat2CIce(TEST_DATA / CS_2CICE_FILE),
        CloudSat2BCLDCLASS(TEST_DATA / CS_2BCLDCLASS_FILE),
    ]
    scenes = gridsat.get_matches(rng, cloudsat_files)
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
