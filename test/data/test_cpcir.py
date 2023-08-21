"""
Tests for the ccic.data.cpcir module.
"""
import os
from pathlib import Path

import numpy as np
import pytest

from ccic.data.cloudsat import CloudSat2CIce, CloudSat2BCLDCLASS
from ccic.data.cpcir import CPCIR

try:
    TEST_DATA = Path(os.environ.get("CCIC_TEST_DATA", None))
    HAS_TEST_DATA = True
except TypeError:
    HAS_TEST_DATA = False


NEEDS_TEST_DATA = pytest.mark.skipif(
    not HAS_TEST_DATA, reason="Needs 'CCIC_TEST_DATA'."
)

CS_2CICE_FILE = "2008032011612_09374_CS_2C-ICE_GRANULE_P1_R05_E02_F00.hdf"
CS_2BCLDCLASS_FILE = "2008032011612_09374_CS_2B-CLDCLASS_GRANULE_P1_R05_E02_F00.hdf"
CPCIR_FILE = "merg_2008020101_4km-pixel.nc4"


@NEEDS_TEST_DATA
def test_find_files():
    """
    Ensure that all four files in test data folder are found.
    """
    files = CPCIR.find_files(TEST_DATA)
    assert len(files) == 4

    start_time = "2008-02-01T01:00:00"
    files = CPCIR.find_files(TEST_DATA, start_time=start_time)
    assert len(files) == 2

    end_time = "2008-02-01T01:00:00"
    files = CPCIR.find_files(TEST_DATA, end_time=end_time)
    assert len(files) == 3

    files = CPCIR.find_files(TEST_DATA, start_time=start_time, end_time=end_time)
    assert len(files) == 1


def test_get_available_files():
    """
    Assert that the correct times are returned for a given day.
    """
    files = CPCIR.get_available_files("2016-01-01T00:00:00")
    assert len(files) == 24

    files = CPCIR.get_available_files(
        start_time="2016-01-01T00:00:00",
        end_time="2016-01-01T11:59:00")
    assert len(files) == 12


@NEEDS_TEST_DATA
def test_get_input_file_attributes():
    """
    Assert that data is loaded with decreasing latitudes.
    """
    cpcir = CPCIR(TEST_DATA / CPCIR_FILE)
    attrs = cpcir.get_input_file_attributes()
    assert isinstance(attrs, dict)

@NEEDS_TEST_DATA
def test_get_retrieval_input():
    """
    Assert that data is loaded with decreasing latitudes.
    """
    cpcir = CPCIR(TEST_DATA / CPCIR_FILE)
    x = cpcir.get_retrieval_input()
    assert x.ndim == 4
    assert x.shape[0] == 2
    assert x.shape[1] == 1
    assert (x >= -1.5).all()
    assert (x <= 1.0).all()

    x = cpcir.get_retrieval_input(roi=(0, 0, 1, 1))
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
    cpcir = CPCIR(TEST_DATA / CPCIR_FILE)
    data = cpcir.to_xarray_dataset()
    assert (np.diff(data.lat.data) < 0.0).all()


@NEEDS_TEST_DATA
def test_matches():
    """
    Make sure that matches are found for files that overlap in time.
    """
    rng = np.random.default_rng(111)
    cpcir = CPCIR(TEST_DATA / CPCIR_FILE)
    cloudsat_files = [
        CloudSat2CIce(TEST_DATA / CS_2CICE_FILE),
        CloudSat2BCLDCLASS(TEST_DATA / CS_2BCLDCLASS_FILE),
    ]

    size = 256
    scenes = cpcir.get_matches(rng, cloudsat_files, size=size)
    assert len(scenes) > 0
    for scene in scenes:
        assert scene.tiwp.shape == (size, size)

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

    # Test subsampling
    scenes = cpcir.get_matches(rng, cloudsat_files, subsample=True)
