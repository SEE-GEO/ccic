"""
Tests for the ccic.validation.radars module.
"""
import os
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from ccic.validation.radars import (
    cloudnet_punta_arenas,
    arm_manacapuru,
    crs_olympex
)


TEST_DATA = os.environ.get("CCIC_TEST_DATA", None)
NEEDS_TEST_DATA = pytest.mark.skipif(
    TEST_DATA is None, reason="Needs 'CCIC_TEST_DATA'."
)
if TEST_DATA is not None:
    TEST_DATA = Path(TEST_DATA)


@NEEDS_TEST_DATA
def test_cloudnet_radar():
    date = np.datetime64("2020-03-14T00:00:00")

    # Check that the correct number of files is found.
    files = cloudnet_punta_arenas.get_files(None, date)
    assert len(files) == 1

    # Check that start and end time are right.
    start_time, end_time = cloudnet_punta_arenas.get_start_and_end_time(None, files[0])
    assert start_time == np.datetime64("2020-03-14T00:00:00")
    assert end_time == np.datetime64("2020-03-15T00:00:00")

    # Try loading data.
    path = TEST_DATA / "validation" / "cloudnet"
    cloudnet_punta_arenas.load_data(path, files[0])


@NEEDS_TEST_DATA
def test_arm_radar():
    date = np.datetime64("2014-12-10T10:00:00")
    path = TEST_DATA / "validation" / "manacapuru"

    # Check that the correct number of files is found.
    files = arm_manacapuru.get_files(path, date)
    assert len(files) == 1

    # Check that start and end time are right.
    start_time, end_time = arm_manacapuru.get_start_and_end_time(path, files[0])
    assert start_time >= np.datetime64("2014-12-10T00:00:00")
    assert end_time < np.datetime64("2014-12-11T00:00:00")

    # Try loading data.
    arm_manacapuru.load_data(path, files[0])


@NEEDS_TEST_DATA
def test_nasa_crs():
    date = np.datetime64("2015-12-03T00:00:00")
    path = TEST_DATA / "validation" / "olympex"

    # Check that the correct number of files is found.
    files = crs_olympex.get_files(path, date)
    assert len(files) == 3

    # Check that start and end time are right.
    start_time, end_time = crs_olympex.get_start_and_end_time(path, files[0])
    assert start_time >= np.datetime64("2015-12-03T14:28:55")
    assert end_time <= np.datetime64("2015-12-03T14:58:50")

    # Try loading data.
    crs_olympex.load_data(path, files[0], TEST_DATA / "validation")
