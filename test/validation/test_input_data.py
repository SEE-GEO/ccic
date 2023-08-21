"""
Tests for the ccic.validation.input_data module.
"""
import os
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from ccic.validation.input_data import (
    era5_files_in_range,
    RetrievalInput,
)
from ccic.validation.radars import (
    cloudnet_punta_arenas,
    arm_manacapuru,
    crs_olympex,
    )

try:
    TEST_DATA = Path(os.environ.get("CCIC_TEST_DATA", None))
    HAS_TEST_DATA = True
except TypeError:
    HAS_TEST_DATA = False

NEEDS_TEST_DATA = pytest.mark.skipif(
    not HAS_TEST_DATA, reason="Needs 'CCIC_TEST_DATA'."
)


@NEEDS_TEST_DATA
def test_era5_files_in_range():
    roi = arm_manacapuru.get_roi()
    time = np.datetime64("2014-12-10T00:00:00")
    path = VALIDATION_DATA / "era5"
    files = era5_files_in_range(path, roi, time, time)
    assert len(files) == 1

    start_time = np.datetime64("2014-12-10T00:00:00")
    end_time = np.datetime64("2014-12-10T22:01:00")
    path = VALIDATION_DATA / "era5"
    files = era5_files_in_range(path, roi, start_time, end_time)
    assert len(files) == 24


@NEEDS_TEST_DATA
def test_retrieval_input_cloudnet():
    """
    Ensure that the loading of Cloudnet radar data works as expected.
    """
    radar_path = VALIDATION_DATA / "cloudnet"
    date = np.datetime64("2020-03-15T10:00:00")
    radar_files = cloudnet_punta_arenas.get_files(radar_path, date)
    retrieval_input = RetrievalInput(
        cloudnet_punta_arenas,
        radar_path,
        radar_files[0],
        VALIDATION_DATA / "era5",
        VALIDATION_DATA / "validation"
    )

    assert retrieval_input.has_data()

    y = retrieval_input.get_y_radar(date)
    range_bins = retrieval_input.get_radar_range_bins(date)
    assert y is not None
    assert y.size == range_bins.size - 1

    p = retrieval_input.get_pressure(date)
    assert np.any(p > 800e2)

    t = retrieval_input.get_temperature(date)
    assert np.all(t > 150)

    h2o = retrieval_input.get_h2o(date)
    assert np.all(h2o >= 0)

    assert p.size == t.size
    assert h2o.size == t.size

    z = retrieval_input.get_altitude(date)
    z_s = retrieval_input.get_surface_altitude(date)
    assert np.all(range_bins > z_s)

    date = np.datetime64("2020-03-15T00:00:00")
    dbz = retrieval_input.get_y_radar(date)

    assert retrieval_input.has_data()

    iwc_data = retrieval_input.get_iwc_data(date, np.timedelta64(10 * 60, "s"))
    assert iwc_data is not None


@NEEDS_TEST_DATA
def test_retrieval_input_arm():
    """
    Ensure that the loading of ARM WACR radar data works as expected.
    """
    radar_path = VALIDATION_DATA / "manacapuru"
    date = np.datetime64("2014-12-10T00:00:00")
    radar_files = arm_manacapuru.get_files(radar_path, date)
    retrieval_input = RetrievalInput(
        arm_manacapuru,
        radar_path,
        radar_files[0],
        VALIDATION_DATA / "era5",
        VALIDATION_DATA / "validation"
    )

    assert retrieval_input.has_data()
    start, end = retrieval_input.get_start_and_end_time()

    y = retrieval_input.get_y_radar(start)
    range_bins = retrieval_input.get_radar_range_bins(start)
    assert y is not None
    assert y.size == range_bins.size - 1

    p = retrieval_input.get_pressure(start)
    assert np.any(p > 800e2)

    t = retrieval_input.get_temperature(start)
    assert np.all(t > 150)

    h2o = retrieval_input.get_h2o(start)
    assert np.all(h2o >= 0)

    assert p.size == t.size
    assert h2o.size == t.size

    z = retrieval_input.get_altitude(start)
    z_s = retrieval_input.get_surface_altitude(start)
    assert np.all(range_bins > z_s)

    assert retrieval_input.has_data()

@NEEDS_TEST_DATA
def test_retrieval_input_crs():
    """
    Ensure that the loading of NASA CRS radar data works as expected.
    """
    radar_path = VALIDATION_DATA / "olympex"
    date = np.datetime64("2015-12-03T00:00:00")
    radar_files = crs_olympex.get_files(radar_path, date)
    retrieval_input = RetrievalInput(
        crs_olympex,
        radar_path,
        radar_files[0],
        VALIDATION_DATA / "era5",
        VALIDATION_DATA
    )

    assert retrieval_input.has_data()
    start, end = retrieval_input.get_start_and_end_time()

    y = retrieval_input.get_y_radar(start)
    range_bins = retrieval_input.get_radar_range_bins(start)
    assert y is not None
    assert y.size == range_bins.size - 1

    p = retrieval_input.get_pressure(start)
    assert np.any(p > 800e2)

    t = retrieval_input.get_temperature(start)
    assert np.all(t > 150)

    h2o = retrieval_input.get_h2o(start)
    assert np.all(h2o >= 0)

    assert p.size == t.size
    assert h2o.size == t.size

    z = retrieval_input.get_altitude(start)
    z_s = retrieval_input.get_surface_altitude(start)
    assert np.all(range_bins > z_s)

    assert retrieval_input.has_data()
