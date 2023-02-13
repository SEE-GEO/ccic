"""
Tests for the ccic.validation.input_data module.
"""
import os
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from ccic.validation.input_data import (
    CloudnetRadar,
    RetrievalInput,
    cloudnet_palaiseau
)

TEST_DATA = os.environ.get("CCIC_TEST_DATA", None)
NEEDS_TEST_DATA = pytest.mark.skipif(
    TEST_DATA is None, reason="Needs 'CCIC_TEST_DATA'."
)
VALIDATION_DATA = Path(TEST_DATA) / "validation"


@NEEDS_TEST_DATA
def test_cloudnet_radar():
    """
    Ensure that the loading of Cloudnet radar data works as expected.
    """
    data = cloudnet_palaiseau.load_data(
        VALIDATION_DATA / "cloudnet",
        np.datetime64("2021-01-02T14:00:00"),
        iwc_path=VALIDATION_DATA / "cloudnet"
    )
    assert data is not None
    assert "iwc" in data


@NEEDS_TEST_DATA
def test_retrieval_input():
    """
    Ensure that the loading of Cloudnet radar data works as expected.
    """
    retrieval_input = RetrievalInput(
        cloudnet_palaiseau,
        VALIDATION_DATA / "cloudnet",
        VALIDATION_DATA / "era5"
    )
    date = np.datetime64("2021-01-02T10:00:00")
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


def test_download_cloudnet_data(tmp_path):
    """
    Test download of Cloudnet data for a given location.
    """
    date = np.datetime64("2023-01-01T00:00:00")
    cloudnet_palaiseau.download_radar_data(date, tmp_path)

    assert len(list(tmp_path.glob("*.nc"))) == 2


def test_download_cloudnet_data(tmp_path):
    """
    Test download of Cloudnet data for a given location.
    """
    date = np.datetime64("2023-01-01T00:00:00")
    cloudnet_palaiseau.download_radar_data(date, tmp_path)

    assert len(list(tmp_path.glob("*.nc"))) == 2




