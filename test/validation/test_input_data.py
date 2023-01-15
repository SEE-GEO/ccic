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
    y = retrieval_input.get_radar_reflectivity(date)
    print(y)
    assert y is not None
