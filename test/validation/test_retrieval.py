"""
Tests for the CCIC radar-only retrievals.
"""
import os
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from ccic.validation.retrieval import RadarRetrieval
from ccic.validation.input_data import RetrievalInput, cloudnet_palaiseau

TEST_DATA = os.environ.get("CCIC_TEST_DATA", None)
NEEDS_TEST_DATA = pytest.mark.skipif(
    TEST_DATA is None, reason="Needs 'CCIC_TEST_DATA'."
)
VALIDATION_DATA = Path(TEST_DATA) / "validation"


def test_retrieval_cloudnet():
    """Test running the retrieval for the Cloudnet radar in Palaiseau."""
    retrieval_input = RetrievalInput(
        cloudnet_palaiseau,
        VALIDATION_DATA / "cloudnet",
        VALIDATION_DATA / "era5"
    )
    retrieval = RadarRetrieval(
        cloudnet_palaiseau,
        retrieval_input,
        Path(TEST_DATA) / "validation" / "data",
        "8-ColumnAggregate"
    )
    results = retrieval.process(
        np.datetime64("2021-01-02T10:00:00"),
        np.timedelta64(8 * 60 * 60, "s")
    )

    assert "radar_reflectivity" in results
    assert "radar_reflectivity_fitted" in results


