"""
Tests for the CCIC radar-only retrievals.
"""
import os
from pathlib import Path

import numpy as np
import pytest


try:
    from ccic.validation.retrieval import RadarRetrieval
    from ccic.validation.input_data import (
        RetrievalInput,
    )
    from ccic.validation.radars import (
        cloudnet_punta_arenas,
        arm_manacapuru,
        crs_olympex,
        rasta_haic_up
    )
    HAS_ARTSSAT = True
except ImportError as e:
    HAS_ARTSSAT = False
    print(e)
NEEDS_ARTSSAT = pytest.mark.skipif(
    not HAS_ARTSSAT, reason="Needs 'artssat' package installed."
)


try:
    TEST_DATA = Path(os.environ.get("CCIC_TEST_DATA", None))
    VALIDATION_DATA = Path(TEST_DATA) / "validation"
    HAS_TEST_DATA = True
except TypeError:
    HAS_TEST_DATA = False

NEEDS_TEST_DATA = pytest.mark.skipif(
    not HAS_TEST_DATA, reason="Needs 'CCIC_TEST_DATA'."
)


@NEEDS_ARTSSAT
@NEEDS_TEST_DATA
def test_retrieval_cloudnet():
    """Test running the retrieval for the Cloudnet radar in Punta Arenas."""
    date = np.datetime64("2020-03-15T10:00:00")
    radar_data_path = VALIDATION_DATA / "cloudnet"
    radar_files = cloudnet_punta_arenas.get_files(radar_data_path, date)
    retrieval_input = RetrievalInput(
        cloudnet_punta_arenas,
        radar_data_path,
        radar_files[0],
        VALIDATION_DATA / "era5",
        Path(TEST_DATA) / "validation" / "data",
    )
    retrieval = RadarRetrieval()
    results = retrieval.process(
        retrieval_input,
        np.timedelta64(8 * 60 * 60, "s"),
        "f07",
        "LargePlateAggregate"
    )
    assert "radar_reflectivity" in results
    assert "radar_reflectivity_fitted" in results


@NEEDS_ARTSSAT
@NEEDS_TEST_DATA
def test_retrieval_arm():
    """Test running the retrieval for the ARM WACR radar in Manacapuru."""
    date = np.datetime64("2014-12-10T10:00:00")
    radar_data_path = VALIDATION_DATA / "manacapuru"
    radar_files = arm_manacapuru.get_files(radar_data_path, date)
    retrieval_input = RetrievalInput(
        arm_manacapuru,
        radar_data_path,
        radar_files[0],
        VALIDATION_DATA / "era5",
        Path(TEST_DATA) / "validation" / "data",
    )
    retrieval = RadarRetrieval()
    results = retrieval.process(
        retrieval_input,
        np.timedelta64(8 * 60 * 60, "s"),
        "f07",
        "LargePlateAggregate"
    )
    assert "radar_reflectivity" in results
    assert "radar_reflectivity_fitted" in results


@NEEDS_ARTSSAT
@NEEDS_TEST_DATA
def test_retrieval_olympex():
    """Test running the retrieval for the CRS radar during Olympex campaign."""
    date = np.datetime64("2015-12-03T10:00:00")
    radar_data_path = VALIDATION_DATA / "olympex"
    radar_files = crs_olympex.get_files(radar_data_path, date)
    retrieval_input = RetrievalInput(
        crs_olympex,
        radar_data_path,
        radar_files[0],
        VALIDATION_DATA / "era5",
        Path(TEST_DATA) / "validation" / "data",
    )
    retrieval = RadarRetrieval()
    results = retrieval.process(
        retrieval_input,
        np.timedelta64(8 * 60 * 60, "s"),
        "f07",
        "LargePlateAggregate"
    )
    assert "radar_reflectivity" in results
    assert "radar_reflectivity_fitted" in results


@NEEDS_ARTSSAT
@NEEDS_TEST_DATA
def test_retrieval_haic():
    """Test running the retrieval for the HAIC campaing."""
    date = np.datetime64("2014-01-16T10:00:00")
    radar_data_path = VALIDATION_DATA / "haic"
    radar_files = rasta_haic_up.get_files(radar_data_path, date)
    retrieval_input = RetrievalInput(
        rasta_haic_up,
        radar_data_path,
        radar_files[0],
        VALIDATION_DATA / "haic" / "era5",
        Path(TEST_DATA) / "validation" / "data",
    )
    retrieval = RadarRetrieval()
    results = retrieval.process(
        retrieval_input,
        np.timedelta64(30 * 60, "s"),
        "f07",
        "LargePlateAggregate"
    )
    assert "radar_reflectivity" in results
    assert "radar_reflectivity_fitted" in results
