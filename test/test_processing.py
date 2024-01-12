"""
Tests for the processing functions defined in ccic.processing.py
"""
import os
from concurrent.futures import ThreadPoolExecutor
from logging import getLogger
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import timeit


import numpy as np
import pytest
from quantnn.mrnn import MRNN
import torch
import xarray as xr

from ccic.data.cpcir import CPCIR
from ccic.data.gridsat import GridSat
from ccic.processing import (
    get_input_files,
    RemoteFile,
    process_input_file,
    get_output_filename,
    RetrievalSettings,
    get_encodings,
    OutputFormat,
    ProcessingLog,
    get_invalid_mask,
    determine_cloud_class,
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
def test_get_input_files():
    """
    Test that input files are determined correctly.
    """

    #
    # CPCIR
    #

    input_files = get_input_files(
        CPCIR,
        start_time="2008-02-01T00:00:00"
    )
    assert len(input_files) == 1
    assert isinstance(input_files[0], RemoteFile)
    input_files = get_input_files(
        CPCIR,
        start_time="2008-02-01T00:00:00",
        end_time="2008-02-01T23:59:00"
    )
    assert len(input_files) == 24
    assert isinstance(input_files[0], RemoteFile)

    input_files = get_input_files(
        CPCIR,
        start_time="2008-02-01T00:00:00",
        path=TEST_DATA
    )
    assert len(input_files) == 2
    assert isinstance(input_files[0], CPCIR)

    input_files = get_input_files(
        CPCIR,
        start_time="2008-02-01T00:00:00",
        end_time="2008-02-02T00:00:00",
        path=TEST_DATA
    )
    assert len(input_files) == 4
    assert isinstance(input_files[0], CPCIR)

    #
    # GridSat
    #

    input_files = get_input_files(
        GridSat,
        start_time="2008-02-01T00:00:00"
    )
    assert len(input_files) == 1
    assert isinstance(input_files[0], RemoteFile)
    input_files = get_input_files(
        GridSat,
        start_time="2008-02-01T00:00:00",
        end_time="2008-02-01T23:59:00"
    )
    assert len(input_files) == 8
    assert isinstance(input_files[0], RemoteFile)

    input_files = get_input_files(
        GridSat,
        start_time="2008-02-01T00:00:00",
        path=TEST_DATA
    )
    assert len(input_files) == 2
    assert isinstance(input_files[0], GridSat)

    input_files = get_input_files(
        GridSat,
        start_time="2008-02-01T00:00:00",
        end_time="2008-02-02T00:00:00",
        path=TEST_DATA
    )
    assert len(input_files) == 3
    assert isinstance(input_files[0], GridSat)


@NEEDS_TEST_DATA
def test_remote_file():
    """
    Test that input files are determined correctly.
    """
    temp_dir_1 = TemporaryDirectory()
    temp_dir_2 = TemporaryDirectory()

    pool = ThreadPoolExecutor(max_workers=4)
    input_files_no_prefetch = get_input_files(
        CPCIR,
        start_time="2008-02-01T00:00:00",
        working_dir=temp_dir_1.name
    )
    input_files_prefetch = get_input_files(
        CPCIR,
        start_time="2008-02-02T00:00:00",
        thread_pool=pool,
        working_dir=temp_dir_2.name
    )

    if_1 = input_files_no_prefetch[0]
    if_2 = input_files_prefetch[0]


@NEEDS_TEST_DATA
def test_processing(tmp_path):
    """
    Test processing and writing of CPCIR and GridSat input files.
    """
    mrnn = MRNN.load(TEST_DATA / "models" / "ccic.pckl")
    cpcir_file = CPCIR(TEST_DATA / "input_data" / "merg_2008020100_4km-pixel.nc4")
    gridsat_file = GridSat(TEST_DATA / "input_data" / "GRIDSAT-B1.2008.02.01.00.v02r01.nc")

    for input_file in [cpcir_file, gridsat_file]:
        results = process_input_file(mrnn, input_file)
        assert "tiwp" in results
        assert "tiwp_ci" in results
        assert "p_tiwp" in results
        assert np.any(np.isfinite(results["tiwp"].data))
        assert np.any(np.isfinite(results["tiwp_ci"].data))
        assert np.any(np.isfinite(results["p_tiwp"].data))

        assert "tiwp_fpavg" in results
        assert "tiwp_fpavg_ci" in results
        assert "p_tiwp_fpavg" in results
        assert np.any(np.isfinite(results["tiwp_fpavg"].data))
        assert np.any(np.isfinite(results["tiwp_fpavg_ci"].data))
        assert np.any(np.isfinite(results["p_tiwp_fpavg"].data))

        assert "tiwc" in results
        assert "tiwc_ci" not in results
        assert "p_tiwc" not in results

        assert "input_filename" in results.attrs
        assert "processing_time" in results.attrs

        # Store as netcdf file.
        retrieval_settings = RetrievalSettings()
        retrieval_settings.output_format = OutputFormat["NETCDF"]
        filename = get_output_filename(
            input_file,
            results.time[0].item(),
            retrieval_settings
        )
        encodings = get_encodings(results.variables.keys(), retrieval_settings)
        results.to_netcdf(tmp_path / filename, encoding=encodings)
        results_nc = xr.load_dataset(tmp_path / filename)

        # Store as zarr file.
        retrieval_settings.output_format = OutputFormat["ZARR"]
        filename = get_output_filename(
            input_file,
            results.time[0].item(),
            retrieval_settings
        )
        encodings = get_encodings(results.variables.keys(), retrieval_settings)
        results.to_zarr(tmp_path / filename, encoding=encodings)
        results_zarr = xr.load_dataset(tmp_path / filename)

        assert results_nc.variables.keys() == results_zarr.variables.keys()


@NEEDS_TEST_DATA
def test_get_output_filename():
    """
    Ensure that filenames have the right suffixes.
    """
    cpcir_file = CPCIR(TEST_DATA / "input_data" / "merg_2008020100_4km-pixel.nc4")
    data = cpcir_file.to_xarray_dataset()
    retrieval_settings = RetrievalSettings()
    retrieval_settings.output_format = OutputFormat["NETCDF"]
    output_filename_netcdf = get_output_filename(cpcir_file, data.time[0].item(), retrieval_settings)
    assert Path(output_filename_netcdf).suffix == ".nc"

    retrieval_settings.output_format = OutputFormat["ZARR"]
    output_filename_zarr = get_output_filename(cpcir_file, data.time[0].item(), retrieval_settings)
    assert Path(output_filename_zarr).suffix == ".zarr"

    assert output_filename_netcdf != output_filename_zarr


def test_processing_logger(tmp_path):
    """
    Tests the processing database by ensuring that log information
    during execution is captured.
    """
    # Check that disabling the database log works.
    db_path = None
    pl = ProcessingLog(db_path, "input_file.nc")
    LOGGER = getLogger()
    with pl.log(LOGGER):
        LOGGER.error("THIS IS A LOG.")
    with pl.log(LOGGER):
        LOGGER.error("THIS IS ANOTHER LOG.")
    data = np.random.normal(size=(100, 100))
    data[90:, 90:]
    results = xr.Dataset({
        "tiwp": (("x", "y"), data)
    })
    pl.finalize(results, "output_file.nc")
    assert len(list(tmp_path.glob("*"))) == 0

    # Check that log events are captured
    db_path = tmp_path / "processing.db"
    pl = ProcessingLog(db_path, "input_file_2.nc")
    LOGGER = getLogger()
    with pl.log(LOGGER):
        LOGGER.error("THIS IS A LOG.")
    with pl.log(LOGGER):
        LOGGER.error("THIS IS ANOTHER LOG.")
    data = np.random.normal(size=(100, 100))
    data[90:, 90:]
    results = xr.Dataset({
        "tiwp": (("x", "y"), data)
    })
    pl.finalize(results, "output_file_2")

    # Check retrieval of successful file
    success = ProcessingLog.get_input_file(db_path, success=True)
    assert len(success) == 1

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        res = cursor.execute("SELECT log FROM files")
        entry = res.fetchone()
        assert entry is not None
        assert len(entry[0]) > 0

    # Check retrieval of failed files.
    failed = ProcessingLog.get_input_file(db_path, success=False)
    assert len(failed) == 0

    pl = ProcessingLog(db_path, "input_file_3.nc")
    LOGGER = getLogger()
    with pl.log(LOGGER):
        LOGGER.error("THIS IS A LOG.")
    with pl.log(LOGGER):
        LOGGER.error("THIS IS ANOTHER LOG.")

    # Check retrieval of failed files.
    failed = ProcessingLog.get_input_file(db_path, success=False)
    assert len(failed) == 1

    # Check retrieval of sucessful files
    pl.finalize({}, 'output_file_3.nc')
    success = ProcessingLog.get_input_file(db_path, success=True)
    assert len(success) == 2

    # Check that all files can be recovered
    pl = ProcessingLog(db_path, "input_file_4.nc")
    all = ProcessingLog.get_input_file(db_path)
    success = ProcessingLog.get_input_file(db_path, success=True)
    failed = ProcessingLog.get_input_file(db_path, success=False)
    assert len(all) == len(success + failed)


@NEEDS_TEST_DATA
def test_invalid_mask():
    """
    Test masking of invalid inputs.
    """
    mrnn = MRNN.load(TEST_DATA / "models" / "ccic.pckl")
    cpcir_file = CPCIR(TEST_DATA / "input_data" / "merg_2008020100_4km-pixel.nc4")
    gridsat_file = GridSat(TEST_DATA / "input_data" / "GRIDSAT-B1.2008.02.01.00.v02r01.nc")

    for input_file in [cpcir_file, gridsat_file]:
        x = input_file.get_retrieval_input()
        mask = get_invalid_mask(x)
        assert np.any(~mask)


def test_determine_cloud_class():
    """
    Test that the cloud class is determined correctly from a tensor
    of cloud type probabilities.
    """
    probs = np.array([
        [0.0, 0.2, 0.8],
        [0.4, 0.35, 0.25],
        [0.64, 0.2, 0.2],
    ])

    types = determine_cloud_class(probs, axis=-1)

    assert types[0] == 2
    assert types[1] == 1
    assert types[2] == 0
