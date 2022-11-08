"""
Tests for the processing functions defined in ccic.processing.py
"""
import os
from pathlib import Path

from quantnn.mrnn import MRNN

from ccic.data.gpmir import GPMIR
from ccic.data.gridsat import GridSatB1
from ccic.processing import (
    get_input_files,
    RemoteFile,
    process_input_file
)


TEST_DATA = Path(os.environ.get("CCIC_TEST_DATA", None))


def test_get_input_files():
    """
    Test that input files are determined correctly.
    """

    #
    # GPMIR
    #

    input_files = get_input_files(
        GPMIR,
        start_time="2008-02-01T00:00:00"
    )
    assert len(input_files) == 1
    assert isinstance(input_files[0], RemoteFile)
    input_files = get_input_files(
        GPMIR,
        start_time="2008-02-01T00:00:00",
        end_time="2008-02-01T23:59:00"
    )
    assert len(input_files) == 24
    assert isinstance(input_files[0], RemoteFile)

    input_files = get_input_files(
        GPMIR,
        start_time="2008-02-01T00:00:00",
        path=TEST_DATA
    )
    assert len(input_files) == 2
    assert isinstance(input_files[0], GPMIR)

    input_files = get_input_files(
        GPMIR,
        start_time="2008-02-01T00:00:00",
        end_time="2008-02-02T00:00:00",
        path=TEST_DATA
    )
    assert len(input_files) == 4
    assert isinstance(input_files[0], GPMIR)

    #
    # GridSat
    #

    input_files = get_input_files(
        GridSatB1,
        start_time="2008-02-01T00:00:00"
    )
    assert len(input_files) == 1
    assert isinstance(input_files[0], RemoteFile)
    input_files = get_input_files(
        GridSatB1,
        start_time="2008-02-01T00:00:00",
        end_time="2008-02-01T23:59:00"
    )
    assert len(input_files) == 8
    assert isinstance(input_files[0], RemoteFile)

    input_files = get_input_files(
        GridSatB1,
        start_time="2008-02-01T00:00:00",
        path=TEST_DATA
    )
    assert len(input_files) == 2
    assert isinstance(input_files[0], GridSatB1)

    input_files = get_input_files(
        GridSatB1,
        start_time="2008-02-01T00:00:00",
        end_time="2008-02-02T00:00:00",
        path=TEST_DATA
    )
    assert len(input_files) == 3
    assert isinstance(input_files[0], GridSatB1)


def test_processing():
    """
    Test processing of GPMIR and GridSat input files.
    """
    mrnn = MRNN.load(TEST_DATA / "models" / "ccic.pckl")
    gpmir_file = GPMIR(TEST_DATA / "input_data" / "merg_2008020100_4km-pixel.nc4")
    gridsat_file = GridSatB1(TEST_DATA / "input_data" / "GRIDSAT-B1.2008.02.01.00.v02r01.nc")

    for input_file in [gpmir_file, gridsat_file]:
        results = process_input_file(mrnn, input_file)
        assert "iwp_mean" in results
        assert "iwp_quantiles" in results
        assert "iwp_sample" in results

        assert "iwp_rand_mean" in results
        assert "iwp_rand_quantiles" in results
        assert "iwp_rand_sample" in results

        assert "iwc_mean" in results
        assert "iwc_quantiles" not in results
        assert "iwc_sample" not in results

        assert "input_filename" in results.attrs
        assert "processing_time" in results.attrs
