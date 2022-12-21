"""
Tests for the processing functions defined in ccic.processing.py
"""
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
import timeit


from quantnn.mrnn import MRNN
import numpy as np

from ccic.data.gpmir import GPMIR
from ccic.data.gridsat import GridSatB1
from ccic.processing import (
    get_input_files,
    RemoteFile,
    process_input_file,
    get_output_filename
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


def test_remote_file():
    """
    Test that input files are determined correctly.
    """
    temp_dir_1 = TemporaryDirectory()
    temp_dir_2 = TemporaryDirectory()

    pool = ThreadPoolExecutor(max_workers=4)
    input_files_no_prefetch = get_input_files(
        GPMIR,
        start_time="2008-02-01T00:00:00",
        working_dir=temp_dir_1.name
    )
    input_files_prefetch = get_input_files(
        GPMIR,
        start_time="2008-02-02T00:00:00",
        thread_pool=pool,
        working_dir=temp_dir_2.name
    )

    if_1 = input_files_no_prefetch[0]
    if_2 = input_files_prefetch[0]
    print("No prefetch :: ", timeit.timeit('if_1.get()', number=1, globals=locals()))
    print("Prefetch ::    ", timeit.timeit('if_2.get()', number=1, globals=locals()))


def test_processing():
    """
    Test processing of GPMIR and GridSat input files.
    """
    mrnn = MRNN.load(TEST_DATA / "models" / "ccic.pckl")
    gpmir_file = GPMIR(TEST_DATA / "input_data" / "merg_2008020100_4km-pixel.nc4")
    gridsat_file = GridSatB1(TEST_DATA / "input_data" / "GRIDSAT-B1.2008.02.01.00.v02r01.nc")

    for input_file in [gpmir_file, gridsat_file]:
        results = process_input_file(mrnn, input_file)
        print(results)
        assert "tiwp" in results
        assert "tiwp_log_std_dev" in results
        assert "p_tiwp" in results
        assert np.all(np.isfinite(results["tiwp"].data))
        assert np.all(np.isfinite(results["tiwp_log_std_dev"].data))
        assert np.all(np.isfinite(results["p_tiwp"].data))

        assert "tiwp_fpavg" in results
        assert "tiwp_fpavg_log_std_dev" in results
        assert "p_tiwp_fpavg" in results
        assert np.all(np.isfinite(results["tiwp_fpavg"].data))
        assert np.all(np.isfinite(results["tiwp_fpavg_log_std_dev"].data))
        assert np.all(np.isfinite(results["p_tiwp_fpavg"].data))

        assert "tiwc" in results
        assert "tiwc_log_std_dev" not in results
        assert "p_tiwc" not in results

        assert "input_filename" in results.attrs
        assert "processing_time" in results.attrs


def test_get_output_filename():
    gpmir_file = GPMIR(TEST_DATA / "input_data" / "merg_2008020100_4km-pixel.nc4")
    data = gpmir_file.to_xarray_dataset()
    output_filename = get_output_filename(gpmir_file, data.time[0].item())
