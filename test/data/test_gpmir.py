"""
Tests for the ccic.data.gpm_ir module.
"""
from pathlib import Path

from ccic.data.cloudsat import CloudSat2CIce, CloudSat2BCLDCLASS
from ccic.data.gpm_ir import GPMIR

TEST_DATA = Path("/home/simonpf/data_3/ccic/test")
CS_2CICE_FILE = "2008032011612_09374_CS_2C-ICE_GRANULE_P1_R05_E02_F00.hdf"
CS_2BCLDCLASS_FILE = "2008032011612_09374_CS_2B-CLDCLASS_GRANULE_P1_R05_E02_F00.hdf"
GPMIR_FILE = "merg_2008020101_4km-pixel.nc4"


def test_get_available_files():
    """
    Assert that the correct times are returned for a given day.
    """
    times = GPMIR.get_available_files("2016-01-01")
    assert len(times) == 24


def test_matches():
    """
    Make sure that matches are found for files that overlap in time.
    """
    gpmir = GPMIR(TEST_DATA / GPMIR_FILE)
    cloudsat_files = [
        CloudSat2CIce(TEST_DATA / CS_2CICE_FILE),
        CloudSat2BCLDCLASS(TEST_DATA / CS_2BCLDCLASS_FILE),
    ]
    scenes = gpmir.get_matches(cloudsat_files)
    assert len(scenes) > 0


