"""
Tests for the ccic.data.gridsat module.
"""
from pathlib import Path

from ccic.data.cloudsat import CloudSat2CIce, CloudSat2BCLDCLASS
from ccic.data.gridsat import GridSatB1


TEST_DATA = Path("/home/simonpf/data_3/ccic/test")
CS_2CICE_FILE = "2008032011612_09374_CS_2C-ICE_GRANULE_P1_R05_E02_F00.hdf"
CS_2BCLDCLASS_FILE = "2008032011612_09374_CS_2B-CLDCLASS_GRANULE_P1_R05_E02_F00.hdf"
GRIDSAT_FILE = "GRIDSAT-B1.2008.02.01.03.v02r01.nc"

def test_get_times():
    """
    Assert that the correct time are returned for a given day.
    """
    times = GridSatB1.get_available_files("2016-01-01")
    assert len(times) == 8


def test_matches():
    """
    Make sure that matches are found for files that overlap in time.
    """
    gridsat = GridSatB1(TEST_DATA / GRIDSAT_FILE)
    cloudsat_files = [
        CloudSat2CIce(TEST_DATA / CS_2CICE_FILE),
        CloudSat2BCLDCLASS(TEST_DATA / CS_2BCLDCLASS_FILE),
    ]
    scenes = gridsat.get_matches(cloudsat_files)
    assert len(scenes) > 0

    assert "iwp" in scenes[0].variables
    assert "cloud_mask" in scenes[0].variables
