"""
Tests for the ccic.data.gridsat module.
"""
from ccic.data.gridsat import GridSatB1

def test_get_times():
    """
    Assert that the correct time are returned for a given day.
    """
    times = GridSatB1.get_available_files("2016-01-01")
    assert len(times) == 8
