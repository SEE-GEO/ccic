"""
Tests for the ccic.data.gpm_ir module.
"""
from ccic.data.gpm_ir import GPMIR

def test_get_available_files():
    """
    Assert that the correct times are returned for a given day.
    """
    times = GPMIR.get_available_files("2016-01-01")
    assert len(times) == 24
