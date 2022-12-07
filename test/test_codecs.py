"""
Test custom codec for compressing IWP data.
"""
import numpy as np

from ccic.codecs import LogBins


def test_codec():
    """
    Ensures that encoding error are small and that NAN values are
    handled correctly.
    """
    codec = LogBins(1e-3, 1e1)
    values = np.logspace(-3, 1, 10_000)
    values[0] = 1e-4
    values[-1] = np.nan
    values_enc = codec.encode(values)

    assert values_enc[0] == 0
    assert values_enc[-2] < 254
    assert values_enc[-1] == 255
    assert values_enc.dtype == np.uint8

    values_dec = codec.decode(values_enc)

    assert np.isclose(values_dec[0], 0.0)
    # Discretization error is within +/- 2% for chosen range.
    assert np.all(np.isclose(values_dec[1:-1], values[1:-1], rtol=2e-2))
    assert np.isnan(values_dec[-1])
    assert np.all(np.isfinite(values_dec[:-1]))

