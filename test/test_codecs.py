"""
Test custom codec for compressing IWP data.
"""
import numpy as np
import xarray as xr
import zarr

from ccic.codecs import LogBins


def test_codec():
    """
    Ensures that encoding error are small and that NAN values are
    handled correctly.
    """
    codec = LogBins(1e-4, 1e2)
    values = np.logspace(-4, 2, 10_000).astype(np.float32)
    values[0] = 1e-5
    values[-1] = np.nan
    values_enc = codec.encode(values)

    assert values_enc[0] == 0
    assert values_enc[-2] < 254
    assert values_enc[-1] == 255
    assert values_enc.dtype == np.uint8

    values_dec = codec.decode(values_enc)

    assert np.isclose(values_dec[0], 0.0)
    # Discretization error is within +/- 2% for chosen range.
    assert np.all(np.isclose(values_dec[1:-1], values[1:-1], rtol=3e-2))
    assert np.isnan(values_dec[-1])
    assert np.all(np.isfinite(values_dec[:-1]))


def test_save_dataset(tmpdir):
    """
    Tests saving an xarray.Dataset using the LogBins codec.
    """
    data = 10 ** (np.random.uniform(-3, 2, size=(1_000, 1_000))).astype(np.float32)
    dataset = xr.Dataset({
        "iwp": (("y", "x"), data)
    })

    filters = [LogBins(1e-4, 1e2)]
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=2)

    encodings = {
        "iwp": {
            "compressor": compressor,
            "filters": filters
        }
    }
    dataset.to_zarr(tmpdir / "test.zarr", encoding=encodings, compute=True)

    dataset_dec = xr.open_zarr(tmpdir / "test.zarr")

    iwp = dataset.iwp.data
    iwp_dec = dataset_dec.iwp.data

    # Discretization error is within +/- 2% for chosen range.
    assert np.all(np.isclose(iwp, iwp_dec, rtol=3e-2))

