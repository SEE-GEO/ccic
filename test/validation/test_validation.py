from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import xarray as xr
from ccic.validation import resample_data


def test_resample_data():
    """
    Test resampling or flight campaign data to CCIC grids for
    30 minute temporal resolution.
    """
    lat_bins = np.linspace(0, 180, 181)[::-1]
    lon_bins = np.linspace(0, 180, 181)

    lons = np.arange(180) + 0.5
    lats = np.arange(180) + 0.5
    mins = np.arange(180) + 0.5
    height = np.linspace(0, 20e3, 181)
    height = 0.5 * (height[1:] + height[:-1])
    start_time = np.datetime64("2020-01-01T22:45:00")
    time = start_time + (60 * np.arange(180)).astype("timedelta64[s]")

    test_data = xr.Dataset({
        "longitude": (("time",), lons),
        "latitude": (("time",), lats),
        "time": (("time",), time),
        "altitude": (("time",), height),
        "values": (("time"), lons)
    })

    with TemporaryDirectory() as tmp:
        resample_data(
            test_data,
            ["values"],
            lon_bins,
            lat_bins,
            tmp,
            "test_{year}{month:02}{day:02}{hour:02}.nc"
        )

        assert (Path(tmp) / "test_2020010123.nc").exists()
        data = xr.load_dataset(Path(tmp) / "test_2020010123.nc")
        valid_times, valid_lats, valid_lons, valid_alts = np.where(
            data["values"].data >= 0.0
        )
        assert valid_times.min() == 0
        assert len(valid_times) == 60
        assert len(valid_lats) == 60
        assert valid_lons.min() == 0
        assert len(valid_lons) == 60
        assert valid_alts.min() == 0
        assert len(valid_alts) == 60
        assert data.time[0] == np.datetime64("2020-01-01T23:00:00", "ns")

        assert (Path(tmp) / "test_2020010200.nc").exists()
        assert (Path(tmp) / "test_2020010201.nc").exists()
        data = xr.load_dataset(Path(tmp) / "test_2020010201.nc")
        valid_times, valid_lats, valid_lons, valid_alts = np.where(
            data["values"].data >= 0.0
        )
        assert valid_times.min() == 0
        assert len(valid_times) == 60
        assert len(valid_lats) == 60
        assert valid_lons.min() == 120
        assert len(valid_lons) == 60
        assert valid_alts.min() == 13
        assert data.time[0] == np.datetime64("2020-01-02T01:00:00", "ns")

        assert not (Path(tmp) / "test_2020010202.nc").exists()


