"""
This script creates test data for the calculation of the monthly means.
All 2D times are filled with the value corresponding to the hour of
the day, so the monthly means should reproduce this value.
"""
from pathlib import Path
import numpy as np
import xarray as xr

output_folder = Path("gridsat_test")
start = np.datetime64("2020-01-01T00:00:00")
end = np.datetime64("2020-02-01T00:00:00")
times = xr.DataArray(np.arange(start, end, np.timedelta64(3, "h")))
n_lats = 180
n_lons = 360

for ind in range(times.size):
    year = times.dt.year[ind].data
    month = times.dt.month[ind].data
    day = times.dt.day[ind].data
    hour = times.dt.hour[ind].data
    filename = f"ccic_gridsat_{year:04}{month:02}{day:02}{hour:02}00.nc"
    dims = ("time", "latitude", "longitude")
    data = xr.Dataset({
        "latitude": (("latitude",), np.linspace(-60, 60, n_lats)),
        "longitude": (("longitude",), np.linspace(-180, 180, n_lons)),
        "time": (("time",), [times[ind].data]),
        "ci_bounds": (("ci_bounds",), np.array([0.05, 0.95])),
        "cloud_prob_2d": (dims, hour * np.ones((1, n_lats, n_lons))),
        "inpainted": (dims, np.ones((1, n_lats, n_lons))),
        "p_tiwp": (dims, hour / 24 * np.ones((1, n_lats, n_lons))),
        "tiwp": (dims, hour * np.ones((1, n_lats, n_lons))),
        "tiwp_ci": (dims + ("ci_bounds",), hour  * np.ones((1, n_lats, n_lons, 2))),
    })

    # Mask a quarter of all values.
    mask = np.random.rand(n_lats, n_lons) > 0.75
    data["tiwp"].data[:, mask] = np.nan
    data["cloud_prob_2d"].data[:, mask] = np.nan

    for var in data.variables:
        data[var].attrs["long_name"] = "long_name"

    data.to_netcdf(output_folder / filename)
