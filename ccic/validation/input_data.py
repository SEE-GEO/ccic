"""
ccic.validation.input_data
==========================

This module provides class to read input data for the cloud-radar
retrieval.
"""
from pathlib import Path

from pansat.time import to_datetime
import numpy as np
import xarray as xr


def resample_time_and_height(
        time_bins,
        height_bins,
        time,
        height,
        input_data
):
    """
    Resample data along time and height.

    Args:
        time_bins: Bin boundaries of the time to resample the data to.
        height_bins: Bin boundaries of the heights to resample the data to.
        time: 1D array of size 'n_times' containing the times at which the
            input data was sampled.
        height: 1D array of size 'n_heights' containing the heights at which the
            input data was sampled.
        data: 2D array of shape '(n_times, n_heights) containing the input
            data to resample.

    Return:
        A 2D array of shape time_bins.size - 1 x height_bins.size - 1.
    """
    time, height = np.meshgrid(time, height, indexing="ij")
    tot = np.histogram2d(
        time.ravel(),
        height.ravel(),
        bins=(time_bins, height_bins),
        weights=input_data.ravel()
    )[0]
    cts = np.histogram2d(
        time.ravel(),
        height.ravel(),
        bins=(time_bins, height_bins),
    )[0]
    return tot / cts


class CloudnetRadar:
    """
    Class to identify and load Cloudnet radar data.
    """
    def __init__(self, location, radar_type):
        self.location = location
        self.radar_type = radar_type

    def load_data(self, path, date, iwc_path=None):
        path = Path(path)

        pydate = to_datetime(date)
        tmpl = f"**/{pydate.strftime('%Y%m%d')}_{self.location}_{self.radar_type}.nc"
        radar_file = next(iter(path.glob(tmpl)))
        radar_data = xr.load_dataset(radar_file)


        # Resample data
        time = radar_data.time.data
        time_bins = np.arange(
            time[0],
            time[-1] + np.timedelta64(1, "s"),
            np.timedelta64(5 * 60, "s")
        )
        height = radar_data.height.data
        height_bins = np.arange(height[0], height[-1] + 200, 200.0)

        refl = radar_data.Zh.data
        z = np.nan_to_num(10 ** refl, 0.0)
        z = resample_time_and_height(time_bins, height_bins, time, height, z)
        z = np.log10(z)

        time = time_bins[0] + 0.5 * (time_bins[1:] - time_bins[:-1])
        height = 0.5 * (height_bins[1:] + height_bins[:-1])
        results = xr.Dataset({
            "time": (("time", ), time),
            "height": (("height",), height),
            "radar_reflectivity": (("time", "height"), z)
        })

        if iwc_path is not None:
            tmpl = f"**/{pydate.strftime('%Y%m%d')}_{self.location}*.nc"
            iwc_data = xr.load_dataset(next(iter(path.glob(tmpl))))
            time = iwc_data.time.data
            height = iwc_data.height.data
            iwc = iwc_data.iwc.data
            iwc = resample_time_and_height(
                time_bins, height_bins, time, height, iwc
            )
            results["iwc"] = (("time", "height"), iwc)

        return results


class RetrievalInput:
    """
    Class to load the retrieval input data implementing the artssat
    data provider interface.
    """
    def __init__(
            self,
            radar,
            radar_data_path,
            era5_data_path,
            iwc_data_path=None
    ):
        self.radar = radar
        self.radar_data_path = Path(radar_data_path)
        self.era5_data_path = Path(era5_data_path)
        self.iwc_data_path = Path(iwc_data_path)
        self._data = None

    def _load_data(self, date):
        """
        Load data for given date into the objects '_data' attribute.
        """
        # Need to load data if it's from another day or no data has been
        # loaded yet.
        if self._data is not None:
            data_day = self._data.time.astyoe("datetime64[d]")
            date_day = date.astyoe("datetime64[d]")
            d_t = np.abs(data_day - date_day)
        else:
            d_t = 0
        if self._data is None or d_t > 0:
            pydate = to_datetime(date)
            tmpl = f"**/*era5*{pydate.strftime('%Y%m%d%H')}*.nc"
            era5_files = sorted(list(self.era5_files.glob(tmpl)))
            era5_data = xr.concat([
                xr.load_dataset(filename) for filename in era5_files
            ], dim="time")

            radar_data = self.radar.load_data(
                self.radar_data_path,
                date,
                iwc_path=self.iwc_data_path
            )
            radar_data.Zh.data = np.nan_to_num(10 ** radar_data.Zh.data, 0.0)
            radar_data = radar_data.resample({"time": "5min"}).mean()
            radar_data.Zh.data = np.log10(radar_data.Zh.data)

            self._data = xr.merge([era5_data, radar_data])

    def get_input_observations(self, date):
        """
        Return radar input observations for given date.
        """
        self._load_data(date)
        return self._data.Zh.interp(time=date).data
