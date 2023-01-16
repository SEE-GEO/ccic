"""
ccic.validation.input_data
==========================

This module provides class to read input data for the cloud-radar
retrieval.
"""
from pathlib import Path

from artssat.data_provider import DataProviderBase
from metpy.constants import dry_air_molecular_weight, water_molecular_weight
from metpy.calc import mixing_ratio_from_relative_humidity
from metpy.units import units
import numpy as np
from pansat.time import to_datetime
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
    time = time.astype(time_bins.dtype)
    height = height.astype(height_bins.dtype)
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
    def __init__(self, location, radar_type, longitude, latitude, elevation):
        self.location = location
        self.radar_type = radar_type
        self.longitude = longitude
        self.latitude = latitude
        self.elevation = elevation

    def load_data(self, path, date, iwc_path=None):
        """
        Load radar data from given data into xarray.Dataset

        This function also resamples the data to a temporal resolution
        of 5 minutes and a vertical resolution of 900m.

        Args:
            path: The path containing the Cloudnet data.
            date: A date specifying the day from which to load the Cloudnet
                data.
            iwc_path: Path for IWC files if available. If given, the IWC
                data will be loaded, resampled to the resolution of the
                radar data and included in the output data.

        Return:
            An 'xarray.Dataset' containing the resampled radar data.
        """
        path = Path(path)

        pydate = to_datetime(date)
        tmpl = f"**/{pydate.strftime('%Y%m%d')}_{self.location}_{self.radar_type}.nc"
        radar_file = next(iter(path.glob(tmpl)))
        radar_data = xr.load_dataset(radar_file)


        # Resample data
        time = radar_data.time.data.astype("datetime64[s]")
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

        time = time_bins[:-1] + 0.5 * (time_bins[1:] - time_bins[:-1])
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

cloudnet_palaiseau = CloudnetRadar("palaiseau", "basta", 2.212, 47.72, 156.0)


class RetrievalInput(DataProviderBase):
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
        super().__init__()
        self.radar = radar
        self.radar_data_path = Path(radar_data_path)
        self.era5_data_path = Path(era5_data_path)
        self.iwc_data_path = iwc_data_path
        if self.iwc_data_path is not None:
            self.iwc_data_path = Path(self.iwc_data_path)
        self._data = None

    def _load_data(self, date):
        """
        Load data for given date into the objects '_data' attribute.
        """
        # Need to load data if it's from another day or no data has been
        # loaded yet.
        if self._data is not None:
            data_day = self._data.time.astype("datetime64[D]")
            date_day = date.astype("datetime64[D]")
            d_t = np.abs(data_day - date_day).max()
        else:
            d_t = 0
        if self._data is None or d_t > 0:
            radar_data = self.radar.load_data(
                self.radar_data_path,
                date,
                iwc_path=self.iwc_data_path
            )

            pydate = to_datetime(date)
            tmpl = f"**/*era5*{pydate.strftime('%Y%m%d')}*.nc"
            era5_files = sorted(list(self.era5_data_path.glob(tmpl)))
            era5_data = []
            for filename in era5_files:
                data = xr.load_dataset(filename)[{"level": slice(None, None, -1)}]
                data = data.interp({
                    "latitude": self.radar.latitude,
                    "longitude": self.radar.longitude,
                }, method="nearest")
                data.coords["height"] = (("level",), data.z[0].data / 9.81)
                data = data.swap_dims({"level": "height"})
                data["p"] = (("height",), np.log(data.level.data))
                data = data.interp(height=radar_data.height.data)
                era5_data.append(data)
            era5_data = xr.concat(era5_data, dim="time")
            era5_data.p.data = np.exp(era5_data.p.data)

            era5_data = era5_data.interp(
                time=radar_data.time,
                method="nearest",
                kwargs={"fill_value": "extrapolate"}
            )
            self._data = xr.merge([era5_data, radar_data])

    def get_radar_reflectivity(self, date):
        """
        Return radar input observations for given date.
        """
        self._load_data(date)
        dbz = self._data.radar_reflectivity.interp(
            time=date,
            method="nearest",
            kwargs={"fill_value": "extrapolate"}
        ).data
        return np.maximum(-40, dbz)

    def get_temperature(self, date):
        """Get temperature in the atmospheric column above the radar."""
        self._load_data(date)
        return self._data.t.interp(
            time=date,
            method="nearest",
            kwargs={"fill_value": "extrapolate"}
        ).data

    def get_pressure(self, date):
        """Get the pressure in the atmospheric column above the radar."""
        self._load_data(date)
        print(self._data)
        return self._data.p.interp(
            time=date,
            method="nearest",
            kwargs={"fill_value": "extrapolate"}
        ).data * 1e2

    def get_altitude(self, date):
        """Get the pressure in the atmospheric column above the radar."""
        self._load_data(date)
        return self._data.z.interp(
            time=date,
            method="nearest",
            kwargs={"fill_value": "extrapolate"}
        ).data

    def get_h2o(self, date):
        """Get H2O VMR in the atmospheric column above the radar."""
        self._load_data(date)
        data = self._data.interp(
            time=date,
            method="nearest",
            kwargs={"fill_value": "extrapolate"}
        )
        temp = (data.t.data - 273.15) * units.degC
        press = data.p.data * units.hPa
        rel = data.r.data

        print(temp, press, rel)

        mmr = mixing_ratio_from_relative_humidity(press, temp, rel)
        return dry_air_molecular_weight / water_molecular_weight * mmr
