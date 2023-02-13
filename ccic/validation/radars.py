"""
ccic.validation.radars
======================

This module provides a Radar class and radar objects for all radars
used for the CCIC radar-only retrievals.
"""
from datetime import datetime, timedelta
from pathlib import Path

from artssat.sensor import ActiveSensor
import numpy as np
from pansat.download.providers.cloudnet import CloudnetProvider
from pansat.products.ground_based.cloudnet import CloudnetProduct
from pansat.time import to_datetime, to_datetime64
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


class CloudRadar(ActiveSensor):
    """
    The 'CloudRadar' class collects radar characteristics relevant for
    the simulation of the radar observations and provides an interface
    to load the radar data.
    """
    def __init__(
            self,
            frequency,
            sensitivity,
            line_of_sight
    ):
        """
        Args:
            frequency: The frequency of the radar in Hz
            sensitivity: The minimum detectable reflectivity in dBZ
            line_of_sight: The zenith angle of the radar. 0 for up-looking
                and 180 for down-looking.
        """
        super().__init__(
            name="radar", f_grid=frequency, range_bins=None, stokes_dimension=1
        )
        self.sensor_line_of_sight = np.array([line_of_sight])
        self.instrument_pol = [1]
        self.instrument_pol_array = [[1]]
        self.extinction_scaling = 1.0
        self.y_min = sensitivity


class CloudnetRadar(CloudRadar):
    """
    Class to identify and load Cloudnet radar data.
    """
    def __init__(self,
                 location,
                 radar_type,
                 longitude,
                 latitude,
                 frequency,
                 sensitivity):
        super().__init__(frequency, sensitivity, 0)
        self.location = location
        self.radar_type = radar_type
        self.longitude = longitude
        self.latitude = latitude
        self.product = CloudnetProduct("radar", "Radar L1B data", location)
        self.iwc_product = CloudnetProduct("iwc", "IWC product", location)
        self.provider = CloudnetProvider(self.product)

    def get_files(self, path, date):
        """
        Get files for a given date.

        Args:
            date: A numpy.datetime64 object specifying the day.

        Return:
            A list of the available input files for the given day.
        """
        pydate = to_datetime(date)
        day = pydate.timetuple().tm_yday
        files = self.provider.get_files_by_day(pydate.year, day)
        return files

    def get_start_and_end_time(self, path, filename):
        """
        Get start and end time from a Cloudnet filename.

        Args:
            filename: The filename for which to determine start and end
                times.

        Return:
            A tuple ``(start_time, end_time)`` containing the start and
            end times of the given file.
        """
        start_time = datetime.strptime(
            Path(filename).stem.split("_")[0], "%Y%m%d"
        )
        end_time = start_time + timedelta(days=1)
        return to_datetime64(start_time), to_datetime64(end_time)

    def download_file(self, filename, destination):
        """
        Download Cloudnet file.

        Args:
            date: Date specifying the day for which to download the data.
            destination: Path in which to store the downloaded data.
        """
        destination = Path(destination)
        if not destination.exists():
            destination.mkdir(exist_ok=True, parents=True)
        self.provider.download_file(filename, destination)

    def get_roi(self, *args):
        """
        Get geographical bounding box around radar.
        """
        lon = self.longitude
        lat = self.latitude
        return [lat - 0.5, lat + 0.5, lon - 0.5, lon + 0.5]

    def load_data(self, path, filename):
        """
        Load radar data from given data into xarray.Dataset

        Resamples the data to a temporal resolution of 5 minutes and a
        vertical resolution of 200m.

        Args:
            path: The path containing the Cloudnet data.
            filename: The filename of the specific file to load.

        Return:
            An 'xarray.Dataset' containing the resampled radar data.
        """
        path = Path(path)
        radar_file = path / filename
        radar_data = xr.load_dataset(radar_file)

        # Resample data
        time = radar_data.time.data.astype("datetime64[s]")
        time_bins = np.arange(
            time[0],
            time[-1] + np.timedelta64(1, "s"),
            np.timedelta64(5 * 60, "s")
        )
        height = radar_data.height.data
        range_bins = np.arange(height[0], height[-1], 100)

        refl = radar_data.Zh.data
        z = np.nan_to_num(10 ** (refl / 10), -40)
        z = resample_time_and_height(time_bins, range_bins, time, height, z)
        z = 10 * np.log10(z)

        time = time_bins[:-1] + 0.5 * (time_bins[1:] - time_bins[:-1])
        radar_range = 0.5 * (range_bins[1:] + range_bins[:-1])
        results = xr.Dataset({
            "time": (("time", ), time),
            "range": (("range",), radar_range),
            "radar_reflectivity": (("time", "range"), z),
            "range_bins": (("range_bins",), range_bins)
        })

        tmpl = "**/" + "_".join(filename.split("_")[:2]) + "_iwc-Z-T-method.nc"
        iwc_files = list(path.glob(tmpl))
        if len(iwc_files) > 0:
            iwc_data = xr.load_dataset(iwc_files[0])
            time = iwc_data.time.data
            height = iwc_data.height.data
            iwc = np.nan_to_num(iwc_data.iwc_inc_rain.data, 0.0)
            iwc = resample_time_and_height(
                time_bins, range_bins, time, height, iwc
            )
            # Resample reliable retrieval flag to reliably IWC values.
            reliability = (iwc_data.iwc_retrieval_status.data <= 3).astype(np.float64)
            reliability = resample_time_and_height(
                time_bins, range_bins, time, height, reliability
            )
            results["iwc"] = (("time", "range"), iwc)
            results["iwc"].attrs = {
                "units": "kg m-3",
                "long_name": "Ice water content",
                "comment": "Resampled Cloudnet IWC. ",
                "ancillary_variables": "iwc_reliability"
            }
            results["iwc_reliability"] = (("time", "range"), reliability)
            results["iwc_reliability"].attrs = {
                "long_name": "Reliability of ice water content retrieval.",
                "comment":
                """
                The reliability encodes the fraction of Cloudnet IWC pixels
                with retrieval status 'reliable' that were used to calculate
                the resampled IWC. Values
                """
            }

        return results


cloudnet_punta_arenas = CloudnetRadar(
    "punta-arenas",
    "mira",
    longitude=-70.883,
    latitude=-53.135,
    frequency=35e9,
    sensitivity=-40
)
cloudnet_palaiseau = CloudnetRadar(
    "palaiseau",
    "basta",
    longitude=2.212,
    latitude=47.72,
    frequency=95e9,
    sensitivity=-25
)


class ARMRadar(CloudRadar):
    """
    Class to load input data from the ARM WACR radar.
    """
    def __init__(self, longitude, latitude, frequency, sensitivity):
        super().__init__(frequency, sensitivity, 0)
        self.longitude = longitude
        self.latitude = latitude

    def get_files(self, path, date):
        """
        Get files for a given date.

        Args:
            date: A numpy.datetime64 object specifying the day.

        Return:
            A list of the available input files for the given day.
        """
        pydate = to_datetime(date)
        templ = f"**/maowacrM1.a1.{pydate.strftime('%Y%m%d')}*.nc"
        files = sorted(list(Path(path).glob(templ)))
        return files

    def get_start_and_end_time(self, path, filename):
        """
        Get start and end time from a Cloudnet filename.

        Args:
            filename: The filename for which to determine start and end
                times.
        """
        with xr.open_dataset(Path(path) / filename) as data:
            start_time = data.time[0].data
            end_time =data.time[-1].data
        return start_time, end_time


    def download_file(self, *args):
        """
        No download option is available for ARM radars.
        """
        raise Exception(
            "ARM radar does not support on-the-fly downloads."
        )

    def get_roi(self, *args):
        """
        Get geographical bounding box around radar.
        """
        lon = self.longitude
        lat = self.latitude
        return [lat - 0.5, lat + 0.5, lon - 0.5, lon + 0.5]

    def load_data(self, path, filename):
        """
        Load radar data from ARM WACR observation file into xarray.Dataset

        Resamples the data to a temporal resolution of 5 minutes and a vertical
        resolution of 100m.

        Args:
            path: The path containing the ARM radar data.
            filename: The name of the file from which to load the data.

        Return:
            An 'xarray.Dataset' containing the resampled radar data.
        """
        path = Path(path)
        radar_file = path / filename

        with xr.open_dataset(radar_file) as radar_data:
            # Resample data
            time = radar_data.time.data.astype("datetime64[s]")
            time_bins = np.arange(
                time[0],
                time[-1] + np.timedelta64(1, "s"),
                np.timedelta64(5 * 60, "s")
            )
            height = radar_data.height.data
            range_bins = np.arange(height[0], height[-1], 100)

            refl = radar_data.reflectivity.data
            z = 10 ** (np.nan_to_num(refl, self.y_min) / 10)
            z = resample_time_and_height(time_bins, range_bins, time, height, z)
            z = 10 * np.log10(z)

            time = time_bins[:-1] + 0.5 * (time_bins[1:] - time_bins[:-1])
            radar_range = 0.5 * (range_bins[1:] + range_bins[:-1])
            results = xr.Dataset({
                "time": (("time", ), time),
                "range": (("range",), radar_range),
                "radar_reflectivity": (("time", "range"), z),
                "range_bins": (("range_bins",), range_bins)
            })
        return results

    def download_era5_data(self, date, destination):
        """
        Download ERA5 data for a given day.

        Args:
            date: Date specifying the day for which to download the data.
            destination: Path in which to store the downloaded data.
        """
        py_date = to_datetime(date)
        start = datetime(py_date.year, py_date.month, py_date.day)
        end = start + timedelta(days=1)
        lon = self.longitude
        lat = self.latitude

        variables = [
            'temperature',
            'relative_humidity',
            'geopotential',
            'specific_cloud_liquid_water_content'
        ]
        product = ERA5Hourly(
            'pressure',
            variables,
            [lat - 0.5, lat + 0.5, lon - 0.5, lon + 0.5]
        )
        product.download(start, end, destination=destination)


arm_manacapuru = ARMRadar(-60.598100, 3.212970, 95e9, -25)
