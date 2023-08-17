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
from scipy.stats import binned_statistic, binned_statistic_2d
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
        radar_product = CloudnetProduct("radar", "Radar L1B data", location)
        self.radar_provider = CloudnetProvider(radar_product)
        iwc_product = CloudnetProduct("iwc", "IWC product", location)
        self.iwc_provider = CloudnetProvider(iwc_product)

    @property
    def instrument_name(self):
        """The radar name is used to label retrieval results."""
        return f"cloudnet_{self.location}"

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
        files = self.radar_provider.get_files_by_day(pydate.year, day)
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
        self.radar_provider.download_file(filename.name, destination / filename)

        start, end = self.get_start_and_end_time(destination, filename)
        pydate = to_datetime(start)
        day = pydate.timetuple().tm_yday
        iwc_files = self.iwc_provider.get_files_by_day(pydate.year, day)
        for filename in iwc_files:
            if not (destination / filename).exists():
                self.iwc_provider.download_file(filename, destination / filename)

    def get_roi(self, *args):
        """
        Get geographical bounding box around radar.
        """
        lon = self.longitude
        lat = self.latitude
        return [lat - 0.5, lat + 0.5, lon - 0.5, lon + 0.5]

    def load_data(self,
                  path,
                  filename,
                  *args,
                  vertical_resolution=100.0):
        """
        Load radar data from given data into xarray.Dataset

        Resamples the data to a temporal resolution of 1 minute and a
        vertical resolution of 100m.

        Args:
            path: The path containing the Cloudnet data.
            filename: The filename of the specific file to load.
            vertical_resolution: The vertical resolution to which the radar
            observations will be resampled.

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
        range_bins = np.arange(height[0], height[-1], vertical_resolution)

        refl = np.nan_to_num(radar_data.Zh.data, copy=True, nan=self.y_min)
        z = 10 ** (refl / 10)
        z = resample_time_and_height(time_bins, range_bins, time, height, z)
        z = 10 * np.log10(z)

        time = time_bins[:-1] + 0.5 * (time_bins[1:] - time_bins[:-1])
        radar_range = 0.5 * (range_bins[1:] + range_bins[:-1])
        sensor_position = range_bins[0] * np.ones(time.size)
        results = xr.Dataset({
            "time": (("time", ), time),
            "range": (("range",), radar_range),
            "radar_reflectivity": (("time", "range"), z),
            "range_bins": (("range_bins",), range_bins),
            "latitude": (("latitude",), [self.latitude]),
            "longitude": (("longitude",), [self.longitude]),
            "sensor_position": (("time",), sensor_position)
        })

        parts = Path(filename).stem.split("_")[:2]
        tmpl = "**/" + "_".join(parts) + "_iwc-Z-T-method.nc"
        iwc_files = list(path.glob(tmpl))
        if len(iwc_files) > 0:
            iwc_data = xr.load_dataset(iwc_files[0])
            time = iwc_data.time.data
            height = iwc_data.height.data
            iwc = np.nan_to_num(iwc_data.iwc_inc_rain.data, copy=True)
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
    latitude=48.71,
    frequency=95e9,
    sensitivity=-25
)

cloudnet_palaiseau.instrument_pol = [6]
cloudnet_palaiseau.instrument_pol_array = [[6]]
cloudnet_palaiseau.stokes_dimension = 2


class ARMRadar(CloudRadar):
    """
    Class to load input data from the ARM WACR radar.
    """
    def __init__(self, campaign, longitude, latitude, frequency, sensitivity):
        super().__init__(frequency, sensitivity, 0)
        self.campaign = campaign
        self.longitude = longitude
        self.latitude = latitude

    @property
    def instrument_name(self):
        """The radar name is used to label retrieval results."""
        return f"arm_{self.campaign}"

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
        return [path.name for path in files]

    def get_start_and_end_time(self, path, filename):
        """
        Get start and end time from a Cloudnet filename.

        Args:
            filename: The filename for which to determine start and end
                times.
        """
        with xr.open_dataset(Path(path) / filename) as data:
            start_time = data.time[0].data
            end_time = data.time[-1].data
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

    def load_data(self, path, filename, *args, vertical_resolution=100.0):
        """
        Load radar data from ARM WACR observation file into xarray.Dataset

        Resamples the data to a temporal resolution of 1 minute and a vertical
        resolution of 100m.

        Args:
            path: The path containing the ARM radar data.
            filename: The name of the file from which to load the data.
            vertical_resolution: The vertical resolution to which the radar
            observations will be resampled.

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
            range_bins = np.arange(height[0], height[-1], vertical_resolution)

            refl = np.nan_to_num(radar_data.reflectivity.data, copy=True, nan=self.y_min)
            z = 10 ** (refl / 10)
            z = resample_time_and_height(time_bins, range_bins, time, height, z)
            z = 10 * np.log10(z)

            time = time_bins[:-1] + 0.5 * (time_bins[1:] - time_bins[:-1])
            radar_range = 0.5 * (range_bins[1:] + range_bins[:-1])
            sensor_position = range_bins[0] * np.ones(time.size)
            results = xr.Dataset({
                "time": (("time", ), time),
                "range": (("range",), radar_range),
                "radar_reflectivity": (("time", "range"), z),
                "range_bins": (("range_bins",), range_bins),
                "latitude": (("latitude",), [self.latitude]),
                "longitude": (("longitude",), [self.longitude]),
                "sensor_position": (("time",), sensor_position)
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


arm_manacapuru = ARMRadar("manacapuru", -60.598100, -3.212970, 95e9, -25)


class NASACRS(CloudRadar):
    """
    Class to load input data from the NASA cloud radar system.
    """
    def __init__(self, campaign, dem):
        """
        Args:
            dem: Filename of an NetCDF file containing surface elevation data
                for the domain of the field campaign.
        """
        # 95 GHz radar, air borne
        super().__init__(95e9, -30, 180)
        self.campaign = campaign
        self.instrument_pol = [1]
        self.instrument_pol_array = [[1]]
        self.dem = dem

    @property
    def instrument_name(self):
        """The radar name is used to label retrieval results."""
        return f"crs_{self.campaign}"

    def get_files(self, path, date):
        """
        Get files for a given date.

        Args:
            date: A numpy.datetime64 object specifying the day.

        Return:
            A list of the available input files for the given day.
        """
        pydate = to_datetime(date)
        templ = f"**/olympex_CRS_{pydate.strftime('%Y%m%d')}*.nc"
        files = sorted(list(Path(path).glob(templ)))
        return [path.name for path in files]

    def get_start_and_end_time(self, path, filename):
        """
        Get start and end time from a Cloudnet filename.

        Args:
            filename: The filename for which to determine start and end
                times.
        """
        parts = Path(filename).stem.split("-")
        start_time = parts[0].split("_")[2:]
        start_time = datetime.strptime("".join(start_time), "%Y%m%d%H%M%S")
        end_time = parts[1].split("_")[:2]
        end_time = datetime.strptime("".join(end_time), "%Y%m%d%H%M%S")
        return to_datetime64(start_time), to_datetime64(end_time)

    def download_file(self, *args):
        """
        No download option is available for NASA CRS radars.
        """
        raise Exception(
            "NASA CRS radar does not support on-the-fly downloads."
        )

    def get_roi(self, path, filename):
        """
        Get geographical bounding box around radar.
        """
        with xr.open_dataset(path / filename) as data:
            lat_min = data.lat.data.min()
            lat_max = data.lat.data.max()
            lon_min = data.lon.data.min()
            lon_max = data.lon.data.max()

        # At least two ERA5 grid points needed for interpolation.
        if lat_max - lat_min < 0.4:
            lat_max += 0.1
            lat_min -= 0.1
        if lon_max - lon_min < 0.4:
            lon_max += 0.1
            lat_min -= 0.1
        return [lat_min, lat_max, lon_min, lon_max,]

    def load_data(
            self,
            path,
            filename,
            static_data_path,
            vertical_resolution=100
    ):
        """
        Load radar data from NASA CRS observation file into xarray.Dataset

        Resamples the data to a temporal resolution of 1 minute and a vertical
        resolution of 100m.

        Args:
            path: The path containing the CRS diles.
            filename: The name of the file from which to load the data.
            static_data_path: Path to the folder containing the surface elevation
                map for the campaign.
            vertical_resolution: The vertical resolution to which the radar
            observations will be resampled.

        Return:
            An 'xarray.Dataset' containing the resampled radar data.
        """
        path = Path(path)
        radar_file = path / filename

        start_time, _ = self.get_start_and_end_time(path, filename)

        dem = xr.load_dataset(Path(static_data_path) / self.dem)

        with xr.open_dataset(radar_file) as radar_data:

            # Sanity check data
            valid = radar_data.altitude.data >= 0.0
            radar_data = radar_data[{"timed": valid}]

            # Resample data
            latitude = radar_data.lat
            longitude = radar_data.lon
            # Elevation data may be missing over ocean. Fill with
            # 0.
            z_surf = dem.elevation.interp(
                latitude=latitude,
                longitude=longitude,
                method="nearest",
                kwargs={"fill_value": 0}
            )

            # timd is in hours since 0 UTC
            start = start_time.astype("datetime64[D]").astype("datetime64[s]")
            time = start + (radar_data.timed.data * 3600).astype("timedelta64[s]")
            time_bins = np.arange(
                time[0],
                time[-1] + np.timedelta64(1, "s"),
                np.timedelta64(20, "s")
            )
            height = (
                radar_data.altitude -
                radar_data.range -
                z_surf
            ).data
            time = np.broadcast_to(time[..., None], height.shape).copy()

            # Discard lowest 500m to get rid of ground clutter.
            range_bins = np.arange(500, height[:, 0].min() - 500, vertical_resolution)

            refl = np.nan_to_num(radar_data.zku.data, copy=self.y_min, nan=self.y_min)
            z = 10 ** (refl / 10)

            excessive_roll = np.abs(radar_data["roll"].data) > 5
            height[excessive_roll] = np.nan

            z = binned_statistic_2d(
                time.ravel().astype(np.float64),
                height.ravel(),
                z.ravel(),
                bins=(time_bins.astype(np.float64), range_bins)
            )[0]
            z = 10 * np.log10(z)

            time_f = time[:, 0].astype(np.float64)

            latitude = binned_statistic(
                time_f,
                latitude.data,
                bins=time_bins.astype(np.float64))[0]
            longitude = binned_statistic(
                time_f,
                longitude.data.astype(np.float64),
                bins=time_bins.astype(np.float64)
            )[0]
            sensor_position = binned_statistic(
                time_f,
                radar_data.altitude.data,
                bins=time_bins.astype(np.float64)
            )[0]

            time = time_bins[:-1] + 0.5 * (time_bins[1:] - time_bins[:-1])
            radar_range = 0.5 * (range_bins[1:] + range_bins[:-1])
            results = xr.Dataset({
                "time": (("time", ), time),
                "range": (("range",), radar_range),
                "radar_reflectivity": (("time", "range"), z),
                "range_bins": (("range_bins",), range_bins),
                "latitude": (("time",), latitude),
                "longitude": (("time",), longitude),
                "sensor_position": (("time",), sensor_position)
            })
        return results


crs_olympex = NASACRS("olympex", "elevation_olympex.nc")


class Rasta(CloudRadar):
    """
    Class to load input data from the airborne RASTA cloud radar.
    """
    def __init__(self, campaign, los, dem):
        """
        Args:
            campaign: Campaign identifier
            los: The zenith angle of the line of sight: 180 for down-looking,
                 0 for up-looking.
            dem: Filename of an NetCDF file containing surface elevation data
                for the domain of the field campaign.
        """
        if not los in [0.0, 180.0]:
            raise ValueError("RASTA line-of-sight (LOS) must be 0 or 180.")

        super().__init__(95e9, -40, los)
        self.los = los
        self.campaign = campaign
        self.instrument_pol = [1]
        self.instrument_pol_array = [[1]]
        self.dem = dem

    @property
    def instrument_name(self):
        """The radar name is used to label retrieval results."""
        direction = "up"
        if self.los > 0.0:
            direction = "down"
        return f"rasta_{direction}_{self.campaign}"

    def get_files(self, path, date):
        """
        Get files for a given date.

        Args:
            date: A numpy.datetime64 object specifying the day.

        Return:
            A list of the available input files for the given day.
        """
        pydate = to_datetime(date)
        templ = f"**/HAIC_DARWIN_{pydate.strftime('%Y%m%d')}*.nc"
        files = sorted(list(Path(path).glob(templ)))
        return [path.name for path in files]

    def get_start_and_end_time(self, path, filename):
        """
        Get start and end time from a Cloudnet filename.

        Args:
            filename: The filename for which to determine start and end
                times.
        """
        parts = Path(filename).stem.split("_")
        start_day = to_datetime64(datetime.strptime(parts[2], "%Y%m%d"))
        start_day = start_day.astype("datetime64[s]")
        with xr.open_dataset(path / filename) as data:
            start_time = start_day + np.timedelta64(int(data.time[0] * 3600), "s")
            end_time = start_day + np.timedelta64(int(data.time[-1] * 3600), "s")
        return start_time, end_time

    def download_file(self, *args):
        """
        No download option is available for RASTA data.
        """
        raise Exception(
            "RASTA S radar does not support on-the-fly downloads."
        )

    def get_roi(self, path, filename):
        """
        Get geographical bounding box around radar.
        """
        with xr.open_dataset(path / filename) as data:
            lat_min = data.latitude.data.min()
            lat_max = data.latitude.data.max()
            lon_min = data.longitude.data.min()
            lon_max = data.longitude.data.max()

        # At least two ERA5 grid points needed for interpolation.
        if lat_max - lat_min < 0.4:
            lat_max += 0.1
            lat_min -= 0.1
        if lon_max - lon_min < 0.4:
            lon_max += 0.1
            lat_min -= 0.1
        return [lat_min, lat_max, lon_min, lon_max,]

    def load_data(self,
                  path,
                  filename,
                  static_data_path,
                  vertical_resolution=100.0):
        """
        Load radar data from NASA CRS observation file into xarray.Dataset

        Resamples the data to a temporal resolution of 1 minute and a vertical
        resolution of 100m.

        Args:
            path: The path containing the CRS diles.
            filename: The name of the file from which to load the data.
            static_data_path: Path to the folder containing the surface elevation
                map for the campaign.
            vertical_resolution: The vertical resolution to which the
                observations will be resampled.

        Return:
            An 'xarray.Dataset' containing the resampled radar data.
        """
        path = Path(path)
        radar_file = path / filename

        start_time, _ = self.get_start_and_end_time(path, filename)

        dem = xr.load_dataset(Path(static_data_path) / self.dem)

        with xr.open_dataset(radar_file) as radar_data:

            # Sanity check data
            valid = radar_data.altitude.data >= 0.0
            radar_data = radar_data[{"time": valid}]

            # Resample data
            latitude = radar_data.latitude
            longitude = radar_data.longitude
            # Elevation data may be missing over ocean. Fill with
            # 0.
            dem.elevation.data[:] = np.nan_to_num(dem.elevation.data, True, 0.0)
            z_surf = dem.elevation.interp(
                latitude=latitude,
                longitude=longitude,
                method="nearest",
                kwargs={"fill_value": 0}
            )
            altitude = radar_data.altitude.data * 1e3

            # timd is in hours since 0 UTC
            start = start_time.astype("datetime64[D]").astype("datetime64[s]")
            time = start + (radar_data.time.data * 3600).astype("timedelta64[s]")
            time_bins = np.arange(
                time[0],
                time[-1] + np.timedelta64(1, "s"),
                np.timedelta64(20, "s")
            )
            if self.los > 0.0:
                height = (altitude[:, None] - radar_data.height_2D.data * 1e3)
                clutter_limit = altitude - z_surf.data - 1e3
                height[height >= clutter_limit[:, None]] = np.nan
            else:
                height = (radar_data.height_2D.data * 1e3 - altitude[:, None])
                atm_limit = 20e3 - altitude - 500
                height[height >= atm_limit[:, None]] = np.nan
            time = np.broadcast_to(time[..., None], height.shape).copy()

            # Discard lowest 500m to get rid of ground clutter.
            range_bins = np.arange(0, 10e3, vertical_resolution)

            refl = np.nan_to_num(
                radar_data.Z_vertical.data,
                copy=self.y_min,
                nan=self.y_min
            )
            z = 10 ** (refl / 10)

            misaligned = np.any(
                np.abs(np.cos(np.deg2rad(radar_data.elevation_hor_vertical))) > 0.075,
                1
            )
            height[misaligned] = np.nan

            z = binned_statistic_2d(
                time.ravel().astype(np.float64),
                height.ravel(),
                z.ravel(),
                bins=(time_bins.astype(np.float64), range_bins)
            )[0]
            z = 10 * np.log10(z)

            time_f = time[:, 0].astype(np.float64)

            latitude = binned_statistic(
                time_f,
                latitude.data,
                bins=time_bins.astype(np.float64))[0]
            longitude = binned_statistic(
                time_f,
                longitude.data.astype(np.float64),
                bins=time_bins.astype(np.float64)
            )[0]
            altitude_binned = binned_statistic(
                time_f,
                altitude.astype(np.float64),
                bins=time_bins.astype(np.float64)
            )[0]
            sensor_position = binned_statistic(
                time_f,
                altitude,
                bins=time_bins.astype(np.float64)
            )[0]

            # IWC
            atm_bins = np.linspace(0, 20, 101)
            iwc = np.nan_to_num(radar_data.iwc_ret.data, True, 0.0)
            apf = radar_data.attenuation_phase_flag.data
            iwc[apf > 3] = np.nan
            iwc = binned_statistic_2d(
                time.ravel().astype(np.float64),
                radar_data.height_2D.data.ravel(),
                iwc.ravel(),
                bins=(time_bins.astype(np.float64), atm_bins)
                )[0]
            iwc_zt = np.nan_to_num(radar_data.iwc_IWC_Z_T.data, True, 0.0)
            iwc_zt[apf > 3] = np.nan
            iwc_zt = binned_statistic_2d(
                time.ravel().astype(np.float64),
                radar_data.height_2D.data.ravel(),
                iwc_zt.ravel(),
                bins=(time_bins.astype(np.float64), atm_bins)
                )[0]
            altitude = 0.5 * (atm_bins[1:] + atm_bins[:-1]) * 1e3

            time = time_bins[:-1] + 0.5 * (time_bins[1:] - time_bins[:-1])
            radar_range = 0.5 * (range_bins[1:] + range_bins[:-1])

            if self.los > 0:
                range_bins = altitude_binned[:, None] - range_bins[None, :]
                range_bins = np.flip(range_bins, 1)
                radar_range = np.flip(radar_range, 0)
                z = np.flip(z, 1)

            else:
                range_bins = altitude_binned[:, None] + range_bins[None, :]

            results = xr.Dataset({
                "time": (("time", ), time),
                "range": (("range",), radar_range),
                "radar_reflectivity": (("time", "range"), z),
                "range_bins": (("time", "bins",), range_bins),
                "latitude": (("time",), latitude),
                "longitude": (("time",), longitude),
                "sensor_position": (("time",), sensor_position),
                "iwc_altitude": (("iwc_altitude",), altitude),
                "iwc": (("time", "iwc_altitude"), iwc),
                "iwc_zt": (("time", "iwc_altitude"), iwc_zt),
                "iwc_reliability": (("time", "iwc_altitude"), np.ones_like(iwc)),
            })

        return results

rasta_haic_up = Rasta("haic", 0.0, "elevation_haic.nc")
rasta_haic_down = Rasta("haic", 180.0, "elevation_haic.nc")
