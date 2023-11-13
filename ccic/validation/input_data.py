"""
ccic.validation.input_data
==========================

This module provides class to read input data for the cloud-radar
retrieval.
"""
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from artssat.data_provider import Fascod
from metpy.constants import dry_air_molecular_weight, water_molecular_weight
from metpy.calc import mixing_ratio_from_relative_humidity
from metpy.units import units
import numpy as np
from pansat.time import to_datetime
from pansat.products.reanalysis.era5 import ERA5Hourly
from pansat.products.ground_based.cloudnet import CloudnetProduct
from pansat.time import to_datetime64, to_datetime

import xarray as xr

def cloudnet_iwc(dbz, temp):
    """
    Calculate IWC from radar reflectivity and temperature profile using
    the formula used in the Cloudnet retrievals.

    Args:
        dbz: The radar reflectivities
        t: The temperature profile.

    Return:
        The estimated IWC in kg/m^3.
    """
    temp_c = temp - 273.5
    dbz_c = 0.944 * dbz
    log_iwc = 0.000242 * dbz_c * temp_c + 0.0699 * dbz_c + -0.0186 * temp_c + -1.63
    return (10 ** log_iwc) / 1e3


ERA5_PRODUCT = ERA5Hourly(
    'pressure',
    ['temperature', 'relative_humidity', 'geopotential', 'specific_cloud_liquid_water_content'],
)

def era5_files_in_range(path, roi, start_time, end_time):
    """
    Return a list of ERA5 files within an hour of a
    given time interval.

    Args:
        path: A path containing ERA5 files.
        start_time: The start time of the time interval.
        end_time: The end time of the time interval.

    Return:
        A list of pathlib.Path object pointing to the ERA5 files within
        the specified interval.
    """
    roi_str = "-".join(np.array(roi).astype(str))
    files = sorted(list(path.glob(f"reanalysis-era5*{roi_str}*.nc")))
    files_within = []
    for filename in files:
        time = to_datetime64(ERA5_PRODUCT.filename_to_date(filename.name))
        delta_start = time - start_time
        delta_end = time - end_time
        if ((delta_start.astype("timedelta64[s]") > -np.timedelta64(3600, "s")) and
            (delta_end.astype("timedelta64[s]") < np.timedelta64(3600, "s"))):
            files_within.append(filename)
    return files_within


class RetrievalInput(Fascod):
    """
    Class to load the retrieval input data implementing the artssat
    data provider interface.
    """
    def __init__(
            self,
            radar,
            radar_data_path,
            radar_file,
            era5_data_path,
            static_data_path,
            vertical_resolution=133.0,
            radar_resolution=100.0
    ):
        """
        Args:
            radar: A radar object representing the radar from which the input
                observations stem.
            radar_data_path: The folder containing the radar observations.
            radar_file: The specific file from which to load the input
                observations
            era5_data_path: The path containing the ERA5 data.
            static_data_path: The path containing the static retrieval data.
            vertical_resolution: The vertical resolution of the retrieval.
        """
        super().__init__("midlatitude", "summer")
        self.radar = radar
        self.radar_data_path = radar_data_path
        self.radar_file = Path(radar_file)
        self.era5_data_path = Path(era5_data_path)
        self.static_data_path = static_data_path
        self.vertical_resolution = vertical_resolution
        self.radar_resolution = radar_resolution
        self._data = None
        self.era5_data = None
        self.radar_data = None

    def _load_data(self):
        """
        Loads data for given date into the objects '_data' attribute.

        Because the data loading applies resampling caching the the loaded
        data for the in the objects '_data' attribute is crucial for
        fast loading the remaining data from that day

        Args:
            date: The date for which to load the data.
        """
        if self._data is None:
            radar_data = self.radar.load_data(
                self.radar_data_path,
                self.radar_file,
                self.static_data_path,
                vertical_resolution=self.radar_resolution
            )
            self.radar_data = radar_data

            start_time, end_time = self.radar.get_start_and_end_time(
                self.radar_data_path,
                self.radar_file
            )
            roi = self.radar.get_roi(self.radar_data_path, self.radar_file)
            era5_files = era5_files_in_range(
                self.era5_data_path,
                roi,
                start_time,
                end_time
                )
            era5_data = []
            for filename in era5_files:
                data = xr.load_dataset(filename)[{"level": slice(None, None, -1)}]
                era5_data.append(data)

            if len(era5_data) == 1:
                data_copy = era5_data[0].assign({
                    "time": era5_data[0].time + np.timedelta64(1, "h")
                })
                era5_data.append(data_copy)

            self.era5_data = xr.concat(era5_data, dim="time")

    def _interpolate_pressure(self, time):
        if self.era5_data is None:
            self._load_data()
        if self._data is None or time != self._data.time:
            radar_data = self.radar_data.interp(
                time=time,
                method="nearest",
                kwargs={"fill_value": "extrapolate"}
            )

            latitude = radar_data.latitude
            longitude = radar_data.longitude
            if latitude.ndim > 0:
                latitude = latitude[0]
                longitude = longitude[0]

            era5_data = self.era5_data.interp(
                time=time,
                latitude=latitude,
                longitude=longitude,
                method="nearest",
                kwargs={"fill_value": "extrapolate"}
            )

            era5_data.coords["altitude"] = (
                ("level",),
                era5_data.z.data / 9.81
            )
            era5_data = era5_data.swap_dims({"level": "altitude"})
            era5_data["level"] = np.log(era5_data["level"])
            altitude = np.arange(0, 20e3, self.vertical_resolution)
            era5_data = era5_data.interp(
                altitude=altitude,
                method="linear",
                kwargs={"fill_value": "extrapolate"}
            )
            era5_data["p"] = (("altitude",), np.exp(era5_data["level"].data))
            lat = radar_data.latitude
            lon = radar_data.longitude
            radar_data = radar_data.drop_vars(["latitude", "longitude"])
            self._data = xr.merge([radar_data, era5_data])
            self._data["latitude"] = lat
            self._data["longitude"] = lon
            self.interpolate_altitude(self._data.altitude.data, extrapolate=True)

    def has_data(self):
        """
        Determines whether input data for a given day is available.

        Args:
            date: datetime64 object specifying the day for which
                to run the retrieval.

        Return:
            'True' if the data is available, 'False' otherwise.
        """
        start_time, end_time = self.radar.get_start_and_end_time(
            self.radar_data_path,
            self.radar_file
        )
        roi = self.radar.get_roi(self.radar_data_path, self.radar_file)
        era5_files = era5_files_in_range(self.era5_data_path, roi, start_time, end_time)
        has_era5 = len(era5_files) > 0
        return has_era5 and (self.radar_data_path / self.radar_file).exists()

    def download_data(self):
        """
        Downloads input data for a given day.

        Args:
            date: datetime64 object specifying the day for which
                to download the input data.
        """
        start_time, end_time = self.radar.get_start_and_end_time(
            self.radar_data_path,
            self.radar_file
        )
        roi = self.radar.get_roi(self.radar_data_path, self.radar_file)
        era5_files = era5_files_in_range(self.era5_data_path, roi, start_time, end_time)
        has_era5 = len(era5_files) > 0
        if not has_era5:
            roi = self.radar.get_roi(self.radar_data_path, self.radar_file)
            variables = [
                'temperature',
                'relative_humidity',
                'geopotential',
                'specific_cloud_liquid_water_content'
            ]
            product = ERA5Hourly('pressure', variables, roi)
            product.download(
                to_datetime(start_time),
                to_datetime(end_time),
                destination=self.era5_data_path)

        if not (self.radar_data_path / self.radar_file).exists():
            self.radar.download_file(self.radar_file, self.radar_data_path)


    def get_start_and_end_time(self):
        """
        Start and end time of the input observations.
        """
        return self.radar.get_start_and_end_time(
            self.radar_data_path,
            self.radar_file
        )

    def get_radar_reflectivity(self, time):
        """
        Return radar input observations for given date.
        """
        self._interpolate_pressure(time)
        dbz = self._data.radar_reflectivity.data
        if np.any(np.isnan(dbz)) and np.any(np.isfinite(dbz)):
            start, end = np.where(np.isfinite(dbz))[0][[0, -1]]
            dbz = dbz[start:end]
        return np.maximum(self.radar.y_min, dbz)

    def get_ice_dm_x0(self, date):
        """
        First guess for $D_m$ parameter of the distribution of frozen
        hydrometeors.
        """
        n0 = 10 ** self.get_ice_n0_xa(date)
        t = self.get_temperature(date)

        dbz = self.get_radar_reflectivity(date)
        z = self.get_altitude(date)
        bins = self.get_radar_range_bins(date)
        centers = 0.5 * (bins[1:] + bins[:-1])
        dbz_i = np.interp(z, centers, dbz, left=-40, right=-40)

        iwc = cloudnet_iwc(dbz_i, t)
        dm = (256 * iwc / (n0 * np.pi * 917.0)) ** (1 / 4)
        dm[n0 < 10 ** 5] = 1e-8
        dm[dbz_i <= self.radar.y_min] = 1e-8
        return dm

    def get_ice_mass_density_x0(self, date):
        """
        First guess for mass density of frozen
        hydrometeors.
        """
        dbz = self.get_radar_reflectivity(date)
        t = self.get_temperature(date)
        z = self.get_altitude(date)
        bins = self.get_radar_range_bins(date)
        centers = 0.5 * (bins[1:] + bins[:-1])
        dbz_i = np.interp(z, centers, dbz, left=-40, right=-40)

        iwc = np.log10(cloudnet_iwc(dbz_i, t))
        iwc[dbz_i <= self.radar.y_min] = -9
        xa = self.get_ice_mass_density_xa(date)
        iwc[xa <= -12] = -12
        return iwc

    def get_rain_mass_density_x0(self, date):
        """
        First guess for $D_m$ parameter of the distribution of liquid
        hydrometeors.
        """
        t = self.get_temperature(date)
        dbz = self.get_radar_reflectivity(date)
        z = self.get_altitude(date)
        bins = self.get_radar_range_bins(date)
        centers = 0.5 * (bins[1:] + bins[:-1])
        dbz_i = np.interp(z, centers, dbz, left=-40, right=-40)

        rwc = np.log10(cloudnet_iwc(dbz_i, t))
        rwc[dbz_i < self.radar.y_min] = -9

        xa = self.get_rain_mass_density_xa(date)
        rwc[xa <= -12] = -12
        return rwc

    def get_cloud_water(self, time):
        """
        Concentraion of liquid cloud in kg/m^3
        """
        self._interpolate_pressure(time)
        p = self.get_pressure(time).data
        z = self.get_altitude(time).data

        dz = np.diff(z)
        dp_dz = np.diff(p) / dz
        m_air = - dp_dz / 9.81
        m_air_e = np.zeros(m_air.size + 1)
        m_air_e[1:-1] = 0.5 * (m_air[1:] + m_air[:-1])
        m_air_e[0] = m_air_e[1]
        m_air_e[-1] = m_air_e[-2]

        clwc = self._data.clwc.data
        m_lc = m_air_e * np.maximum(clwc, 0.0)
        return m_lc


    def get_y_radar(self, time):
        """
        Return radar input observations for given time.
        """
        return self.get_radar_reflectivity(time)

    def get_radar_range_bins(self, time):
        """
        Return range bins of radar as center points between radar measurements.
        """
        self._interpolate_pressure(time)
        dbz = self._data.radar_reflectivity.data
        range_bins = self._data.range_bins.data
        if np.any(np.isnan(dbz)) and np.any(np.isfinite(dbz)):
            start, end = np.where(np.isfinite(dbz))[0][[0, -1]]
            range_bins = range_bins[start:end + 1]
        return range_bins

    def get_y_radar_nedt(self, time):
        """
        Return nedt for radar observations.
        """
        self._interpolate_pressure(time)
        range_bins = self.get_radar_range_bins(time)
        return np.ones(range_bins.size - 1)

    def get_temperature(self, time):
        """Get temperature in the atmospheric column above the radar."""
        self._interpolate_pressure(time)
        return self._data.t.data

    def get_pressure(self, time):
        """Get the pressure in the atmospheric column above the radar."""
        self._interpolate_pressure(time)
        return self._data.p.data * 1e2

    def get_altitude(self, time):
        """Get the altitude in the atmospheric column above the radar."""
        self._interpolate_pressure(time)
        return self._data.altitude.data

    def get_surface_altitude(self, time):
        """Get the surface altitude."""
        return np.array([[0]])

    def get_radar_sensor_position(self, time):
        """Position of the radar"""
        self._interpolate_pressure(time)
        return np.array([[self._data.sensor_position]])

    def get_H2O(self, time):
        """Get H2O VMR in the atmospheric column above the radar."""
        self._interpolate_pressure(time)
        temp = (self._data.t.data - 273.15) * units.degC
        press = self._data.p.data * units.hPa
        rel = self._data.r.data * units.percent

        mmr = mixing_ratio_from_relative_humidity(press, temp, rel)
        vmr = dry_air_molecular_weight / water_molecular_weight * mmr
        return vmr.to("dimensionless")

    def get_iwc_data(self, time, timestep):
        """
        Extracts Cloudnet reference IWC retrievals.

        Args:
            time: datetime64 object specifying the day for which to extract the
                reference data.
            timestep: The temporal resolution of the data.

        Return:
            An 'xarray.Dataset' containing the Cloudnet retrieval results for the
            given day.
        """
        self._interpolate_pressure(time)
        start, end = self.get_start_and_end_time()
        times = np.arange(start, end, timestep)

        data = self.radar_data[["iwc", "iwc_reliability"]]
        z = self.get_altitude(time)
        if "range" in data:
            data = data.rename({"range": "iwc_altitude"})
        data = data.interp(
            time=times,
            iwc_altitude=z,
            method="nearest",
        )
        return data

    def get_latitude(self, time):
        self._interpolate_pressure(time)
        return self._data.latitude.data

    def get_longitude(self, time):
        self._interpolate_pressure(time)
        return self._data.longitude.data
