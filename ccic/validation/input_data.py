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
from pansat.time import to_datetime64

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
    [lat - 0.5, lat + 0.5, lon - 0.5, lon + 0.5]
)

def era5_file_in_range(path, start_time, end_time):
    files = sorted(list(path.glob("reanalysis-era5*.nc")))
    files_within = []
    for filename in files:
        time = to_datetime64(ERA5_PRODUCT.filename_to_date(filename.name))
        delta = start_time
        if delta < 36




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
        self.product = CloudnetProduct("radar", "Radar L1B data", location)
        self.iwc_product = CloudnetProduct("iwc", "IWC product", location)
        self.y_min = -40

    def has_data(self, path, date):
        """
        Determine whether input files are available for the given date.

        Args:
            path: Root of the directory tree in which to look for input
                files.
            date: The date for which to run the retrieval.

        Return:
            A boolean indicating whether the data is available.
        """
        path = Path(path)
        pydate = to_datetime(date)
        tmpl = f"**/{pydate.strftime('%Y%m%d')}_{self.location}_{self.radar_type}.nc"
        has_radar = len(list(path.glob(tmpl))) > 0
        tmpl = f"**/{pydate.strftime('%Y%m%d')}_*iwc*.nc"
        has_iwc = len(list(path.glob(tmpl))) > 0
        return has_radar and has_iwc

    def load_data(self, path, date):
        """
        Load radar data from given data into xarray.Dataset

        This function also resamples the data to a temporal resolution
        of 5 minutes and a vertical resolution of 200m.

        Args:
            path: The path containing the Cloudnet data.
            date: A date specifying the day from which to load the Cloudnet
                data.

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

        tmpl = f"**/{pydate.strftime('%Y%m%d')}_{self.location}*iwc*.nc"
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

    def download_radar_data(self, date, destination):
        """
        Download Radar L1B and IWC data for a given day.

        Args:
            date: Date specifying the day for which to download the data.
            destination: Path in which to store the downloaded data.
        """
        end = to_datetime(date)
        start = end - timedelta(days=1)
        self.product.download(
            start,
            end,
            destination=destination,
        )
        self.iwc_product.download(
            start,
            end,
            destination=destination,
        )

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
        product = ERA5Hourly(
            'pressure',
            ['temperature', 'relative_humidity', 'geopotential', 'specific_cloud_liquid_water_content'],
            [lat - 0.5, lat + 0.5, lon - 0.5, lon + 0.5]
        )
        product.download(start, end, destination=destination)


cloudnet_palaiseau = CloudnetRadar("palaiseau", "basta", 2.212, 47.72, 156.0)
cloudnet_galati = CloudnetRadar("galati", "basta", 28.037, 45.435, 40.0)
cloudnet_punta_arenas = CloudnetRadar("punta-arenas", "mira", -70.883, -53.135, 9)


class ARMRadar:
    """
    Class to load input data from the ARM WACR radar.
    """
    def __init__(self, longitude, latitude, elevation):
        self.location = "manacapuru"
        self.longitude = longitude
        self.latitude = latitude
        self.elevation = elevation
        self.y_min = -25

    def has_data(self, path, date):
        """
        Determine whether input files are available for the given date.

        Args:
            path: Root of the directory tree in which to look for input
                files.
            date: The date for which to run the retrieval.

        Return:
            A boolean indicating whether the data is available.
        """
        path = Path(path)
        pydate = to_datetime(date)
        date_str = pydate.strftime("%Y%m%d")
        tmpl = f"**/maowacrM1.a1.{date_str}*.nc"
        has_radar = len(list(path.glob(tmpl))) > 0
        return has_radar

    def load_data(self, path, date):
        """
        Load radar data from given data into xarray.Dataset

        This function also resamples the data to a temporal resolution
        of 5 minutes and a vertical resolution of 900m.

        Args:
            path: The path containing the Cloudnet data.
            date: A date specifying the day from which to load the Cloudnet
                data.

        Return:
            An 'xarray.Dataset' containing the resampled radar data.
        """
        path = Path(path)
        pydate = to_datetime(date)
        date_str = pydate.strftime("%Y%m%d")
        tmpl = f"**/maowacrM1.a1.{date_str}*.nc"
        radar_file = sorted(list(path.glob(tmpl)))[-1]

        with xr.open_dataset(radar_file) as radar_data:

            # Use only reflectivities received in copolarization.
            copol = radar_data.polarization == 0
            radar_data = radar_data[{"time": copol}]
            print("SELECTING COPOL")

            # Resample data
            time = radar_data.time.data.astype("datetime64[s]")
            time_bins = np.arange(
                time[0],
                time[-1] + np.timedelta64(1, "s"),
                np.timedelta64(30, "s")
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


arm_manacapuru = ARMRadar(-60.598100, -3.212970, 250.0)


class RetrievalInput(Fascod):
    """
    Class to load the retrieval input data implementing the artssat
    data provider interface.
    """
    def __init__(
            self,
            radar,
            radar_data_path,
            era5_data_path
    ):
        """
        Args:
            radar: A radar object representing the radar from which the input
                observations are derived.
            radar_data_path: The path containing the radar data.
            era5_data_path: The path containing the ERA5 data.
        """
        super().__init__("midlatitude", "summer")
        self.radar = radar

        if radar_data_path is None and era5_data_path is None:
            self._tempdir = TemporaryDirectory()
            radar_data_path = Path(self._tempdir.name) / "radar"
            era5_data_path = Path(self._tempdir.name) / "era5"
        else:
            self._tempdir = None

        self.radar_data_path = Path(radar_data_path)
        self.era5_data_path = Path(era5_data_path)
        self._data = None

    def _load_data(self, date):
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
                date,
            )
            pydate = to_datetime(date)
            tmpl = f"**/*era5*{pydate.strftime('%Y%m%d')}*.nc"
            era5_files = sorted(list(self.era5_data_path.glob(tmpl)))
            era5_data = []
            for filename in era5_files:
                data = xr.load_dataset(filename)[{"level": slice(None, None, -1)}]
                data.coords["altitude"] = (("level",), data.z[0].data / 9.81)
                data = data.swap_dims({"level": "altitude"})
                altitudes = np.arange(
                    self.radar.elevation,
                    radar_data.range_bins.data.max() + 201.0,
                    200.0
                )
                data["p"] = (("altitude",), np.log(data.level.data))
                data = data.interp(altitude=altitudes, kwargs={"fill_value": "extrapolate"})
                era5_data.append(data)

            era5_data = xr.concat(era5_data, dim="time")
            era5_data.p.data = np.exp(era5_data.p.data)

            data = data.interp(
                {
                    "latitude": radar_data.latitude,
                    "longitude": radar_data.longitude,
                    "time": radar_data.time
                },
                method="nearest",
                kwargs={"fill_value": "extrapolate"}
            )

            era5_data = era5_data.interp(
                time=radar_data.time,
                method="nearest",
                kwargs={"fill_value": "extrapolate"}
            )
            self._data = xr.merge([era5_data, radar_data])
            self.interpolate_altitude(self._data.altitude.data)

    def has_data(self, date):
        """
        Determines whether input data for a given day is available.

        Args:
            date: datetime64 object specifying the day for which
                to run the retrieval.

        Return:
            'True' if the data is available, 'False' otherwise.
        """
        pydate = to_datetime(date)
        tmpl = f"**/*era5*{pydate.strftime('%Y%m%d')}*.nc"
        era5_files = sorted(list(self.era5_data_path.glob(tmpl)))
        has_era5 = len(era5_files) > 12
        return has_era5 and self.radar.has_data(self.radar_data_path, date)

    def download_data(self, date):
        """
        Downloads input data for a given day.

        Args:
            date: datetime64 object specifying the day for which
                to download the input data.
        """
        pydate = to_datetime(date)
        tmpl = f"**/*era5*{pydate.strftime('%Y%m%d')}*.nc"
        era5_files = sorted(list(self.era5_data_path.glob(tmpl)))
        has_era5 = len(era5_files) > 12
        if not has_era5:
            self.radar.download_era5_data(date, self.era5_data_path)

        if not self.radar.has_data(self.radar_data_path, date):
            self.radar.download_radar_data(date, self.radar_data_path)

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

    def get_rain_dm_x0(self, date):
        """
        First guess for $D_m$ parameter of the distribution of liquid
        hydrometeors.
        """
        n0 = 10 ** self.get_rain_n0_xa(date)
        t = self.get_temperature(date)

        dbz = self.get_radar_reflectivity(date)
        z = self.get_altitude(date)
        bins = self.get_radar_range_bins(date)
        centers = 0.5 * (bins[1:] + bins[:-1])
        dbz_i = np.interp(z, centers, dbz, left=-40, right=-40)

        iwc = cloudnet_iwc(dbz_i, t)
        dm = (16 * iwc / (n0 * np.pi * 1000.0)) ** (1 / 4)
        dm[n0 < 10 ** 3] = 1e-8
        return dm

    def get_cloud_water(self, date):
        """
        Concentraion of liquid cloud in kg/m^3
        """
        self._load_data(date)
        p = self.get_pressure(date).data
        z = self.get_altitude(date).data

        dz = np.diff(z)
        dp_dz = np.diff(p) / dz
        m_air = - dp_dz / 9.81
        m_air_e = np.zeros(m_air.size + 1)
        m_air_e[1:-1] = 0.5 * (m_air[1:] + m_air[:-1])
        m_air_e[0] = m_air_e[1]
        m_air_e[-1] = m_air_e[-2]

        clwc = self._data.clwc.interp(
            time=date,
            method="nearest",
            kwargs={"fill_value": "extrapolate"}
        )
        m_lc = m_air_e * np.maximum(clwc.data, 0.0)
        return m_lc


    def get_y_radar(self, date):
        """
        Return radar input observations for given date.
        """
        return self.get_radar_reflectivity(date)

    def get_radar_range_bins(self, date):
        """
        Return range bins of radar as center points between radar measurements.
        """
        self._load_data(date)
        return self._data.range_bins.data

    def get_y_radar_nedt(self, date):
        """
        Return nedt for radar observations.
        """
        self._load_data(date)
        range_bins = self.get_radar_range_bins(date)
        return np.ones(range_bins.size - 1)

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
        return self._data.p.interp(
            time=date,
            method="nearest",
            kwargs={"fill_value": "extrapolate"}
        ).data * 1e2

    def get_altitude(self, date):
        """Get the altitude in the atmospheric column above the radar."""
        self._load_data(date)
        return self._data.altitude.data

    def get_surface_altitude(self, date):
        """Get the surface altitude."""
        return np.array([[self.radar.elevation]])

    def get_radar_sensor_position(self, date):
        """Position of the radar"""
        return np.array([[self.radar.elevation]])

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

        mmr = mixing_ratio_from_relative_humidity(press, temp, rel)
        return dry_air_molecular_weight / water_molecular_weight * mmr

    def get_iwc_data(self, date, timestep):
        """
        Extracts Cloudnet reference IWC retrievals.

        Args:
            date: datetime64 object specifying the day for which to extract the
                reference data.
            timestep: The temporal resolution of the data.

        Return:
            An 'xarray.Dataset' containing the Cloudnet retrieval results for the
            given day.
        """
        self._load_data(date)
        start = date.astype("datetime64[D]").astype("datetime64[s]")
        end = start + np.timedelta64(1, "D").astype("timedelta64[s]")
        times = np.arange(
            start,
            end,
            timestep
        )
        self._load_data(date)
        data = self._data[["iwc", "iwc_reliability"]]
        z = self.get_altitude(date)
        data = data.interp(
            time=times,
            range=z,
            method="nearest",
            kwargs={"fill_value": 0}
        ).rename({
            "range": "altitude"
        })
        return data
