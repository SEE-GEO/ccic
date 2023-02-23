"""
ccic.validation.retrieval
=========================

Implements an interface to run mcrf radar-only retrievals.
"""
from contextlib import contextmanager
import io
import logging
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import xarray as xr

from artssat.jacobian import Log10, Identity
from artssat.sensor import ActiveSensor
from artssat.retrieval import a_priori
from mcrf.retrieval import CloudRetrieval
from mcrf.hydrometeors import Hydrometeor
from mcrf.psds import D14NDmLiquid, D14NDmIce
from mcrf.liras.common import n0_a_priori, rh_a_priori, ice_mask, rain_mask
from mcrf.faam_combined import ObservationError
from pansat.time import to_datetime


@contextmanager
def capture_stdout(stream):
    """
    Context manager to capture stdout.

    Based on the following blog post:
        https://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/

    Args:
        stream: A buffer to which to write the output.
    """
    stdout_fd = sys.stdout.fileno()

    def redirect_stdout(to_fd):
        """Redirect stdout to the given file descriptor."""
        sys.stdout.close()
        os.dup2(to_fd, stdout_fd)
        sys.stdout = io.TextIOWrapper(os.fdopen(stdout_fd, "wb"))

    stdout_fd_copy = os.dup(stdout_fd)
    try:
        tfile = tempfile.TemporaryFile(mode="w+b")
        redirect_stdout(tfile.fileno())
        yield
        redirect_stdout(stdout_fd_copy)
        tfile.flush()
        tfile.seek(0, io.SEEK_SET)
        stream.write(tfile.read().decode())
    finally:
        tfile.close()
        os.close(stdout_fd_copy)


def iwc(n0, dm):
    """
    Calculate ice-water content from D14 PSD moments.

    Args:
        n0: Array containing the :math:`N_0^*` values of the PSD.
        dm: Array containing the :math:`D_m` values of the PSD.

    Return:
        Array containing the corresponding ice water content in
        :math:`kg/m^3`.
    """
    return np.pi * 917.0 * dm**4 * n0 / 4**4


def rwc(n0, dm):
    """
    Calculate rain-water content from D14 PSD moments.

    Args:
        n0: Array containing the :math:`N_0^*` values of the PSD.
        dm: Array containing the :math:`D_m` values of the PSD.

    Return:
        Array containing the corresponding rain water content in
        :math:`kg/m^3`.
    """
    return np.pi * 1000.0 * dm**4 * n0 / 4**4


def dm_a_priori(t):
    """
    Functional relation for of the a priori mean of :math:`D_m`
    using the DARDAR :math:`N_0^*` a priori and a fixed water
    content of :math:`10^{-7}` kg m:math:`^{-3}`.

    Args:
        t: Array containing the temperature profile.

    Returns:
        A priori for :math:`D_m`
    """
    n0 = 10**n0_a_priori(t)
    iwc = 1e-7
    dm = (4.0**4 * iwc / (np.pi * 917.0) / n0)**0.25
    return dm


def get_hydrometeors(static_data, shape):
    """
    Get hydrometeors for retrieval.

    Args:
        static_data: Path of the static retrieval data.
        shape: The name of the ice particle shape to load.

    Return:
        A list containing the liquid and frozen hydrometeors for
        the retrieval.
    """
    ice_shape = static_data / f"{shape}.xml"
    ice_shape_meta = static_data / f"{shape}.meta.xml"

    ice_mask = a_priori.FreezingLevel(lower_inclusive=True, invert=False)
    ice_covariance = a_priori.Diagonal(1e-3**2, mask=ice_mask, mask_value=1e-12)
    ice_covariance = a_priori.SpatialCorrelation(ice_covariance, 1e3, mask=ice_mask)
    ice_dm_a_priori = a_priori.FunctionalAPriori(
        "ice_dm",
        "temperature",
        dm_a_priori,
        ice_covariance,
        mask=ice_mask,
        mask_value=1e-8,
    )

    ice_covariance = a_priori.Diagonal(1e-6 ** 2, mask=ice_mask, mask_value=1e-12)
    ice_covariance = a_priori.SpatialCorrelation(ice_covariance, 2e3, mask=ice_mask)
    ice_n0_a_priori = a_priori.FunctionalAPriori(
        "ice_n0",
        "temperature",
        n0_a_priori,
        ice_covariance,
        mask=ice_mask,
        mask_value=4,
    )
    ice = Hydrometeor(
        "ice",
        D14NDmIce(),
        [ice_n0_a_priori, ice_dm_a_priori],
        str(ice_shape),
        str(ice_shape_meta),
    )

    ice.transformations = [Log10(), Identity()]
    # Lower limits for N_0^* and m in transformed space.
    ice.limits_low = [2, 1e-9]

    rain_shape = static_data / "LiquidSphere.xml"
    rain_shape_meta = static_data / "LiquidSphere.meta.xml"
    rain_mask = a_priori.FreezingLevel(lower_inclusive=False, invert=True)
    rain_covariance = a_priori.Diagonal(500e-6**2, mask=rain_mask, mask_value=1e-12)
    rain_dm_a_priori = a_priori.FixedAPriori(
        "rain_dm", 1e-3, rain_covariance, mask=rain_mask, mask_value=1e-8
    )
    rain_covariance = a_priori.Diagonal(1e-6, mask=rain_mask, mask_value=1e-12)
    rain_n0_a_priori = a_priori.FixedAPriori(
        "rain_n0", 7, rain_covariance, mask=rain_mask, mask_value=2
    )
    rain = Hydrometeor(
        "rain",
        D14NDmLiquid(),
        [rain_n0_a_priori, rain_dm_a_priori],
        str(rain_shape),
        str(rain_shape_meta),
    )
    rain.transformations = [Log10(), Identity()]
    rain.limits_low = [2, 1e-9]

    return [rain, ice]


class GroundRadar(ActiveSensor):
    """
    Artssat sensor object for Cloudnet ground-based radars.
    """

    def __init__(self, frequency):
        """
        Args: The frequency of the sensors in GHz
        """
        super().__init__(
            name="radar", f_grid=frequency, range_bins=None, stokes_dimension=4
        )
        self.sensor_line_of_sight = np.array([0.0])
        self.instrument_pol = [6]
        self.instrument_pol_array = [[6]]
        self.extinction_scaling = 1.0
        self.y_min = -40.0

    @property
    def nedt(self):
        return 0.5 * np.ones(self.range_bins.size - 1)



# Supported Cloudnet radars.
RADARS = {
    "palaiseau": GroundRadar(95e9),
    "schneefernerhaus": GroundRadar(35e9),
    "punta-arenas": GroundRadar(35e9),
    "manacapuru": GroundRadar(95e9)
}
RADARS["manacapuru"].y_min = -25


class RadarRetrieval:
    """
    Simple interface to run artssat/mcrf retrievals.
    """
    def setup_retrieval(self, input_data, ice_shape):

        hydrometeors = get_hydrometeors(input_data.static_data_path, ice_shape)

        input_data.add(hydrometeors[0].a_priori[0])
        input_data.add(hydrometeors[0].a_priori[1])
        input_data.add(hydrometeors[1].a_priori[0])
        input_data.add(hydrometeors[1].a_priori[1])

        sensors = [input_data.radar]
        input_data.add(ObservationError(sensors))

        self.retrieval = CloudRetrieval(
            hydrometeors, sensors, input_data, data_path=input_data.static_data_path,
            include_cloud_water=True
        )

        self.retrieval.simulation.retrieval.callbacks = []
        retrieval_settings = self.retrieval.simulation.retrieval.settings
        retrieval_settings["max_iter"] = 10
        retrieval_settings["stop_dx"] = 1e-2
        retrieval_settings["method"] = "lm"
        retrieval_settings["lm_ga_settings"] = np.array([200.0, 3.0, 2.0, 10e3, 5.0, 5.0])
        self.retrieval.simulation.setup()

    def process(self, input_data, time_interval, ice_shape):
        """
        Process day of radar observations.

        Args:
            date: Numpy datetime64 object specifying the day to process.
            time_interval: Interval determining how many retrieval to run
                for the day.

        Return:
            An xarray Dataset containing the retrieval results.
        """
        start, end = input_data.get_start_and_end_time()
        times = np.arange(start, end, time_interval)

        results = {}
        iwcs = []
        iwcs_n0 = []
        iwcs_n0_xa = []
        iwcs_dm = []
        iwcs_dm_xa = []
        rwcs = []
        rwcs_n0 = []
        rwcs_n0_xa = []
        rwcs_dm = []
        rwcs_dm_xa = []
        lcwcs = []
        temps = []
        press = []
        h2o = []
        diagnostics = []

        ys = []
        y_fs = []

        self.setup_retrieval(input_data, ice_shape)
        simulation = self.retrieval.simulation


        for time in times:

            simulation.run(time)
            results_t = simulation.retrieval.get_results()

            if results_t["yf_radar"][0].size == 0:
                results_t["yf_radar"][0] = np.nan * np.zeros_like(
                    results_t["y_radar"][0]
                )

            dm = results_t["ice_dm"][0]
            n0 = results_t["ice_n0"][0]
            iwcs.append(iwc(n0, dm))
            iwcs_n0.append(n0)
            iwcs_dm.append(dm)
            iwcs_n0_xa.append(input_data.get_ice_n0_xa(time))
            iwcs_dm_xa.append(input_data.get_ice_dm_xa(time))

            dm = results_t["rain_dm"][0]
            n0 = results_t["rain_n0"][0]
            rwcs.append(rwc(n0, dm))
            rwcs_n0.append(n0)
            rwcs_dm.append(dm)
            rwcs_n0_xa.append(input_data.get_rain_n0_xa(time))
            rwcs_dm_xa.append(input_data.get_rain_dm_xa(time))

            lcwcs.append(input_data.get_cloud_water(time))
            press.append(input_data.get_pressure(time))
            temps.append(input_data.get_temperature(time))
            h2o.append(input_data.get_H2O(time))

            ys.append(results_t["y_radar"][0])
            y_fs.append(results_t["yf_radar"][0])

            diagnostics.append(results_t["diagnostics"][0])

        radar_bins = input_data.get_radar_range_bins(times[0])

        # Expand observation vector if required.
        n_bins = np.array([y.size for y in ys])
        ys_new = []
        y_fs_new = []
        if n_bins.min() != n_bins.max():
            for index in range(len(ys)):
                y_padded = np.nan * np.zeros(n_bins.max())
                y_padded[:n_bins[index]] = ys[index]
                ys_new.append(y_padded)
                y_padded = np.nan * np.zeros(n_bins.max())
                y_padded[:n_bins[index]] = y_fs[index]
                y_fs_new.append(y_padded)
            ys = ys_new
            y_fs = y_fs_new
            radar_bins = (np.arange(n_bins.max() + 1) *
                          np.abs(radar_bins[1] - radar_bins[0]))
        radar_bins = 0.5 * (radar_bins[1:] + radar_bins[:-1])

        results = xr.Dataset(
            {
                "time": (("time",), times),
                "altitude": (("altitude",), input_data.get_altitude(times[0])),
                "iwc": (("time", "altitude"), np.stack(iwcs)),
                "iwc_dm": (("time", "altitude"), np.stack(iwcs_dm)),
                "iwc_dm_xa": (("time", "altitude"), np.stack(iwcs_dm_xa)),
                "iwc_n0": (("time", "altitude"), np.stack(iwcs_n0)),
                "iwc_n0_xa": (("time", "altitude"), np.stack(iwcs_n0_xa)),
                "rwc": (("time", "altitude"), np.stack(rwcs)),
                "rwc_dm": (("time", "altitude"), np.stack(rwcs_dm)),
                "rwc_dm_xa": (("time", "altitude"), np.stack(rwcs_dm_xa)),
                "rwc_n0": (("time", "altitude"), np.stack(rwcs_n0)),
                "rwc_n0_xa": (("time", "altitude"), np.stack(rwcs_n0_xa)),
                "lcwc": (("time", "altitude"), np.stack(lcwcs)),
                "temperature": (("time", "altitude"), np.stack(temps)),
                "pressure": (("time", "altitude"), np.stack(press)),
                "h2o": (("time", "altitude"), np.stack(h2o)),
                "oem_diagnostics": (("time", "diagnostics"), np.stack(diagnostics)),
                "radar_bins": (("radar_bins",), radar_bins),
                "radar_reflectivity": (("time", "radar_bins"), np.stack(ys)),
                "radar_reflectivity_fitted": (("time", "radar_bins"), np.stack(y_fs)),
            }
        )
        results["time"].attrs = {
            "long_name": "Time UTC",
            "standard_name": "time",
            "axis": "T",
        }
        results["altitude"].attrs = {
            "units": "m",
            "long_name": "Height above sea level.",
            "standard_name": "height_above_mean_sea_level",
        }
        results["iwc"].attrs = {"units": "kg m-3", "long_name": "Ice water content"}
        results["iwc_dm"].attrs = {
            "units": "m",
            "long_name": (
                "Mass-weighted mean diameter of the PSD of ice " "hydrometeors."
            ),
        }
        results["iwc_n0"].attrs = {
            "units": "m-4",
            "long_name": ("N0* of the PSD of ice hydrometeors."),
        }
        results["rwc"].attrs = {"units": "kg m-3", "long_name": "Rain water content"}
        results["rwc_dm"].attrs = {
            "units": "m",
            "long_name": (
                "Mass-weighted mean diameter of the PSD of rain " "hydrometeors."
            ),
        }
        results["rwc_n0"].attrs = {
            "units": "m-4",
            "long_name": ("N0* of the PSD of rain hydrometeors."),
        }
        results["radar_reflectivity"].attrs = {
            "units": "dbZ",
            "long_name": ("Radar reflectivity input to the retrieval"),
        }
        results["radar_reflectivity_fitted"].attrs = {
            "units": "dbZ",
            "long_name": ("The fitted radar reflectivity"),
        }
        return results


def process_day(
    date,
    input_data,
    ice_shapes,
    output_data_path,
    time_step=np.timedelta64(10 * 60, "s"),
):
    """
    Run retrieval for a day of input data.

    Args:
        date: A numpy datetime64 object defining the day for which to run the retrieval.
        input_data: An input_data that provides the retrieval input data.
        static_data_path: Path containing the static retrieval data.
        ice_shapes: List containing the names of the particle shapes to run the retrieval with.
        output_data_path: The path to which to write the output data.
        time_step: Time step defining how many retrievals to run for the given day.

    """
    py_date = to_datetime(date)

    start_time, end_time = input_data.get_start_and_end_time()
    start_time_str = to_datetime(start_time).strftime("%Y%m%d%H%M")
    end_time_str = to_datetime(end_time).strftime("%Y%m%d%H%M")
    output_data_path = Path(output_data_path)
    output_filename = f"{input_data.radar.instrument_name}_{start_time_str}_{end_time_str}.nc"

    for ice_shape in ice_shapes:

        retrieval = RadarRetrieval()
        output = io.StringIO()
        with capture_stdout(output):
            results = retrieval.process(input_data, time_step, ice_shape)
            results.to_netcdf(
                output_data_path / output_filename, group=ice_shape, mode="a"
            )

    iwc_data = input_data.get_iwc_data(date, time_step)
    iwc_data.to_netcdf(output_data_path / output_filename, group="cloudnet", mode="a")
    try:
        iwc_data = input_data.get_iwc_data(date, time_step)
        iwc_data.to_netcdf(output_data_path / output_filename, group="cloudnet", mode="a")
    except KeyError:
        logger = logging.getLogger(__name__)
        logger.warning(
            "No IWC reference data for retrieval %s on %s",
            input_data.radar.instrument_name,
            date
        )
