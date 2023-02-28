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
from artssat.scattering.psd.f07 import F07
from artssat.scattering.psd import D14M, AB12
from mcrf.psds import D14NDmLiquid
from mcrf.retrieval import CloudRetrieval
from mcrf.hydrometeors import Hydrometeor
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


def get_hydrometeors(static_data, shape, ice_psd="d14"):
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

    if ice_psd == "d14":
        psd = D14M(-0.26, 1.75, 917.0)
    else:
        psd = F07(),
    psd.t_max = 274.0

    ice_mask = a_priori.FreezingLevel(lower_inclusive=True, invert=False)
    ice_covariance = a_priori.Diagonal(6, mask=ice_mask, mask_value=1e-12)
    ice_covariance = a_priori.SpatialCorrelation(ice_covariance, 200.0, mask=ice_mask)
    ice_a_priori = a_priori.FixedAPriori(
        "ice_mass_density",
        -9,
        mask=ice_mask,
        covariance=ice_covariance,
        mask_value=-12
    )
    ice = Hydrometeor(
        "ice",
        psd,
        [ice_a_priori],
        str(ice_shape),
        str(ice_shape_meta),
    )
    ice.transformations = [Log10()]
    ice.limits_low = [-12]

    rain_shape = static_data / "LiquidSphere.xml"
    rain_shape_meta = static_data / "LiquidSphere.meta.xml"

    rain_mask = a_priori.FreezingLevel(lower_inclusive=False, invert=True)
    rain_covariance = a_priori.Diagonal(6, mask=rain_mask, mask_value=1e-12)
    rain_a_priori = a_priori.FixedAPriori(
        "rain_mass_density",
        -9,
        rain_covariance,
        mask=rain_mask,
        mask_value=-12
    )
    psd = AB12()
    psd.t_min = 273.0
    rain = Hydrometeor(
        "rain",
        psd,
        [rain_a_priori],
        str(rain_shape),
        str(rain_shape_meta),
    )
    rain.transformations = [Log10()]
    rain.limits_low = [-12]

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

        hydrometeors = get_hydrometeors(input_data.static_data_path, ice_shape, ice_psd="d14")
        for hydrometeor in hydrometeors:
            for prior in hydrometeor.a_priori:
                input_data.add(prior)

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
        rwcs = []
        rwcs_n0 = []
        rwcs_n0_xa = []
        lcwcs = []
        temps = []
        press = []
        h2o = []
        diagnostics = []
        sensor_pos = []

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

            iwc = results_t["ice_mass_density"][0]
            iwcs.append(iwc)
            rwc = results_t["rain_mass_density"][0]
            rwcs.append(rwc)

            lcwcs.append(input_data.get_cloud_water(time))
            press.append(input_data.get_pressure(time))
            temps.append(input_data.get_temperature(time))
            h2o.append(input_data.get_H2O(time))

            ys.append(results_t["y_radar"][0])
            y_fs.append(results_t["yf_radar"][0])

            diagnostics.append(results_t["diagnostics"][0])
            sensor_pos.append(input_data.get_radar_sensor_position(time)[0, 0])

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
                "rwc": (("time", "altitude"), np.stack(rwcs)),
                "lcwc": (("time", "altitude"), np.stack(lcwcs)),
                "temperature": (("time", "altitude"), np.stack(temps)),
                "pressure": (("time", "altitude"), np.stack(press)),
                "h2o": (("time", "altitude"), np.stack(h2o)),
                "oem_diagnostics": (("time", "diagnostics"), np.stack(diagnostics)),
                "radar_bins": (("radar_bins",), radar_bins),
                "radar_reflectivity": (("time", "radar_bins"), np.stack(ys)),
                "radar_reflectivity_fitted": (("time", "radar_bins"), np.stack(y_fs)),
                "sensor_position": (("time",), np.stack(sensor_pos))
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
        results["rwc"].attrs = {"units": "kg m-3", "long_name": "Rain water content"}
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
        #with capture_stdout(output):
        results = retrieval.process(input_data, time_step, ice_shape)
        results.to_netcdf(
            output_data_path / output_filename, group=ice_shape, mode="a"
        )

    try:
        iwc_data = input_data.get_iwc_data(date, time_step)
        iwc_data.to_netcdf(output_data_path / output_filename, group="reference", mode="a")
    except KeyError:
        logger = logging.getLogger(__name__)
        logger.warning(
            "No IWC reference data for retrieval %s on %s",
            input_data.radar.instrument_name,
            date
        )
