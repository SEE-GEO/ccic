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
import scipy as sp
import xarray as xr

from artssat.atmosphere.absorption import (
    O2,
    N2,
    H2O,
    CloudWater,
    RelativeHumidity,
    VMR
)
from artssat.atmosphere import Atmosphere1D
from artssat.atmosphere.catalogs import Aer, Perrin
from artssat.atmosphere.surface import Tessem
from artssat.jacobian import Log10, Identity
from artssat.sensor import ActiveSensor
from artssat.retrieval import a_priori
from artssat.scattering.psd.f07 import F07
from artssat.scattering.psd import D14M, AB12
from artssat.scattering.solvers import Disort, RT4
from artssat.simulation import ArtsSimulation
from artssat.data_provider import DataProviderBase
from artssat.scattering import ScatteringSpecies
from artssat.jacobian import Log10, Atanh, Composition, Identity
from pansat.time import to_datetime


class ObservationError(DataProviderBase):
    """
    """
    def __init__(self,
                 sensors,
                 footprint_error=False,
                 forward_model_error=False):
        """
        Arguments:
            sensors(:code:`list`): List of :code:`parts.sensor.Sensor` objects
                containing the sensors that are used in the retrieval.

            footprint_error(:code:`Bool`): Include footprint error for :code:`lcpr`
                sensor.

            forward_model_error(:code:`Bool`): Include estimated model error for
                all sensors.

        """
        self.sensors = sensors

    def _get_nedt(self, sensor, i_p):
        try:
            f_name = "get_y_" + sensor.name + "_nedt"
            f = getattr(self.owner, f_name)
            nedt_dp = f(i_p)
        except:
            nedt_dp = 0.0
        return nedt_dp

    def get_observation_error_covariance(self, i_p):
        m = 0

        diag = []

        for s in self.sensors:

            nedt_dp = self._get_nedt(s, i_p)
            if isinstance(s, ActiveSensor):
                if not isinstance(nedt_dp, (np.ndarray, list)):
                    nedt_dp = [nedt_dp] * s.y_vector_length
                diag += [nedt_dp**2]

        for s in self.sensors:
            nedt_dp = self._get_nedt(s, i_p)

        diag = np.concatenate(diag).ravel()
        covmat = sp.sparse.diags(diag, format="coo")

        return covmat


class Hydrometeor(ScatteringSpecies):
    """
    Specialization of the artssat.scattering.ScatteringSpecies class that
    adds serveral attributes that are used to customize the behavior of
    the CloudRetrieval class.

    Attributes:

        a_priori: A priori data provider for this hydrometeor species.

        transformations: List containing two transformations to apply to the
            two moments of the hydrometeor.

        scattering_data: Path of the file containing the scattering data to
            use for this hydrometeor.

        scattering_data_meta: Path of the file containing the meta data for
            this hydrometeor.

    """
    def __init__(self,
                 name,
                 psd,
                 a_priori,
                 scattering_data,
                 scattering_meta_data):
        super().__init__(name, psd, scattering_data, scattering_meta_data)
        self.a_priori = a_priori
        self.transformations = [Log10(), Log10()]
        self.limits_low      = [1e-12, 2]
        self.radar_only             = True
        self.retrieve_first_moment = True
        self.retrieve_second_moment = True


class CloudRetrieval:
    """
    Class for performing cloud retrievals.

    Attributes:

        simulation(artssat.ArtsSimulation): artssat ArtsSimulation object that is used
            to perform retrieval caculations.

        h2o(artssat.atmosphere.AtmosphericQuantity): The AtmosphericQuantity instance
            that represent water vapor in the ARTS simulation.

        cw(artssat.atmosphere.AtmosphericQuantity): The AtmosphericQuantity instance
            that represent cloud liquid in the ARTS simulation.

        sensors(artssat.sensor.Sensor): The sensors used in the retrieval.

        data_provider: The data provider used to perform the retrieval.
    """
    def _setup_retrieval(self):
        """
        Setup the artssat simulation used to perform the retrieval.
        """

        for q in self.hydrometeors:
            for ind, mom in enumerate(q.moments):
                if hasattr(q, "limits_high"):
                    limit_high = q.limits_high[ind]
                else:
                    limit_high = np.inf

                self.simulation.retrieval.add(mom)
                mom.transformation = q.transformations[ind]
                mom.retrieval.limit_low = q.limits_low[ind]
                mom.retrieval.limit_high = limit_high

        h2o = self.simulation.atmosphere.absorbers[-1]
        h2o_a = [p for p in self.data_provider.subproviders \
                 if getattr(p, "name", "") == "H2O"]
        if len(h2o_a) > 0:
            h2o_a = h2o_a[0]
            self.simulation.retrieval.add(h2o)
            atanh = Atanh(0.0, 1.1)
            if h2o_a.transformation is not None:
                h2o.transformation = h2o_a.transformation
            h2o.retrieval.unit = RelativeHumidity()

            if hasattr(h2o_a, "limit_low"):
                h2o.retrieval.limit_low = h2o_a.limit_low
            if hasattr(h2o_a, "limit_high"):
                h2o.retrieval.limit_high = h2o_a.limit_high
            self.h2o = h2o
        else:
            self.h2o = None

        cw_a = [p for p in self.data_provider.subproviders \
                if getattr(p, "name", "") == "cloud_water"]
        if len(cw_a) > 0 and self.include_cloud_water:
            cw_a = cw_a[0]
            cw = self.simulation.atmosphere.absorbers[-2]
            self.simulation.retrieval.add(cw)
            pl = PiecewiseLinear(cw_a)
            cw.transformation = Composition(Log10(), pl)
            cw.retrieval.limit_high = -3
            self.cw = cw
        else:
            self.cw = None

        t_a = [p for p in self.data_provider.subproviders \
               if getattr(p, "name", "") == "temperature"]
        if len(t_a) > 0:
            t = self.simulation.atmosphere.temperature
            self.temperature = t
            self.simulation.retrieval.add(self.temperature)
        else:
            self.temperature = None

    def __init__(
            self,
            hydrometeors,
            sensors,
            data_provider,
            data_path=None,
            include_cloud_water=False
    ):

        cw_a = [p for p in data_provider.subproviders \
                if getattr(p, "name", "") == "cloud_water"]
        self.include_cloud_water = (len(cw_a) > 0) or include_cloud_water

        self.hydrometeors = hydrometeors
        absorbers = [
            O2(model="TRE05", from_catalog=False),
            N2(model="SelfContStandardType", from_catalog=False),
            H2O(model=["SelfContCKDMT320", "ForeignContCKDMT320"],
                from_catalog=True,
                lineshape="VP",
                normalization="VVH",
                cutoff=750e9)
        ]
        if self.include_cloud_water:
            absorbers.insert(2, CloudWater(model="ELL07", from_catalog=False))
        scatterers = hydrometeors
        surface = Tessem()

        if data_path is None:
            catalog = Aer("h2o_lines.xml.gz")
        else:
            catalog = Aer(Path(data_path) / "h2o_lines.xml.gz")

        atmosphere = Atmosphere1D(absorbers, scatterers, surface, catalog=catalog)
        self.simulation = ArtsSimulation(atmosphere,
                                         sensors=sensors,
                                         scattering_solver=Disort(nstreams=16))
        self.sensors = sensors

        self.data_provider = data_provider
        self.simulation.data_provider = self.data_provider

        self._setup_retrieval()

        self.radar_only = all(
            [isinstance(s, ActiveSensor) for s in self.sensors])

        def radar_only(rr):

            rr.settings["max_iter"] = 30
            rr.settings["stop_dx"] = 1e-4
            rr.settings["method"] = "lm"
            rr.settings["lm_ga_settings"] = np.array(
                [1000.0, 3.0, 2.0, 10e3, 1.0, 1.0])

            rr.sensors = [s for s in rr.sensors if isinstance(s, ActiveSensor)]
            rr.retrieval_quantities = [h.moments[0] for h in self.hydrometeors]
            rr.retrieval_quantities += [
                h.moments[1] for h in self.hydrometeors
            ]
            #rr.retrieval_quantities = [h.moments[1] for h in self.hydrometeors]

        def all_quantities(rr):

            rr.settings["max_iter"] = 30
            rr.settings["stop_dx"] = 1e-2
            rr.settings["method"] = "lm"
            rr.settings["lm_ga_settings"] = np.array(
                [20.0, 5.0, 2.0, 1e5, 0.1, 1.0])

            if all([isinstance(s, PassiveSensor) for s in rr.sensors]):
                rr.settings["lm_ga_settings"] = np.array(
                    [20.0, 3.0, 2.0, 1e5, 0.1, 1.0])
            #else:
            #    rr.settings["lm_ga_settings"] = np.array(
            #        [10.0, 3.0, 2.0, 1e5, 1.0, 1.0])
            rr.retrieval_quantities = [h.moments[0] for h in self.hydrometeors]
            rr.retrieval_quantities += [
                h.moments[1] for h in self.hydrometeors
            ]

            if not self.h2o is None:
                rr.retrieval_quantities += [self.h2o]
            if not self.cw is None:
                rr.retrieval_quantities += [self.cw]
            if not self.temperature is None:
                rr.retrieval_quantities += [self.temperature]

        if all([isinstance(s, ActiveSensor) for s in self.sensors]):
            self.simulation.retrieval.callbacks = [("Radar only", radar_only)]
        elif any([isinstance(s, ActiveSensor) for s in self.sensors]):
            self.simulation.retrieval.callbacks = [
                #("Radar only", radar_only),
                ("All quantities", all_quantities)
            ]
        else:
            self.simulation.retrieval.callbacks = [("All quantities",
                                                    all_quantities)]

    def setup(self, verbosity=1):
        """
        Run artssat setup of simulation instance. This function needs to be executed
        before the retrieval can be calculated.

        Arguments:

            verbosity: ARTS workspace verbosity. 0 for silent.
        """

        self.simulation.setup(verbosity=verbosity)

    def run(self, i):
        """
        Run retrieval with simulation argument i.

        Arguments:
            i: The simulation argument that is passed to the run method of the
               ArtsSimulation object.
        """
        self.index = i
        return self.simulation.run(i)


################################################################################
# Cloud simulation
################################################################################


class CloudSimulation:
    """
    Class for performing forward simulation on GEM model data.
    """
    def __init__(self,
                 hydrometeors,
                 sensors,
                 data_provider,
                 include_cloud_water=False):
        """
        Arguments:

            hydrometeors(list): List of the hydrometeors to use in the simulation.

            sensors(list): List of sensors for which to simulate observations.

            data_provider: Data provider object providing the simulation data.

            include_cloud_water(bool): Whether or not to include cloud water
        """
        self.include_cloud_water = include_cloud_water

        self.hydrometeors = hydrometeors
        absorbers = [
            O2(model="TRE05", from_catalog=False),
            N2(model="SelfContStandardType", from_catalog=False),
            H2O(model=["SelfContCKDMT320", "ForeignContCKDMT320"],
                lineshape="VP",
                normalization="VVH",
                cutoff=750e9)
        ]
        absorbers = [O2(), N2(), H2O()]
        if self.include_cloud_water:
            absorbers.insert(2, CloudWater(model="ELL07", from_catalog=False))
        scatterers = hydrometeors
        surface = Tessem()
        atmosphere = Atmosphere2D(absorbers, scatterers, surface)
        self.simulation = ArtsSimulation(atmosphere,
                                         sensors=sensors,
                                         scattering_solver=Disort(nstreams=16))
        self.sensors = sensors

        self.data_provider = data_provider
        self.simulation.data_provider = self.data_provider

    def setup(self, verbosity=1):
        """
        Run setup method of ArtsSimulation.
        """
        self.simulation.setup(verbosity=verbosity)

    def run(self, *args, **kwargs):
        return self.simulation.run(*args, **kwargs)


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


def get_hydrometeors(static_data, ice_psd, ice_shape):
    """
    Get hydrometeors for retrieval.

    Args:
        static_data: Path of the static retrieval data.
        ice_psd: The PSD to use for frozen hydrometeors.
        ice_shape: The name of the ice particle shape to use.

    Return:
        A list containing the liquid and frozen hydrometeors for
        the retrieval.
    """
    ice_shape_meta = static_data / f"{ice_shape}.meta.xml"
    ice_shape = static_data / f"{ice_shape}.xml"

    if ice_psd == "d14":
        psd = D14M(-0.26, 1.75, 917.0)
    elif ice_psd.lower() == "f07_tropical":
        psd = F07(regime="tropical")
    else:
        psd = F07(regime="midlatitudes")
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
    def setup_retrieval(self, input_data, ice_psd, ice_shape):
        hydrometeors = get_hydrometeors(
            input_data.static_data_path,
            ice_psd,
            ice_shape,
        )
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

    def process(self, input_data, time_interval,  ice_psd, ice_shape):
        """
        Process day of radar observations.

        Args:
            date: Numpy datetime64 object specifying the day to process.
            time_interval: Interval determining how many retrieval to run
                for the day.
            ice_psd: String representing the PSD to use for ice hydrometeors.
                Should be 'd14' for the Delanoe 2014 PSD or 'F07' for the
                Field 2007 PSD.
            ice_shape: The name of the ice habit to use to represent ice
                particles.

        Return:
            An xarray Dataset containing the retrieval results.
        """
        start, end = input_data.get_start_and_end_time()
        times = np.arange(start, end, time_interval)

        results = {}
        iwcs = []
        rwcs = []
        lcwcs = []
        temps = []
        press = []
        h2o = []
        diagnostics = []
        sensor_pos = []
        latitude = []
        longitude = []

        ys = []
        y_fs = []

        self.setup_retrieval(input_data, ice_psd, ice_shape)
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

            latitude.append(input_data.get_latitude(time))
            longitude.append(input_data.get_longitude(time))

        latitude = np.array(latitude).ravel()
        longitude = np.array(longitude).ravel()

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
                "sensor_position": (("time",), np.stack(sensor_pos)),
                "latitude": (("time",), latitude),
                "longitude": (("time",), longitude),
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
    ice_psd,
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
    start_time, end_time = input_data.get_start_and_end_time()
    start_time_str = to_datetime(start_time).strftime("%Y%m%d%H%M")
    end_time_str = to_datetime(end_time).strftime("%Y%m%d%H%M")
    output_data_path = Path(output_data_path)
    output_filename = f"{input_data.radar.instrument_name}_{start_time_str}_{end_time_str}.nc"

    for ice_shape in ice_shapes:

        retrieval = RadarRetrieval()
        output = io.StringIO()
        #with capture_stdout(output):
        results = retrieval.process(input_data, time_step, ice_psd, ice_shape)
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
