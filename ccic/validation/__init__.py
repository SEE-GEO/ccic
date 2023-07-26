from datetime import datetime, timedelta
from pathlib import Path

from netCDF4 import Dataset
import numpy as np
import xarray as xr
from scipy.stats import binned_statistic, binned_statistic_dd
from pansat.time import to_datetime64


def calculate_water_paths(dataset):
    """
    Add ice and rain water path to dataset.

    Args:
        dataset: A 'xarray.Dataset' containing CCIC radar-only
            retrieval results.
    """
    z = dataset.altitude.data

    iwc = dataset.iwc.data
    iwp = np.trapz(iwc, x=z)
    dims = dataset.iwc.dims[:-1]
    dataset["iwp"] = (dims, iwp)

    if "rwc" in dataset:
        rwc = dataset.rwc.data
        iwp = np.trapz(rwc, x=z)
        dims = dataset.rwc.dims[:-1]
        dataset["rwp"] = (dims, iwp)


def load_radar_results(files):
    """
    Load results from radar-only retrieval.

    Args:
        path: The path to the retrieval result file.

    Return:
        An 'xarray.Dataset' containing the retrieval results.
    """
    results = {}

    for input_file in files:
        with Dataset(input_file, "r") as input_data:
            groups = list(input_data.groups)

        for group in groups:
            data = xr.load_dataset(input_file, group=group)
            calculate_water_paths(data)
            results.setdefault(group, []).append(data)

    for group in results:
        results[group] = xr.concat(results[group], dim="time")
    return results


def load_ccic_results(files, longitude, latitude):
    """
    Load CCIC retrieval results for a single location.

    Args:
        files: A list of files from which to load the data.
        longitude: The longitude coordinate of the target location.
        latitude: The latitude coordinate of the target location.

    Return:
        An 'xarray.Dataset' containing the time series for the requested
        location.
    """
    files = sorted(files)
    results = []

    for input_file in files:
        with xr.open_zarr(input_file) as input_data:
            results.append(input_data.interp(
                latitude=latitude,
                longitude=longitude,
                method="nearest"
            ).compute())

    results = xr.concat(results, dim="time")
    return results


def calculate_daily_cycle(data, longitude=None, variable="iwp"):

    hours = data.time.dt.hour.data.astype(np.float64)
    iwp = data[variable].data
    valid = iwp >= 0.0

    hours = hours[valid]
    iwp = iwp[valid]

    if longitude is not None:
        delta = 12.0 / 180 * longitude
        hours += delta
    hours[hours > 23.5] -= 24.0
    hours[hours < -0.5] += 24.0

    bins = np.linspace(-0.5, 23.5, 25)
    mean_iwp = binned_statistic(hours, iwp, bins=bins)[0]

    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    return bin_centers, mean_iwp


def great_circle_distance(lats_1, lons_1, lats_2, lons_2):
    """
    Approximate distance between locations on earth.

    Uses haversine formulat with an earth radius of 6 371 km.

    Args:
        lats_1: Latitude coordinates of starting points.
        lons_1: Longitude coordinates of starting points.
        lats_2: Latitude coordinates of target points.
        lons_2: Longitude coordinates of target points.

    Return:
        The distance between the points described by the input
        arrays in m.
    """
    lats_1 = np.deg2rad(lats_1)
    lons_1 = np.deg2rad(lons_1)
    lats_2 = np.deg2rad(lats_2)
    lons_2 = np.deg2rad(lons_2)

    d_lons = lons_2 - lons_1
    d_lats = lats_2 - lats_1

    a = np.sin(d_lats / 2.0) ** 2 + np.cos(lats_1) * np.cos(lats_2) * np.sin(
        d_lons / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    R = 6371e3
    return R * c


def resample_data(
        data,
        variables,
        lon_bins,
        lat_bins,
        output_path,
        filename_pattern):
    """
    Resample field-campaign data to CCIC grid.

    Resamples a set of variables from an irregular swath to the CCIC grids.

    Args:
        data: xarray.Dataset containing the data to resample
        variables: List of names of the variables to resample.
        lon_bins: The longitude bins
        lat_bins: The latitude bins.
        output_path: The path to which to write the output files.
        filename_patther: Pattern to use for the filenames.

    """

    output_path = Path(output_path)

    time = data.time.data
    start_hour = time.min().astype("datetime64[h]")
    end_hour = time.max().astype("datetime64[h]") + np.timedelta64(1, "h")
    hours = np.arange(
        start_hour,
        end_hour + np.timedelta64(1, "h"),
        np.timedelta64(1, "h")
    )
    alt_bins = np.arange(21)

    for ind in range(hours.size - 1):

        start = hours[ind].astype("datetime64[m]") - np.timedelta64(15, "m")
        end = hours[ind + 1] - np.timedelta64(15, "m")


        t_inds = (time >= start) * (time < end)
        data_h = data[{"time": t_inds}]

        bins = np.arange(
            start,
            end + np.timedelta64(1, "m"),
            np.timedelta64(30, "m")
        ).astype(time.dtype)

        lons = data_h.longitude.data
        lats = data_h.latitude.data
        alt = data_h.altitude.data / 1e3

        time_h = time[t_inds]


        results = xr.Dataset({
            "latitude": (("latitude",), 0.5 * (lat_bins[1:] + lat_bins[:-1])),
            "longitude": (("longitude",), 0.5 * (lon_bins[1:] + lon_bins[:-1])),
            "time": (("time",), bins[:-1] + 0.5 * (bins[1] - bins[0])),
            "altitude": (("altitude",), 0.5 * (alt_bins[1:] + alt_bins[:-1])),
        })

        for var in variables:

            values = data_h[var].data
            time_v = time_h
            lons_v = lons
            lats_v = lats
            alt_v = alt

            if time_h.size < values.size:
                shape = values.shape
                time_v = np.broadcast_to(time_v[..., None], shape).ravel()
                lons_v = np.broadcast_to(lons_v[..., None], shape).ravel()
                lats_v = np.broadcast_to(lats_v[..., None], shape).ravel()
                alt_v = np.broadcast_to(alt_v[None], shape).ravel()

            values = values.ravel()
            valid = np.isfinite(values)
            if valid.sum() == 0:
                continue

            values_r = binned_statistic_dd(
                [time_v[valid].astype(np.float64), lats_v[valid], lons_v[valid], alt_v[valid]],
                values[valid],
                bins=[bins.astype(np.float64), lat_bins[::-1], lon_bins, alt_bins]
            )[0]
            values_r = np.flip(values_r, 1)
            results[var] = (
                ("time", "latitude", "longitude", "altitude"),
                values_r
            )


        year = results.time.dt.year[0].data
        month = results.time.dt.month[0].data
        day = results.time.dt.day[0].data
        hour = results.time.dt.hour[0].data

        filename = filename_pattern.format(**{
            "year": year,
            "month": month,
            "day": day,
            "hour": hour
        })
        output_filename = output_path / filename
        results.to_netcdf(output_filename)


def get_latlon_bins(ccic_file):
    """
    Extract latitude and longitude bins from a CCIC file.

    Args:
        ccic_file: String or path object pointing to a CCIC file
           to extract longitude and latitude bins from.

    Return:
        A tuple ``(lats, lons)`` containing the latitude and longitude
        bins.
    """
    with xr.open_dataset(ccic_file) as ccic_data:
        lats = ccic_data.latitude.data
        lons = ccic_data.longitude.data
        d_lat = np.diff(lats)[0]
        d_lon = np.diff(lons)[0]
        lat_c = 0.5 * (lats[1:] + lats[:-1])
        lon_c = 0.5 * (lons[1:] + lons[:-1])
        lats = np.concatenate([[lat_c[0] - d_lat], lat_c, [lat_c[-1] + d_lat]])
        lons = np.concatenate([[lon_c[0] - d_lon], lon_c, [lon_c[-1] + d_lon]])
        return lats, lons
