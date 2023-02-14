from netCDF4 import Dataset
import numpy as np
import xarray as xr


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


