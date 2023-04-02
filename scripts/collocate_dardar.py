"""
This script collocates DARDAR data with the data prepared for the training and
test sets of CCIC.

Requirements and assumptions:
- All DARDAR granules used in the training are locally accessible,
  i.e. they do not need to be downloaded.
- Recording which DARDARDAR were missing takes place outside this script, e.g.,
  by analysing the stderr.
- No training or test scenes cross the antimeridian.

Notes:
- The variables stored have `_dardar` appended instead of `_cloudsat` to
  differentiate them from the resampled CloudSat DPC data (2C-ICE, 2B-CLDCLASS)
- The resampling done aims to mimic the resampling done for CloudSat DPC data
- The variables in the output files only have the resampled DARDAR data
- There is currently no support for collocating with the coarser CPCIR data
"""

import argparse
import glob
import logging
from pathlib import Path
import sys
import warnings

import numpy as np
from pyresample import create_area_def
import tqdm
import xarray as xr

from ccic.data import write_scenes
from ccic.data.cloudsat import resample_data
from ccic.data.dardar import DardarFile
from ccic.data.gridsat import GRIDSAT_GRID
from ccic.data.cpcir import CPCIR_GRID

def resample_to_scene(source_dataset_filepath, source_global_grid,
                      dardar_files_dict, dst) -> None:
    """
    Resample DARDAR data to a source dataset, where the source dataset is
    CloudSat-collocated CPCIR or GridSat data.

    Args:
        source_dataset_filepath: path to the source dataset
        source_global_grid: ``pyresample.geometry.AreaDefinition`` used to
            create the source dataset
        dardar_files_dict: a dictionary with CloudSat granule IDs as keys
            and DARDAR file paths as values
        dst: directory to write the resamped scenes

    Notes:
    - No scenes must be cross the antimeridian
    - The CloudSat granule used for the source dataset must be in
      ``dardar_files_dict``, else a KeyError is raised
    """
    assert source_global_grid in [GRIDSAT_GRID, CPCIR_GRID]

    # Open the source dataset
    source_dataset = xr.open_dataset(source_dataset_filepath)

    # Get the cloudsat times to slice the dardar data
    times_cloudsat = np.unique(source_dataset.time_cloudsat)

    # Clear all variables from the source dataset except `ir_win`
    # to avoid storing also the CloudSat DPC data
    source_dataset = source_dataset.drop_vars(set(list(source_dataset.keys())) - {'ir_win'})

    # Get the extent of the scene, taking into account the pixel resolution
    resolution = source_global_grid.resolution
    delta_lon = resolution[0] / 2
    delta_lat = resolution[1] / 2
    lon_scene = source_dataset.longitude.data
    lat_scene = source_dataset.latitude.data
    bbox_scene = [
        lon_scene.min() - delta_lon,
        lat_scene.min() - delta_lat,
        lon_scene.max() + delta_lon,
        lat_scene.max() + delta_lat
    ]

    # Create a pyresample.geometry.AreaDefinition for the scene
    # Silence the warning "shape found from radius and resolution
    # does not contain only integers" issued through the logging module
    level = logging.getLogger().getEffectiveLevel()
    logging.getLogger().setLevel(logging.CRITICAL)
    scene_grid = create_area_def(
        source_global_grid.area_id,
        source_global_grid.proj_dict,
        area_extent=bbox_scene,
        resolution=resolution,
        units="degrees"
    )
    logging.getLogger().setLevel(level)
    
    # This should hold, but assert as the warning above was silenced
    assert scene_grid.width == 384
    assert scene_grid.height == 384

    # Get the matching DARDAR file
    granule = int(source_dataset.attrs['granule'])
    dardar_file_path = dardar_files_dict[granule]
    dardar_file = DardarFile(dardar_file_path)

    # Resample using the time constraints
    ds_resampled = resample_data(
        source_dataset,
        scene_grid,
        [dardar_file],
        start_time=times_cloudsat.min(),
        end_time=times_cloudsat.max()
    )

    # Add an attribute to register the input source for the resampling
    ds_resampled.attrs['input_source_resampling'] = Path(source_dataset_filepath).name
    # and the DARDAR version
    ds_resampled.attrs['DARDAR_version'] = Path(dardar_file_path).stem.split('V')[-1]

    # Save to file
    write_scenes([ds_resampled], dst, product="dardar")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Resample and collocate DARDAR with existing collocations"
    )

    parser.add_argument(
        '--cpcir',
        type=Path,
        help="Root directory containing CloudSat collocated CPCIR files"
    )
    parser.add_argument(
        '--dardar',
        type=Path,
        help="Root directory where DARDAR files are located (possibily in subfolders)",
        required=True
    )
    parser.add_argument(
        '--gridsat',
        type=Path,
        help="Root directory containing CloudSat collocated GridSat files"
    )
    parser.add_argument(
        '--output',
        type=Path,
        help="Directory to save the output datasets",
        required=True
    )
    args = parser.parse_args()

    # Get DARDAR files
    dardar_files_dict = {DardarFile(f).granule: f for f in glob.glob(str(args.dardar / '**/*nc'), recursive=True)}

    # Process for CPCIR
    if args.cpcir:
        # Get CPCIR files
        cpcir_files = sorted(glob.glob(str(args.cpcir / '**/*nc'), recursive=True))

        # Make sure the destination directory exists
        dst_cpcir = args.output / 'cpcir'
        dst_cpcir.mkdir(parents=True, exist_ok=True)

        # Collocate with CPCIR files
        with warnings.catch_warnings():
            # Silence pyproj warning
            warnings.filterwarnings(
                "ignore",
                message="You will likely lose important projection information"
            )
            print("Processing CPCIR data")
            for f in tqdm.tqdm(cpcir_files, ncols=80, file=sys.stdout):
                try:
                    resample_to_scene(f, CPCIR_GRID, dardar_files_dict, dst_cpcir)
                except KeyError as e:
                    logging.error(f'DARDAR granule not found, {str(e)}')
    
    # Process for GridSat
    if args.gridsat:
        # Get GridSat
        gridsat_files = sorted(glob.glob(str(args.gridsat / '**/*nc'), recursive=True))

        # Make sure the destination directory exists
        dst_gridsat = args.output / 'gridsat'
        dst_gridsat.mkdir(parents=True, exist_ok=True)

        # Collocate with GridSat files
        with warnings.catch_warnings():
            # Silence pyproj warning
            warnings.filterwarnings(
                "ignore",
                message="You will likely lose important projection information"
            )
            print("Processing GridSat data")
            for f in tqdm.tqdm(gridsat_files, ncols=80, file=sys.stdout):
                try:
                    resample_to_scene(f, GRIDSAT_GRID, dardar_files_dict, dst_gridsat)
                except KeyError as e:
                    logging.error(f'DARDAR granule not found, {str(e)}')