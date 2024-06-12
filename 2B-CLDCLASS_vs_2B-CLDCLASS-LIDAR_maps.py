from pathlib import Path
import pickle

from pansat.products.satellite.cloud_sat import l2b_cldclass, l2b_cldclass_lidar

import matplotlib.pyplot as plt
import numpy as np
import tqdm
import xarray as xr


lon_edges = np.arange(-180, 181)
lat_edges = np.arange(-90, 91)

ro_files = sorted(list(Path('/data/ro_vs_lr/radar/2010').glob('**/*hdf')))
lr_files = sorted(list(Path('/data/ro_vs_lr/lidarradar/2010').glob('**/*hdf')))

ro_granules = {f.name.split('_')[0]: f for f in ro_files}
lr_granules = {f.name.split('_')[0]: f for f in lr_files}

# Keep only common files:
keys = sorted(list(ro_granules.keys() & lr_granules.keys()))
ro_granules = {k: ro_granules[k] for k in keys}
lr_granules = {k: lr_granules[k] for k in keys}

def to_shape(a, shape):
    y_, x_ = shape
    y, x = a.shape
    y_pad = (y_-y)
    x_pad = (x_-x)
    return np.pad(a,((0, y_pad), 
                     (0, x_pad)),
                  mode = 'constant')

def layers_to_bins(ds: xr.Dataset, var: str, fill_value=0) -> xr.Dataset:
    a = xr.full_like(ds.height, fill_value, dtype=ds[var].dtype)
    a.attrs = {}

    start = xr.where(ds.cloud_layer_base == -99, np.nan, ds.cloud_layer_base) * 1e3
    end = xr.where(ds.cloud_layer_top == -99, np.nan, ds.cloud_layer_top) * 1e3

    quality = xr.where(ds.cloud_type_quality == -99, np.nan, 1)

    for l in ds.layers:
        condition = (start.sel(layers=l) <= ds.height) & (ds.height <= end.sel(layers=l))
        a = xr.where(condition, ds[var].sel(layers=l), a)
    
    a = xr.where(np.isnan(quality).all(dim='layers'), -1, a)

    
    return a


hits_ro = np.zeros((360, 180), dtype=int)
counts_ro = np.zeros((360, 180), dtype=int)

hits_lr = np.zeros((360, 180), dtype=int)
counts_lr = np.zeros((360, 180), dtype=int)

hits_lrq = np.zeros((360, 180), dtype=int)
counts_lrq = np.zeros((360, 180), dtype=int)

heights_rl = np.array([], dtype=bool)

acc_rolr = np.zeros((40000, 125), dtype=int)
counts_rolr = np.zeros((40000, 125), dtype=int)

acc_lrlrq = np.zeros((40000, 125), dtype=int)
counts_lrlrq = np.zeros((40000, 125), dtype=int)

for granule in tqdm.tqdm(keys, ncols=40):
    try:
        ds_ro = l2b_cldclass.open(ro_granules[granule])
        ro_height_mask = xr.where(ds_ro.height <= xr.where(ds_ro.surface_elevation == -9999, 0, ds_ro.surface_elevation), np.nan, 1)
        ro_mask_2D = ((ds_ro.cloud_class > 0) * (ds_ro.cloud_class_flag == 1)).astype(int) * ro_height_mask * xr.where(ds_ro.cloud_class < 0, np.nan, 1)

        ro_mask_1D_valid = np.isfinite(ro_mask_2D).any(dim='bins')
        ro_mask_1D = (ro_mask_2D == 1).any(dim='bins') & ro_mask_1D_valid

        ds_lr = l2b_cldclass_lidar.open(lr_granules[granule])
        ds_lr['cloud_class_bins'] = (("rays", "bins"), layers_to_bins(ds_lr, "cloud_class").data)
        ds_lr['cloud_type_quality_bins'] = (("rays", "bins"), layers_to_bins(ds_lr, "cloud_type_quality").data)
        lr_height_mask = xr.where(ds_lr.height <= xr.where(ds_lr.surface_elevation == -9999, 0, ds_lr.surface_elevation), np.nan, 1)
        lr_mask_2D = (ds_lr.cloud_class_bins > 0).astype(int) * lr_height_mask * xr.where(ds_lr.cloud_class_bins < 0, np.nan, 1)
        lr_mask_quality_2D = lr_mask_2D * np.isfinite(np.where((0 <= ds_lr.cloud_type_quality_bins.data) & (ds_lr.cloud_type_quality_bins.data <= 1), ds_lr.cloud_type_quality_bins.data, np.nan))

        lr_mask_1D_valid = np.isfinite(lr_mask_2D).any(dim='bins')
        lr_mask_1D = (lr_mask_2D == 1).any(dim='bins') & lr_mask_1D_valid
        lr_mask_quality_1D_valid = np.isfinite(lr_mask_quality_2D).any(dim='bins')
        lr_mask_quality_1D = (lr_mask_quality_2D == 1).any(dim='bins') & lr_mask_quality_1D_valid

        hits_ro += np.histogram2d(ro_mask_1D.longitude.data[ro_mask_1D_valid.data], ro_mask_1D.latitude.data[ro_mask_1D_valid.data], bins=[lon_edges, lat_edges], weights=ro_mask_1D.data[ro_mask_1D_valid.data])[0].astype(int)
        counts_ro += np.histogram2d(ro_mask_1D_valid.longitude.data, ro_mask_1D_valid.latitude.data, bins=[lon_edges, lat_edges], weights=ro_mask_1D_valid.data.astype(int))[0].astype(int)

        hits_lr += np.histogram2d(lr_mask_1D.longitude.data[lr_mask_1D_valid.data], lr_mask_1D.latitude.data[lr_mask_1D_valid.data], bins=[lon_edges, lat_edges], weights=lr_mask_1D.data[lr_mask_1D_valid.data])[0].astype(int)
        counts_lr += np.histogram2d(lr_mask_1D_valid.longitude.data, lr_mask_1D_valid.latitude.data, bins=[lon_edges, lat_edges], weights=lr_mask_1D_valid.data.astype(int))[0].astype(int)

        hits_lrq += np.histogram2d(lr_mask_quality_1D.longitude.data[lr_mask_quality_1D_valid.data], lr_mask_quality_1D.latitude.data[lr_mask_quality_1D_valid.data], bins=[lon_edges, lat_edges], weights=lr_mask_quality_1D.data[lr_mask_quality_1D_valid.data])[0].astype(int)
        counts_lrq += np.histogram2d(lr_mask_quality_1D_valid.longitude.data, lr_mask_quality_1D_valid.latitude.data, bins=[lon_edges, lat_edges], weights=lr_mask_quality_1D_valid.data.astype(int))[0].astype(int)

        rolr_valid = (np.isfinite(ro_mask_2D) & np.isfinite(lr_mask_2D)).astype(int)
        rolr = np.where(rolr_valid, np.sign(lr_mask_2D - ro_mask_2D), 0)
        acc_rolr += to_shape(rolr.astype(int), (40000, 125))
        counts_rolr += to_shape(rolr_valid.data, (40000, 125))

        lrqlr_valid = (np.isfinite(lr_mask_quality_2D) & np.isfinite(lr_mask_2D)).astype(int)
        lrqlr = np.where(lrqlr_valid, np.sign(lr_mask_2D - lr_mask_quality_2D), 0)
        acc_lrlrq += to_shape(lrqlr.astype(int), (40000, 125))
        counts_lrlrq += to_shape(lrqlr_valid.data, (40000, 125))

        heights_rl = np.append(heights_rl, (ds_ro.height.data == ds_lr.height.data).all())
    except:
        pass

arrays = {
    'hits_ro': hits_ro,
    'counts_ro': counts_ro,
    'hits_lr': hits_lr,
    'counts_lr': counts_lr,
    'hits_lrq': hits_lrq,
    'counts_lrq': counts_lrq,
    'acc_rolr': acc_rolr,
    'counts_rolr': counts_rolr,
    'acc_lrlrq': acc_lrlrq,
    'counts_lrlrq': counts_lrlrq,
    'heights_lr': heights_rl
}

with open('2B-CLDCLASS_vs_2B-CLDCLASS-LIDAR_maps.pickle', 'wb') as handle:
    pickle.dump(arrays, handle)
