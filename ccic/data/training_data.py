"""
ccic.data.training_data
=======================

Provides the ``CCICDataset`` to load the CCIC training data.
"""
import os
from pathlib import Path

import numpy as np
from quantnn.normalizer import MinMaxNormalizer
import torch
import xarray as xr

import torchvision.transforms.functional as tf


NORMALIZER_ALL = MinMaxNormalizer(np.ones((3, 1, 1)), feature_axis=0)
NORMALIZER_ALL.stats = {
    0: (0.0, 1.0),
    1: (170, 310),
    2: (170, 310),
}
NORMALIZER = MinMaxNormalizer(np.ones((2, 1, 1)), feature_axis=0)
NORMALIZER.stats = {
    0: (170, 310),
}

MASK_VALUE = -100


def replace_zeros(data, low, high, rng):
    """
    In-place replacement of zero values.

    Replace zeros in a given array with random values from a fixed
    range.

    Args:
        data: The array in which to replace the zero values.
        low: The lower boundary of the range of random values.
        high: The upper boundary of the range of random values.
        rng: A numpy RNG instance to use to generate the random values.

    Return:
        A copy of 'data' with zeros replaced with random values.
    """
    data = data.copy()
    mask = (data >= 0.0) * (data < high)
    low = np.log10(low)
    high = np.log10(high) - 1.0
    n_valid = mask.sum()
    data[mask] = 10 ** rng.uniform(low, high, size=n_valid)
    return data


def expand_sparse(size, row_indices, col_indices, data):
    """
    Expands the sparse first dimension of an array.

    Args:
        size: The width and height of the dense data.
        row_indices: The row-indices of the elements in data.
        col_indices: The column-indices of the elements in data.
        data: Array whose first dimension to expand into two dense
            dimensions.

    Return:
        A dense array with one more dimension than data such that
        ``dense[row_indices, col_indices] == data``.
    """
    new_shape = (size, size) + data.shape[1:]
    data_dense = np.ones(new_shape, dtype=data.dtype) * MASK_VALUE
    data_dense[row_indices, col_indices] = data
    return data_dense


def load_output_data(dataset, name, low=None, high=None, rng=None):
    """
    Load output variable from dataset.

    Loads the input variable, transforms it from sparse to dense format
    and replaces zeros, if requested.

    Args:
       dataset: The ``xarray.Dataset`` containing the training sample.
       name: The name of the variable from which to load the data.
       low: If given, the lower boundary of the range of random values to use
           to replace zeros.
       low: If given, the upper boundary of the range of random values to use
           to replace zeros.
       rng: A numpy RNG object to use to generate random values.

    Return:
       A numpy array containing the loaded data.
    """
    # IWC
    size = dataset.latitude.size
    row_inds = dataset.profile_row_inds.data
    col_inds = dataset.profile_column_inds.data
    exp = expand_sparse(size, row_inds, col_inds, dataset[name].data)
    if low is not None:
        exp = replace_zeros(exp, low, high, rng)
    if exp.ndim == 3:
        exp = np.transpose(exp, [2, 0, 1])
    mask = np.isnan(exp)
    exp[mask] = MASK_VALUE
    return exp


def apply_transformations(x, y, rng, input_size=256):
    """
    Applies random transformation to the input data and extracts image samples
    of the given size.

    Args:
        x: Tensor containing the input from a single training sample.
        y: Dict containing the corresponding training output.
        rng: Numpy RNG object to use for generation of random numbers.
        input_size: The size of the input. If this is smaller than the input image
            a random crop will be extracted from the input.

    Return:
       A tuple ``(x, y)`` containing the network input ``x`` and corresponding retrieval
       output ``y``.
    """
    # Random rotation
    angle = rng.uniform(0, 360)
    flip_h = rng.random() > 0.5

    flip_dims = []
    if flip_h:
        flip_dims.append(-1)

    left = (x.shape[-2] // 2) - 128
    top = (x.shape[-2] // 2) - 128
    shift_h = int(rng.uniform(-7, 7))
    left += shift_h
    shift_v = int(rng.uniform(-7, 7))
    top += shift_v
    top = max(top, 0)
    left = max(left, 0)

    x = tf.rotate(x, angle)
    if len(flip_dims) > 0:
        x = torch.flip(x, flip_dims)
    x = tf.crop(x, top, left, input_size, input_size)

    for key in y:
        y_k = y[key]
        if (y_k.ndim < 3):
            y_k = tf.rotate(y_k[None], angle)[0]
        else:
            y_k = tf.rotate(y_k, angle)
        if len(flip_dims) > 0:
            y_k = torch.flip(y_k, flip_dims)
        y_k = tf.crop(y_k, top, left, input_size, input_size)
        y[key] = y_k

    return x, y


class CCICDataset:
    """
    PyTorch dataset class to load the CCIC training data.
    """
    def __init__(self, path, input_size=None, all_channels=False, inference=False):
        """
        Args:
            path: Path to the folder containing the training samples.
            input_size: The size of the input observations.
            all_channels: Use all input channels (includes visible)
            inference: to set to True when testing phase
        """
        self.path = Path(path)
        self.input_size = input_size
        self.all_channels = all_channels
        self.files = np.array(sorted(list(self.path.glob("**/cloudsat_match*.nc"))))
        self.inference = inference
        seed = int.from_bytes(os.urandom(4), "big") + os.getpid()
        self.rng = np.random.default_rng(seed)

    def seed(self, *args):
        """
        Seed the data loader's random generator.
        """
        seed = int.from_bytes(os.urandom(4), "big") + os.getpid()
        self.rng = np.random.default_rng(seed)

    def _load_sample(self, filename):
        """
        Load training data from file.

        Args:
            filename: The name of the file to load.

        Return:
            Tuple ``(x, y)`` of input ``x`` and input dict
            of output variables ``y``.
        """
        with xr.open_dataset(filename) as data:

            height = data.latitude.size
            width = data.longitude.size
            if self.input_size is None:
                i_start = 0
                i_end = height
                j_start = 0
                j_end = width
            else:
                d_i = max((height - self.input_size) // 2, 0)
                i_start = self.rng.integers(0, d_i)
                i_end = i_start + self.input_size

                d_j = max((width - self.input_size) // 2, 0)
                j_start = self.rng.integers(0, d_j)
                j_end = j_start + self.input_size

            data = data[
                {
                    "latitude": slice(i_start, i_end),
                    "longitude": slice(j_start, j_end),
                }
            ]

            height = i_end - i_start
            width = j_end - j_start

            #
            # Input
            #

            if self.all_channels:
                x = np.nan * np.ones((3, height, width), dtype="float32")
                if "vis" in data:
                    x[0] = data.vis.data
                if "ir_wv" in data:
                    x[1] = data.ir_wv.data
                if "ir_win" in data:
                    x[2] = data.ir_win.data
                x = torch.tensor(NORMALIZER_ALL(x))
            else:
                x = np.nan * np.ones((1, height, width), dtype="float32")
                if "ir_win" in data:
                    x[0] = data.ir_win.data
                if "cpcir2" in filename.stem:
                    x[0] /= 4
                x = torch.tensor(NORMALIZER(x))


            #
            # Output
            #
            data["tiwp_fpavg"] = data["tiwp_fpavg"] * 1e-3
            data["tiwp"] = data["tiwp"] * 1e-3
            tiwp_fpavg = load_output_data(data, "tiwp_fpavg", 1e-6, 1e-3, self.rng)
            tiwp = load_output_data(data, "tiwp", 1e-6, 1e-3, self.rng)
            tiwc = load_output_data(data, "tiwc", 1e-10, 1e-7, self.rng)
            cloud_class = load_output_data(data, "cloud_class").astype(np.int64)
            cloud_mask = load_output_data(data, "cloud_mask").astype(np.int64)

            y = {}
            y["tiwp"] = torch.tensor(tiwp_fpavg.copy())
            y["tiwp_fpavg"] = torch.tensor(tiwp.copy())
            y["tiwc"] = torch.tensor(tiwc.copy())
            y["cloud_mask"] = torch.tensor(cloud_mask.copy())
            y["cloud_class"] = torch.tensor(cloud_class.copy())

            # Include latitude and longitude in inference mode.
            if self.inference:
                latitude = data.latitude.data
                longitude = data.longitude.data
                longitude, latitude = np.meshgrid(longitude, latitude)
                y["latitude"] = torch.tensor(latitude.astype(np.float32))
                y["longitude"] = torch.tensor(longitude.astype(np.float32))
                granule = np.ones_like(latitude) * int(data.attrs["granule"])
                y["granule"] = torch.tensor(granule.astype(np.int64))
                y["tbs"] = torch.tensor(data.ir_win.data)

            input_size = self.input_size
            if input_size is None:
                input_size = 256

            if not self.inference:
                x, y = apply_transformations(x, y, self.rng, input_size=input_size)
            else:
                x = tf.center_crop(x, (input_size, input_size))
                y = {key: tf.center_crop(y[key], (input_size, input_size)) for key in y}

            # Ensure there's a minimum of valid training data in the sample.
            valid_output = (y["tiwp"] >= 0.0).sum()
            valid_input = (x >= -1.4).sum()

            if (valid_output < 50) or (valid_input < 50):
                if not self.inference:
                    new_index = self.rng.integers(0, len(self))
                    return self[new_index]
                else:
                    # Replace inputs and outputs with invalid values to flag it
                    y["tiwp"] = torch.full_like(y["tiwp"], MASK_VALUE)
                    x = torch.full_like(x, MASK_VALUE)

            return x, y

    def __getitem__(self, index):
        filename = self.files[index]
        return self._load_sample(filename)

    def __len__(self):
        return len(self.files)
