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


NORMALIZER = MinMaxNormalizer(np.ones((3, 1, 1)), feature_axis=0)
NORMALIZER.stats = {
    0: (0.0, 1.0),
    1: (170, 310),
    2: (170, 310),
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
    high = np.log10(high)
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


def apply_transformations(x, y, rng):
    """
    Randomly applies transpose and flip transformations to input.

    Args:
        x: Tensor containing the input from a single training sample.
        y: Dict containing the corresponding training output.
        rng: Numpy RNG object to use for generation of random numbers.
    """
    # Transpose
    transpose = rng.random() > 0.5
    flip_v = rng.random() > 0.5
    flip_h = rng.random() > 0.5
    flip_dims = []
    if flip_v:
        flip_dims.append(-2)
    if flip_h:
        flip_dims.append(-1)

    if transpose:
        x = torch.transpose(x, -2, -1)
    if len(flip_dims) > 0:
        x = torch.flip(x, flip_dims)

    for key in y:
        y_k = y[key]
        if transpose:
            y_k = torch.transpose(y_k, -2, -1)
        if len(flip_dims) > 0:
            y_k = torch.flip(y_k, flip_dims)
        y[key] = y_k

    return x, y


class CCICDataset:
    """
    PyTorch dataset class to load the CCIC training data.
    """

    def __init__(self, path, input_size=None):
        """
        Args:
            path: Path to the folder containing the training samples.
            input_size: The size of the input observations.
        """
        self.path = Path(path)
        self.input_size = input_size
        self.files = np.array(list(self.path.glob("**/cloudsat_match*.nc")))
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

            x = np.nan * np.ones((3, height, width), dtype="float32")
            if "vis" in data:
                x[0] = data.vis.data
            if "ir_wv" in data:
                x[1] = data.ir_wv.data
            if "ir_win" in data:
                x[2] = data.ir_win.data

            #
            # Output
            #
            data["iwp"] = data["iwp"] * 1e-3
            data["iwp_rand"] = data["iwp_rand"] * 1e-3
            iwp = load_output_data(data, "iwp", 1e-6, 1e-3, self.rng)
            iwp_rand = load_output_data(data, "iwp_rand", 1e-6, 1e-3, self.rng)
            iwc = load_output_data(data, "iwc", 1e-6, 1e-3, self.rng)
            cloud_class = load_output_data(data, "cloud_class").astype(np.int64)
            cloud_mask = (cloud_class.max(0) > 0).astype(np.int64)

            x = torch.tensor(NORMALIZER(x))
            y = {}
            y["iwp"] = torch.tensor(iwp.copy())
            y["iwp_rand"] = torch.tensor(iwp_rand.copy())
            y["iwc"] = torch.tensor(iwc.copy())
            y["cloud_mask"] = torch.tensor(cloud_mask.copy())
            y["cloud_class"] = torch.tensor(cloud_class.copy())

            x, y = apply_transformations(x, y, self.rng)
            return x, y

    def __getitem__(self, index):
        filename = self.files[index]
        return self._load_sample(filename)

    def __len__(self):
        return len(self.files)
