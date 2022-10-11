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



def replace_zeros(data, low, high, rng):
    """
    In-place replacement of zero values.

    Args

    """
    data = data.copy()
    mask = data < high
    low = np.log10(low)
    hi = np.log10(high)
    n = mask.sum()
    data[mask] = 10 ** rng.uniform(low, hi, size=n)
    return data


class CCICDataset:

    def __init__(self, path, output_size=None):
        self.path = Path(path)
        self.output_size = output_size
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
            Tuple ``(x, y)`` of input ``x`` and output dict
            of output variables ``y``.
        """
        with xr.open_dataset(filename) as data:

            m = data.latitude.size
            n = data.longitude.size
            if self.output_size is None:
                i_start = 0
                i_end = m
                j_start = 0
                j_end = n
            else:
                d_i = max((m - self.output_size) // 2, 0)
                i_start = self.rng.integers(0, d_i)
                i_end = i_start + self.output_size

                d_j = max((n - self.output_size) // 2, 0)
                j_start = self.rng.integers(0, d_j)
                j_end = j_start + self.output_size
            data = data[{
                "latitude": slice(i_start, i_end),
                "longitude": slice(j_start, j_end),
            }]

            m = i_end - i_start
            n = j_end - j_start

            x = np.nan * np.ones((3, m, n), dtype="float32")

            if "vis" in data:
                x[0] = data.vis.data
            if "ir_wv" in data:
                x[1] = data.ir_wv.data
            if "ir_window" in data:
                x[2] = data.ir_window.data

            iwp = replace_zeros(data.iwp.data / 1e3, 1e-6, 1e-3, self.rng)
            iwc = replace_zeros(data.iwc.data, 1e-6, 1e-3, self.rng)
            iwc = np.transpose(iwc, [2, 0, 1])

            mask = np.isnan(iwp)
            iwp[mask] = -100
            mask = np.isnan(iwc)
            iwc[mask] = -100

            if self.rng.random() > 0.5:
                x = np.transpose(x, [0, 2, 1])
                iwp = np.transpose(iwp, [1, 0])
                iwc = np.transpose(iwc, [0, 2, 1])

            if self.rng.random() > 0.5:
                x = np.flip(x, 1)
                iwp = np.flip(iwp, 0)
                iwc = np.flip(iwc, 1)

            if self.rng.random() > 0.5:
                x = np.flip(x, 2)
                iwp = np.flip(iwp, 1)
                iwc = np.flip(iwc, 2)

            x = torch.tensor(NORMALIZER(x, rng=self.rng))
            y = {}
            y["iwp"] = torch.tensor(iwp.copy())
            y["iwc"] = torch.tensor(iwc.copy())
            return x, y

    def __getitem__(self, index):
        filename = self.files[index]
        return self._load_sample(filename)

    def __len__(self):
        return len(self.files)
