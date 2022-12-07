"""
ccic.codecs
===========

Defines a codex for encoding floating point values using logarithmic
binning.
"""

from numcodecs.abc import Codec
from numcodecs.compat import ensure_ndarray
import numpy as np


class LogBins(Codec):
    """
    Encodes floating points as 1-byte, unsigned integers using
    logarithmically-spaced bins.
    """
    codec_id = 'log_bins'

    def __init__(self, low, high):
        """
        Args:
            low: The lower limit of the encoding range
            high: The upper limit of the encoding range



        """
        super().__init__()
        self.low = low
        self.high = high
        bins = np.logspace(np.log10(low), np.log10(high), 254)
        self.bins = bins.astype(np.float32)


    def encode(self, buf):
        """
        Encode the given buffer.

        Args:
            buf: A numpy array containing the data to encoder.

        Return:
            The buffer encoded as unsigned 8-bit integer.
        """
        arr = ensure_ndarray(buf)

        if not arr.dtype == np.float32:
            raise ValueError(
                "The LogBins filter only accepts input data in "
                "single-precicision float-point format."
            )
        n_bins = self.bins.size

        nan_mask = np.isnan(arr)

        # Determine indices
        indices = np.digitize(arr, self.bins).astype(np.uint8)
        # Cutoff values exceeding range
        indices = np.minimum(indices, 254)
        indices[nan_mask] = 255
        return indices

    def decode(self, buf, out=None):
        """
        Decode the given buffer.

        Args:
            buf: A numpy array containing the data to decode.
            out: Optional, pre-allocated output array.

        Return:
            The buffer decoded to single-precision floating-point
            format.
        """

        arr = ensure_ndarray(buf)
        centroids = 0.5 * (self.bins[:-1] + self.bins[1:])

        if out is None:
            values = np.zeros_like(arr, dtype=np.float32)
        else:
            values = out

        values[:] = centroids[np.clip(arr, 1, 253) - 1]
        values[arr == 0] = 0.0
        values[arr == 254] = self.bins[-1]
        values[arr == 255] = np.nan

        return values


    def get_config(self):
        # override to handle encoding dtypes
        return dict(
            id=self.codec_id,
            low=self.low,
            high=self.high
        )


    def __repr__(self):
        return f"LogBins(low={self.low}, high={self.high})"
