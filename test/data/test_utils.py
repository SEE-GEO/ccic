import numpy as np

from ccic.data.utils import included_pixel_mask


def test_included_pixel_mask():
    """
    Ensure that included pixel mask

    """
    indices = (np.arange(101), np.arange(101))
    mask = included_pixel_mask(indices, 50, 50, 10)

    assert np.isclose(mask.sum(), 10)
