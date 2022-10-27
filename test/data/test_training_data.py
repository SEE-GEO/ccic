"""
Tests for ccic.data.training_data
"""
import os
from pathlib import Path

import pytest
import numpy as np

from ccic.data.training_data import CCICDataset


TEST_DATA = Path(os.environ.get("CCIC_TEST_DATA", None))
NEEDS_TEST_DATA = pytest.mark.skipif(
    TEST_DATA is None, reason="Needs 'CCIC_TEST_DATA'."
)


def test_load_data():
    """
    Ensure that the dataset length matches the number of training
    scenes.
    """
    dataset = CCICDataset(TEST_DATA / "training_data")
    n_files = len(list((TEST_DATA / "training_data").glob("*.nc")))
    assert len(dataset) == n_files


def test_load_sample():
    """
    Test that loading a sample from the data works and that data
    doesn't contain NANs.
    """
    dataset = CCICDataset(TEST_DATA / "training_data")

    for i in range(len(dataset)):

        x, y = dataset[0]

        assert "iwp" in y
        assert "iwp_rand" in y
        assert "iwc" in y
        assert "cloud_class" in y
        assert "cloud_mask" in y

        assert np.all(np.isfinite(x.numpy()))
        for key in y:
            assert np.all(np.isfinite(y[key].numpy()))
