"""
Tests for the ccic.models module.
"""
import os
from pathlib import Path

import pytest
import numpy as np

import torch
from torch.utils.data import DataLoader

from ccic.data.training_data import CCICDataset
from ccic.models import CCICModel, SCALAR_VARIABLES, PROFILE_VARIABLES

try:
    TEST_DATA = Path(os.environ.get("CCIC_TEST_DATA", None))
    HAS_TEST_DATA = True
except TypeError:
    HAS_TEST_DATA = False

NEEDS_TEST_DATA = pytest.mark.skipif(
    not HAS_TEST_DATA, reason="Needs 'CCIC_TEST_DATA'."
)

@NEEDS_TEST_DATA
def test_forward():
    """
    Propagate a small training batch through a CCIC retrieval NN and ensure that
    the output has the expected shape.
    """
    data = CCICDataset(TEST_DATA / "training_data")
    data_loader = DataLoader(data, batch_size=2)

    x, y = next(iter(data_loader))
    model = CCICModel(4, 64, 64, n_blocks=2)
    with torch.no_grad():
        y_pred = model(x)

    assert "tiwc" in y_pred
    assert y_pred["tiwc"].shape[1] == 64 // 4
    assert "tiwp" in y_pred
    assert y_pred["tiwp"].shape[1] == 64
    assert "tiwp_fpavg" in y_pred
    assert y_pred["tiwp_fpavg"].shape[1] == 64
    assert "cloud_mask" in y_pred
    assert y_pred["cloud_mask"].shape[1] == 1
    assert "cloud_class" in y_pred
    assert y_pred["cloud_class"].shape[1] == 9
    assert not "encodings" in y_pred

    with torch.no_grad():
        y_pred = model(x, return_encodings=True)
        assert "encodings" in y_pred
