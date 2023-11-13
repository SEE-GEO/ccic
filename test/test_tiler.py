"""
Tests for the ccic.tiler module.
================================
"""
import numpy as np

from ccic.tiler import Tiler, calculate_padding


def test_tiler():
    """
    Ensure that tiling and reassembling a tensor conserves its content.
    """
    for _ in range(1000):
        height = np.random.randint(64, 1024)
        width = np.random.randint(64, 1024)
        x = np.arange(width).astype(np.float32)
        y = np.arange(height).astype(np.float32)
        xy = np.stack(np.meshgrid(x, y))

        tiler = Tiler(xy, tile_size=128, overlap=32)
        tiles = []
        for row_ind in range(tiler.M):
            row = []
            for col_ind in range(tiler.N):
                tile = tiler.get_tile(row_ind, col_ind)
                assert tile.shape[1:] == (min(height, 128), min(width, 128))
                row.append(tile)
            tiles.append(row)

        xy_tiled = tiler.assemble(tiles)
        assert np.all(np.isclose(xy, xy_tiled))


def test_tiler_wrap_columns():
    """
    Ensure that tiling and reassembling a tensor conserves its content.
    """
    for _ in range(1000):
        height = np.random.randint(64, 1024)
        width = np.random.randint(64, 1024)
        x = np.arange(width).astype(np.float32)
        y = np.arange(height).astype(np.float32)
        xy = np.stack(np.meshgrid(x, y))

        tiler = Tiler(xy, tile_size=128, overlap=32, wrap_columns=True)
        tiles = []
        for row_ind in range(tiler.M):
            row = []
            for col_ind in range(tiler.N):
                tile = tiler.get_tile(row_ind, col_ind)
                assert tile.shape[1:] == (min(height, 128), min(width, 128))
                row.append(tile)
            tiles.append(row)

        xy_tiled = tiler.assemble(tiles)
        assert np.all(np.isclose(xy, xy_tiled))


def test_calculate_padding():
    """
    Test calculation of padding.
    """
    x = np.arange(23).astype(np.float32)
    y = np.arange(23).astype(np.float32)
    xy = np.stack(np.meshgrid(x, y))

    padding = calculate_padding(xy, 32)
    assert padding[0] == 4
    assert padding[1] == 5
    assert padding[2] == 4
    assert padding[3] == 5
