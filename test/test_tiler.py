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
    # Choose tile size so that a tile of size 1 is required between
    # first and last tile. This will cause the antepenultimate tile and
    # last tile to overlap, which triggered a bug in a previous version
    # of the tiler.
    x = np.arange(128 * 2 - 32 + 1).astype(np.float32)
    y = np.arange(128 * 2 - 32 + 1).astype(np.float32)
    xy = np.stack(np.meshgrid(x, y))

    tiler = Tiler(xy, tile_size=128, overlap=32)
    tiles = []
    for row_ind in range(tiler.M):
        row = []
        for col_ind in range(tiler.N):
            tile = tiler.get_tile(row_ind, col_ind)
            assert tile.shape[1:] == (128, 128)
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
