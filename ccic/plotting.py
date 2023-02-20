"""
ccic.plotting
=============

Helper functions for plotting results.
"""
from pathlib import Path

import cmocean
import matplotlib.pyplot as plt


def set_style():
    """
    Set the CCIC matplotlib style.
    """
    plt.style.use(Path(__file__).parent / "files" / "ccic.mplstyle")
