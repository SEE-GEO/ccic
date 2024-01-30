"""
ccic.plotting
=============

Helper functions for plotting results.
"""
from pathlib import Path
from typing import List

import cartopy.crs as ccrs
import cmocean
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import gridspec
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from matplotlib.cm import ScalarMappable, get_cmap
from matplotlib.colors import (
    Normalize,
    to_hex,
    LinearSegmentedColormap
)
from matplotlib.ticker import FixedLocator
import numpy as np
from pansat.time import to_datetime
import xarray as xr

from ccic.data.cloudsat import CLOUD_CLASSES


def set_style():
    """
    Set the CCIC matplotlib style.
    """
    plt.style.use(Path(__file__).parent / "files" / "ccic.mplstyle")


def scale_bar(
        ax,
        length,
        location=(0.5, 0.05),
        linewidth=3,
        height=0.01,
        border=0.05,
        border_color="k",
        parts=4,
        zorder=50,
        textcolor="k"
):
    """
    Draw a scale bar on a cartopy map.

    Args:
        ax: The matplotlib.Axes object to draw the axes on.
        length: The length of the scale bar in meters.
        location: A tuple ``(h, w)`` defining the fractional horizontal
            position ``h`` and vertical position ``h`` in the given axes
            object.
        linewidth: The width of the line.
    """
    import cartopy.crs as ccrs
    lon_min, lon_max, lat_min, lat_max = ax.get_extent(ccrs.PlateCarree())

    lon_c = lon_min + (lon_max - lon_min) * location[0]
    lat_c = lat_min + (lat_max - lat_min) * location[1]
    transverse_merc = ccrs.TransverseMercator(lon_c, lat_c)

    x_min, x_max, y_min, y_max = ax.get_extent(transverse_merc)

    x_c = x_min + (x_max - x_min) * location[0]
    y_c = y_min + (y_max - y_min) * location[1]

    x_left = x_c - length / 2
    x_right = x_c  + length / 2

    def to_axes_coords(point):
        crs = ax.projection
        p_data = crs.transform_point(*point, src_crs=transverse_merc)
        return ax.transAxes.inverted().transform(ax.transData.transform(p_data))

    def axes_to_lonlat(point):
        p_src = ax.transData.inverted().transform(ax.transAxes.transform(point))
        return ccrs.PlateCarree().transform_point(*p_src, src_crs=ax.projection)


    left_ax = to_axes_coords([x_left, y_c])
    right_ax = to_axes_coords([x_right, y_c])

    l_ax = right_ax[0] - left_ax[0]
    l_part = l_ax / parts



    left_bg = [
        left_ax[0] - border,
        left_ax[1] - height / 2 - border
    ]

    background = Rectangle(
        left_bg,
        l_ax + 2 * border,
        height + 2 * border,
        facecolor="none",
        transform=ax.transAxes,
        zorder=zorder
    )
    ax.add_patch(background)

    for i in range(parts):
        left = left_ax[0] + i * l_part
        bottom = left_ax[1] - height / 2

        color = "k" if i % 2 == 0 else "w"
        rect = Rectangle(
            (left, bottom),
            l_part,
            height,
            facecolor=color,
            edgecolor=border_color,
            transform=ax.transAxes,
            zorder=zorder
        )
        ax.add_patch(rect)

    x_bar = [x_c - length / 2, x_c + length / 2]
    x_text = 0.5 * (left_ax[0] + right_ax[0])
    y_text = left_ax[1] + 0.5 * height + 2 * border
    ax.text(x_text,
            y_text,
            f"{length / 1e3:g} km",
            transform=ax.transAxes,
            horizontalalignment='center',
            verticalalignment='top',
            color=textcolor
    )


def render_model(
        activations,
        outputs
):
    """
    Generates an SVG image of the CCIC neural network model.

    Args:
        activations: A list of tensors containing the intermediate
            activations from a forward pass through the CCIC model.
        outputs: A dictionary containing the predicted raw outputs.

    Return:
        A drawnn Image object representing a SVG drawing.
    """
    from drawnn.base import Image
    from drawnn.modules import Encoder, Decoder
    from drawnn.layers import InputLayer2D, Layer2D
    from drawnn import path

    # Mapppable for input.
    m_inpt = ScalarMappable(
        cmap="cmo.amp",
        norm=Normalize()
    )
    # Mappable for internal activations.
    m_int = ScalarMappable(
        cmap="Greys",
        norm=Normalize()
    )
    # Mappable for outputs.
    m_out = ScalarMappable(
        cmap="cmo.dense",
        norm=Normalize()
    )

    inputs = activations[0]
    inputs = [m_inpt.to_rgba((inpt - inpt.mean()) / inpt.std()) for inpt in inputs]

    base_ext = 150

    # Colors for modules
    cmap = get_cmap("cmo.dense")
    colors = cmap(np.linspace(0, 1, 11))[1:-1]

    with Image() as img:

        #
        # Input
        #

        cntr = [-20, 0, 0]
        ext = [10, base_ext, base_ext]
        input_layer = InputLayer2D(
            cntr, ext, inputs,
            edge_properties={"color": "black"},
            orientation="x",
        )
        input_layer.color_data = colors[0]
        input_layer.color_data[..., -1] = 0.5

        path.Path(
            input_layer.rcc,
            [50 - 12.5, 0, 0]
        )

        #
        # Stem
        #

        cntr = [50, 0, 0]
        ext = [25, base_ext, base_ext]
        stem = Layer2D(
            cntr,
            ext,
            (1, 1)
        )
        stem.color_data[:] = colors[0]

        path.Path(
            stem.rcc,
            [100 - 12.5, 0, 0]
        )

        cntr = [100, 0, 0]
        ext = [5, base_ext, base_ext]
        inputs = activations[1]
        inputs = [
            m_int.to_rgba((inpt - inpt.mean()) / inpt.std()) for inpt in inputs[:5]
        ]
        f_0 = InputLayer2D(
            cntr, ext, inputs,
            edge_properties={"color": "grey", "width": 1},
            fill_properties={"opacity": 0.3},
            orientation="x",
        )

        path.Path(
            f_0.rcc,
            [150, 0, 0]
        )

        #
        # Encoder
        #

        cntr = [150, 0, 0]
        ext = [50, base_ext, base_ext]
        encoder = Encoder(
            5,
            cntr,
            ext,
            shrink_factor=0.7,
            mappable=m_int,
            feature_maps=activations[2:],
            draw_feature_maps=True
        )
        encoder.translate(encoder.bb.rcc - cntr)
        for ind, stage in enumerate(encoder.stages):
            stage.fill_properties = {
                "color": to_hex(colors[ind + 1]),
                "opacity": 0.5
            }

        cntr = encoder.rcc + [50, 0, 0]
        ext = [50, base_ext, base_ext]
        decoder = Decoder(
            5,
            cntr,
            ext,
            shrink_factor=0.7,
            mappable=m_int,
            feature_maps=activations[7:],
            draw_feature_maps=True
        )
        decoder.translate(decoder.bb.rcc - cntr)
        for ind, stage in enumerate(decoder.stages):
            stage.fill_properties = {
                "color": to_hex(colors[5 - ind]),
                "opacity": 0.5
            }

        final = activations[-1]
        final = [m_int.to_rgba((fnl - fnl.mean()) / fnl.std()) for fnl in final[:5]]
        cntr = decoder.rcc + [50, 0, 0]
        ext = [3, base_ext, base_ext]
        final_act = InputLayer2D(
            cntr, ext, final,
            edge_properties={"color": "grey", "width": 1},
            fill_properties={"opacity": 0.3},
            orientation="x",
        )

        offset = 200

        #
        # Skip connections
        #

        for ind, (act, stage) in enumerate(
                zip([f_0] + encoder.feature_layers, decoder.stages[::-1])
        ):

            y_offset = -(5 - ind + 1) * 10
            path.Path(
                act.clc,
                act.clc + [0, y_offset, 0],
                stage.clc + [0, y_offset - act.clc[1] + stage.clc[1], 0],
                stage.clc
            )


        #
        # Outputs
        #

        outpt = path.Path(decoder.rcc, decoder.rcc + [50, 0, 0])

        keys = ["tiwp", "tiwc", "cloud_mask", "cloud_class"]
        n_outputs = len(keys)

        for ind, key in enumerate(keys):

            offs = -(n_outputs - 1) / 2 * offset + ind * offset
            path.Path(
                outpt.end,
                outpt.end + [25, 0, 0],
                outpt.end + [25, 0, offs],
                outpt.end + [50, 0, offs]
            )

        heads = {}

        for ind, key in enumerate(keys):

            offs = -(n_outputs - 1) / 2 * offset + ind * offset
            results = outputs[key]

            # Head module
            ext = [25, base_ext, base_ext]
            cntr = final_act.ccc + [50, 0, offs]
            heads[key] = Layer2D(
                cntr, ext, (1, 1)
            )
            heads[key].color_data[:] = colors[0]
            heads[key].color_data[..., -1] = 0.5


        for ind, key in enumerate(keys):

            offs = -(n_outputs - 1) / 2 * offset + ind * offset
            results = outputs[key]
            head = heads[key]

            path.Path(head.rcc, head.rcc + [50, 0, 0])

            # Output
            cntr = head.ccc + [50, 0]
            results = results[0]
            if results.ndim > 3:
                results = results.flatten(0, 1)
            results = [
                m_out.to_rgba((res - res.mean()) / res.std()) for res in results[:5]
            ]

            ext = [3, base_ext, base_ext]
            out_layer = InputLayer2D(
                cntr, ext, results,
                edge_properties={"color": "grey", "width": 1},
                fill_properties={"opacity": 1.0},
                orientation="x",
        )

    return img


def get_cloud_type_cmap():
    """
    Return cmap for displaying cloud types.
    """
    blues = get_cmap("Greens")
    b1, b2, b3, b4, b5 = (blues(x) for x in [0.3, 0.5, 0.7, 0.9, 1.0])
    reds = get_cmap("Reds")
    r1, r2, r3 = (reds(x) for x in [0.4, 0.7, 1.0])
    colors = ["#FFFFFF", b1, b2, b3, b4, b5, r1, r2, r3]
    cloud_class_cmap = LinearSegmentedColormap.from_list("cloud_classes", colors, N=9)
    return cloud_class_cmap


def add_ticks(ax, lons, lats, left=True, bottom=True):
    """
    Add ticks to a cartopy plot.

    Args:
        ax: A Matplotlib axes object to which to add the ticks.
        lons: The longitude values at which to draw the ticks.
        lats: The latitude values at which to draw ticks.
        left: Flag indicating whether to draw ticks on the left
            y-axis. 'lats' argument will be ignored if this is 'False'.
        bottom: Flag indicating whether to draw ticks on the lower
            x-axis. 'lons' argument will be ignored if this is 'False'.
    """
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0, color='none')
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = left
    gl.bottom_labels = bottom
    gl.xlocator = FixedLocator(lons)
    gl.ylocator = FixedLocator(lats)


def animate_tiwp(
        results: List[Path],
        output_path: Path,
        temporal_resolution: np.timedelta64 = np.timedelta64(15 * 60, "s")
) -> None:
    """
    Make animation of TIWP.

    Args:
        results: List of CCIC output files from which to create the animation.
        output_path: Filename to which to write the animation.
        temporal_resolution: The temporal resolution of each frame.

    Return:
        The figure object used to generate the animation.
    """
    gs = gridspec.GridSpec(2, 4, height_ratios=[1.0, 0.05])
    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(gs[0, :], projection=ccrs.Mollweide())
    norm = LogNorm(1e-2, 1e1)

    results = xr.open_mfdataset(results)["tiwp"][{
        "latitude": slice(0, None, 8),
        "longitude": slice(0, None, 8)
    }]
    start_time = results.time.min().data
    end_time = results.time.max().data

    cmap = get_cmap()
    cmap.set_bad("grey")
    m_inpt = ScalarMappable(
        cmap=cmap,
        norm=norm
    )
    cax = fig.add_subplot(gs[1, 1:3])
    plt.colorbar(m_inpt, label="TIWP [kg m$^{-2}$]", cax=cax, orientation="horizontal", extend="both")

    def draw_frame(time):
        ax.clear()
        date = to_datetime(time)
        time_str = date.strftime('%Y-%m-%d %H:%M:%S')
        print(f"time = {time_str}")

        results_t = results.interp(time=time, kwargs={"fill_value": "extrapolate"}).compute()
        lons = results_t.longitude.data
        lats = results_t.latitude.data
        tiwp = np.maximum(results_t.data, 1e-3)

        ax.coastlines(color="grey")

        m = ax.pcolormesh(lons, lats, tiwp, norm=norm, transform=ccrs.PlateCarree(), cmap=cmap)
        ax.set_title(f"{time_str}", loc="center")

        return m,

    times = np.arange(start_time, end_time, temporal_resolution)

    ani = animation.FuncAnimation(
        fig, draw_frame, times, interval=50, blit=True
    )
    ani.save(output_path, savefig_kwargs={"bbox_inches": "tight"})
    return fig
