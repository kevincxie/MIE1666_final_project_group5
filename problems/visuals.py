from typing import Union, Optional, Tuple, List

import jax.numpy as jnp
import numpy as np
import scipy.interpolate as interp

from jax.numpy import DeviceArray

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

DEFAULT_SCENE_HEIGHT = 6. # random number lol - Y

def __paint_wall(x_loc : float, n_holes : int, y_min : float,
                 y_max : float, wall_offsets : DeviceArray,
                 hole_weights : DeviceArray, wall_width : float,
                 hole_height : float):
    scale = y_max - y_min

    x = x_loc - wall_width * 0.5
    walls = np.arange(n_holes + 2) / (n_holes + 1)
    walls = scale * walls + y_min

    hole_heights = hole_weights * hole_height
    walls[1:-1] += wall_offsets * scale / DEFAULT_SCENE_HEIGHT  + hole_heights * 0.5
    walls[-1] += hole_height
    ys = walls

    hole_heights = jnp.concatenate([hole_heights,
                                    jnp.array([hole_height])])
    rects = [
        Rectangle((x, ys[i]), wall_width, (ys[i+1] - hole_heights[i] - ys[i]))
                                                 for i in range(ys.shape[0] - 1)
    ]

    return rects

def plot_background(fig : plt.Figure, ax : plt.Axes,
                    psi : Tuple[DeviceArray, DeviceArray],
                    n_walls : int, n_holes : int,
                    wall_color : Union[Tuple[int, int, int], int]=127,
                    x_lim : Optional[Tuple[float]]=(-1., 1.),
                    y_lim : Optional[Tuple[float]]=(-1., 1.),
                    wall_width_pct : Optional[float]=0.1,
                    wall_height_pct : Optional[float]=0.3):

    wall_horizontal_spacing = (x_lim[1] - x_lim[0]) / (n_walls + 1)
    wall_vertical_spacing = (y_lim[1] - y_lim[0]) / (n_holes + 1)

    phi, weight = psi

    wall_patches = [
            __paint_wall(x_lim[0] + (i+1)*wall_horizontal_spacing,
                         n_holes,
                         y_lim[0],
                         y_lim[1],
                         phi[i],
                         weight[i],
                         wall_width_pct * wall_horizontal_spacing,
                         wall_height_pct * wall_vertical_spacing)
            for i in range(n_walls)
    ]

    for patch in wall_patches:
        for rect in patch:
            ax.add_patch(rect)

    return fig, ax


def plot_solution(ax : plt.Axes, soln : DeviceArray, marker='x', line_style='-', **kwargs):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    scale = DEFAULT_SCENE_HEIGHT / (ymax - ymin)
    n_walls = soln.shape[-1]
    wall_horizontal_spacing = (xmax - xmin) / (n_walls + 1)

    x_ = (np.arange(n_walls) + 1) * wall_horizontal_spacing + xmin
    x = np.repeat(x_, soln.shape[0])
    y = soln.reshape(-1)

    y *= 1. / scale
    ax.scatter(x, y, marker=marker, **kwargs)

    x = np.concatenate([-np.ones((1,)), x, np.ones((1,))])
    y = np.concatenate([np.zeros((1,)), y, np.zeros((1,))])
    
    f = interp.interp1d(x, y, 'linear')
    x_sm = np.linspace(-1, 1, 100)
    y_sm = f(x_sm)
    ax.plot(x_sm, y_sm, linestyle=line_style, **kwargs)
    
    return ax
