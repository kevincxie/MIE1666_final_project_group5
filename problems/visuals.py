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
                    connecting_steps=0,
                    wall_color : Union[Tuple[int, int, int], int]=127,
                    x_lim : Optional[Tuple[float]]=(-1., 1.),
                    y_lim : Optional[Tuple[float]]=(-1., 1.),
                    wall_width_pct : Optional[float]=10.,
                    wall_height_pct : Optional[float]=0.3):

    traj_length = n_walls + connecting_steps * (n_walls)
    wall_indices = jnp.arange(0,traj_length,connecting_steps+1)

    # wall_horizontal_spacing = (x_lim[1] - x_lim[0]) / (n_walls + 1)
    wall_horizontal_spacing = (x_lim[1] - x_lim[0]) / (traj_length + 1)
    wall_vertical_spacing = (y_lim[1] - y_lim[0]) / (n_holes + 1)
    # print("horiz", wall_horizontal_spacing, wall_vertical_spacing)

    phi, weight = psi

    wall_patches = [
            __paint_wall(x_lim[0] + (wall_idx+1)*wall_horizontal_spacing,
                         n_holes,
                         y_lim[0],
                         y_lim[1],
                         phi[i],
                         weight[i],
                         wall_width_pct * wall_horizontal_spacing,
                         wall_height_pct * wall_vertical_spacing)
            for i, wall_idx in enumerate(wall_indices)
    ]

    for patch in wall_patches:
        for rect in patch:
            ax.add_patch(rect)

    return fig, ax


def plot_solution(ax : plt.Axes, soln : DeviceArray, marker='o', line_style='-', **kwargs):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    scale = DEFAULT_SCENE_HEIGHT / (ymax - ymin)
    n_walls = soln.shape[-1]
    wall_horizontal_spacing = (xmax - xmin) / (n_walls)

    x_ = (np.arange(n_walls)+1) * wall_horizontal_spacing + xmin
#    print(x_)
    x = np.repeat(x_, soln.shape[0])
    y = soln.reshape(-1)

  #  y = np.concatenate([y, np.zeros((1,))])

    y *= 1. / scale
#    ax.scatter(x, y, marker=marker, **kwargs)

#    x = np.concatenate([-np.ones((1,)), x, np.ones((1,))])

#   f = interp.interp1d(x, y, 'linear')
#   x_sm = np.linspace(xmin + wall_horizontal_spacing, xmax, 100)
#   y_sm = f(x_sm)
    ax.plot(x, y, linestyle=line_style, **kwargs)

    return ax

