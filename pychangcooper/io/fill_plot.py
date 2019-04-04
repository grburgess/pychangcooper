import numpy as np
import matplotlib.pyplot as plt

from pychangcooper.utils.cmap_intervals import cmap_intervals


def fill_plot_static(x, y_values, cmap="viridis", alpha=1.0, ax=None):
    """
    
    plot all the y values 
    
    :param x: the x values 
    :param y_values: the matrix of y values
    :param cmap: mpl cmap
    :param alpha: the transparency 
    :param ax: optional ax
    :return: mpl figure
    """

    # create an axis and fig if not provided
    if ax is None:
        fig, ax = plt.subplots()

    else:

        fig = ax.get_figure()

    # create a list of colors

    colors = cmap_intervals(len(y_values), cmap)

    # keep figures up front

    zorder = len(y_values) + 10

    # loop through the y matrix and plot

    for i, y in enumerate(y_values):

        plot_time_step(i, x, y_values, colors[i], zorder, alpha, ax)

        # decrease the z order

        zorder -= 1

    return fig


def plot_time_step(iteration, x, y_values, color, zorder, alpha, ax):
    """
    plot an individual time step with a filled plot
    
    :param iteration: the iteration to consider
    :param x: the x values
    :param y_values: the matrix of y vaules
    :param color: the color to plot
    :param zorder: the zorder of the plot
    :param alpha: the transparency
    :param ax: the ax to plot with
    :return: None
    """

    # the first iteration is a simple line

    if iteration == 0:

        ax.plot(x, y_values[iteration], alpha=alpha, zorder=zorder, color=color)

    # don't fill to zero if there is nothing to plot

    elif np.all(y_values[iteration - 1] == 0.0):

        ax.plot(x, y_values[iteration], alpha=alpha, zorder=zorder, color=color)

    # fill between this iteration and the last
    else:

        ax.fill_between(
            x,
            y_values[iteration - 1],
            y_values[iteration],
            alpha=alpha,
            zorder=zorder,
            color=color,
        )
