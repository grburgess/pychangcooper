import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from pychangcooper.utils.cmap_intervals import cmap_intervals

def fill_plot_static(x, y_values, cmap='viridis', alpha=1., ax=None ):

    if ax is None:
        fig, ax = plt.subplots()

    else:

        fig = ax.get_figure()

    colors = cmap_intervals(len(y_values), cmap)

            
    zorder = len(y_values)

    for i, y in enumerate(y_values):

        plot_time_step(i, x, y_values, colors[i], zorder, alpha, ax)

        zorder -= 1

    return fig


def plot_time_step(iteration, x, y_values,color, zorder, alpha, ax):


    if iteration == 0:

        ax.plot(x, y_values[iteration], alpha=alpha, zorder=zorder,color=color)

    elif np.all(y_values[iteration-1] == 0.):

        ax.plot(x, y_values[iteration], alpha=alpha, zorder=zorder,color=color)

    else:

        ax.fill_between(x, y_values[iteration-1], y_values[iteration], alpha=alpha, zorder=zorder,color=color)
        

    
        
