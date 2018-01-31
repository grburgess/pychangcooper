import numpy as np


class ContinuousPowerlawInjection(object):

    def __init__(self, x_min, x_max, N=1):


        self._x_min = x_min
        self._x_max = x_max
        self._N = N


    def _source_function(self, x):
        """
        power law injection
        
        """

        out = np.zeros(self._n_grid_points)

        idx = (self._gamma_injection <= self._grid) & (self._grid <=
                                                       self._gamma_max)

        out[idx] = np.power(self._grid[idx], self._index)
        return out

