import numpy as np


class ContinuousPowerlawInjection(object):
    def __init__(self, x_min, x_max, index, N=1):

        self._x_min = x_min
        self._x_max = x_max
        self._N = N
        self._index = index

        self._norm = (np.power(x_max, index + 1) - np.power(x_min, index + 1)) / (
            index + 1
        )

    def _source_function(self, x):
        """
        power law injection
        
        """

        out = np.zeros(self._n_grid_points)

        idx = (self._x_min <= self._grid) & (self._grid <= self._x_max)

        out[idx] = self._N * self._norm * np.power(self._grid[idx], self._index)
        return out
