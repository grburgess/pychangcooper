import numpy as np

from pychangcooper.chang_cooper import ChangCooper


class GenericCoolingComponent(object):

    def __init__(self, C0, cooling_index):

        self._cooling_index = cooling_index
        self._C0 = C0


    def _define_terms(self):

        self._dispersion_term = np.zeros(self._n_grid_points)
        self._heating_term = self._C0 * np.power(self._half_grid, self._cooling_index)
        


