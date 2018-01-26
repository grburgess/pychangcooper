import numpy as np
from pychangcooper.chang_cooper import ChangCooper

class GenericCoolingAcceleration(ChangCooper):

    def __init__(self, n_grid_points = 300,
                 C0 = 1.,
                 t_acc=1,
                 cooling_index = -2,
                 acceleration_index = 2,
                 max_grid = 1E7,
                 initial_distribution=None
                 store_progress=False):


        self._C0 = C0
        self._t_acc = t_acc
        self._cooling_index = cooling_index
        self._acceleration_index = acceleration_index


        delta_t = min(self._t_cool(max_grid), self._t_acc)
        
        super(GenericCoolingAcceleration, self).__init__(n_grid_points,
                                                         max_grid,
                                                         delta_t,
                                                         initial_distribution,
                                                         store_progress=False)
    def _define_terms(self):


        self._dispersion_term = 0.5 * np.power(self._half_grid, self._acceleration_index) / self._t_acc
        
        self._heating_term = self._C0 * np.power(self._half_grid,self._cooling_index) - 2 * self._dispersion_term/self._half_grid
        
    def _t_cool(self, gamma):


        return 1. / (self._C0 * gamma)
