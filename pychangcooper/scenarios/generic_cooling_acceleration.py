import numpy as np

from pychangcooper.chang_cooper import ChangCooper


class GenericCoolingAcceleration(object):
    def __init__(self,
                 C0=1.,
                 t_acc=1,
                 cooling_index=-2,
                 acceleration_index=2):
                 
        """
        Gerneric cooling and acceleration
        :param C0:
        :param t_acc:
        :param cooling_index:
        :param acceleration_index:
        """

        self._C0 = C0
        self._t_acc = t_acc
        self._cooling_index = cooling_index
        self._acceleration_index = acceleration_index

        



    def _define_terms(self):
        self._dispersion_term = 0.5 * np.power(
            self._half_grid, self._acceleration_index) / self._t_acc

        self._heating_term = self._C0 * np.power(
            self._half_grid,
            self._cooling_index) - 2 * self._dispersion_term / self._half_grid

        
    def _t_cool(self, gamma):
        return 1. / (self._C0 * gamma)


    def _get_min_timescale(self):
        
        RuntimeWarning("This must be subclassed")

        return self._t_acc




class CoolingAcceleration(GenericCoolingAcceleration, ChangCooper):
    def __init__(self,
                 n_grid_points=300,
                 C0=1.,
                 t_acc=1,
                 cooling_index=-2,
                 acceleration_index=2,
                 max_grid=1E7,
                 store_progress=False,
                 initial_distribution=None):
                 
        """
        Gerneric cooling and acceleration
        :param C0:
        :param t_acc:
        :param cooling_index:
        :param acceleration_index:
        """

        GenericCoolingAcceleration.__init__(self, C0, t_acc, cooling_index, acceleration_index)

        delta_t = self._get_min_timescale()

        ChangCooper.__init__(self, n_grid_points, max_grid, delta_t, initial_distribution, store_progress)



# class GenericCoolingAcceleration(ChangCooper):
#     def __init__(self,
#                  n_grid_points=300,
#                  C0=1.,
#                  t_acc=1,
#                  cooling_index=-2,
#                  acceleration_index=2,
#                  max_grid=1E7,
#                  initial_distribution=None,
#                  store_progress=False):
#         """
#         Gerneric cooling and acceleration
#         :param n_grid_points:
#         :param C0:
#         :param t_acc:
#         :param cooling_index:
#         :param acceleration_index:
#         :param max_grid:
#         """

#         self._C0 = C0
#         self._t_acc = t_acc
#         self._cooling_index = cooling_index
#         self._acceleration_index = acceleration_index

#         delta_t = self._t_acc

#         super(GenericCoolingAcceleration, self).__init__(
#             n_grid_points, max_grid, delta_t, initial_distribution,
#             store_progress)

#     def _define_terms(self):
#         self._dispersion_term = 0.5 * np.power(
#             self._half_grid, self._acceleration_index) / self._t_acc

#         self._heating_term = self._C0 * np.power(
#             self._half_grid,
#             self._cooling_index) - 2 * self._dispersion_term / self._half_grid

#     def _t_cool(self, gamma):
#         return 1. / (self._C0 * gamma)

#     def _clean(self):
#         idx_too_small = self._n_current < 1E-15

#         self._n_current[idx_too_small] = 0.
