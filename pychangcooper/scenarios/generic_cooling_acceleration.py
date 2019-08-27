import numpy as np

from pychangcooper.chang_cooper import ChangCooper


class GenericCoolingAccelerationComponent(object):
    def __init__(self, C0=1.0, t_acc=1, cooling_index=-2, acceleration_index=2):

        """
        Generic cooling and acceleration component that must be co-inherited with 
        ChangCooper to form a complete solution. This is to reduce code duplication
        for similar problems
        
        
        :param C0: the cooling constant
        :param t_acc: the acceleration time
        :param cooling_index: the cooling index
        :param acceleration_index: the acceleeration index
        """

        self._C0 = C0
        self._t_acc = t_acc
        self._cooling_index = cooling_index
        self._acceleration_index = acceleration_index

    def _define_terms(self):
        self._dispersion_term = (
            0.5 * np.power(self._half_grid, self._acceleration_index) / self._t_acc
        )

        self._heating_term = (
            self._C0 * np.power(self._half_grid, self._cooling_index)
            - 2 * self._dispersion_term / self._half_grid
        )

    def _t_cool(self, gamma):
        """
        calculate the cooling time of a particle with energy gamma
        :param gamma: the energy of the particle
        :return: 
        """
        return 1.0 / (self._C0 * gamma)

    def _get_min_timescale(self):

        RuntimeWarning("This must be subclassed")

        return self._t_acc

    @property
    def equilbrium_energy(self):
        """
        The equilibrium energy of the electrons
        """

        return 1.0 / (self._C0 * self._t_acc)


class CoolingAcceleration(GenericCoolingAccelerationComponent, ChangCooper):
    def __init__(
        self,
        n_grid_points=300,
        C0=1.0,
        t_acc=1,
        cooling_index=-2,
        acceleration_index=2,
        max_grid=1e7,
        store_progress=False,
        initial_distribution=None,
    ):
        """
        Cooling and acceleration of an unspecified form. 
        
        
        :param n_grid_points: number of grid points 
        :param C0: the cooling constant
        :param t_acc: the acceleration time
        :param cooling_index: the index of the cooling term
        :param acceleration_index: the index of the acceleration term
        :param max_grid: the maximum grid energy
        :param store_progress: to store the progress of the evolution
        :param initial_distribution: the initial distribution of the electrons
        """

        # first call the cool-accel component constructor to setup the terms and special
        # properties

        GenericCoolingAccelerationComponent.__init__(
            self, C0, t_acc, cooling_index, acceleration_index
        )

        # figure out the delta t
        delta_t = self._get_min_timescale()

        ChangCooper.__init__(
            self, n_grid_points, max_grid, delta_t, initial_distribution, store_progress
        )
