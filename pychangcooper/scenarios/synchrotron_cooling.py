import numpy as np

from pychangcooper import ChangCooper
from pychangcooper.utils.progress_bar import progress_bar


class SynchrotronCooling(ChangCooper):
    def __init__(self,
                 B=10.,
                 index= -2.2,
                 gamma_injection=1E3,
                 gamma_cool=2E3,
                 gamma_max=1E5,
                 n_grid_points=300,
                 max_gamma=1E5,
                 initial_distribution = None,
                 store_progress=False):

        #     Define a factor that is dependent on the magnetic field B.
        #     The cooling time of an electron at energy gamma is
        #     DT = 6 * pi * me*c / (sigma_T B^2 gamma)
        #     DT = (1.29234E-9 B^2 gamma)^-1 seconds
        #     DT = (cool * gamma)^-1 seconds
        #     where B is in Gauss.

        bulk_gamma = 300.
        const_factor = 1.29234E-9

        sync_cool = 1. / (B * B * const_factor)

        ratio = gamma_max / gamma_cool

        self._steps = np.round(ratio)

        self._gamma_max = gamma_max
        self._gamma_cool = gamma_cool
        self._gamma_injection = gamma_injection

        self._index = index
        self._cool = 1.29234E-9 * B * B

        delta_t = sync_cool / (gamma_max)

        super(SynchrotronCooling,
              self).__init__(n_grid_points, max_gamma, delta_t,
                             initial_distribution, store_progress)

    def _define_terms(self):

        self._dispersion_term = np.zeros(self._n_grid_points)

        self._heating_term = self._cool * self._half_grid2

    def _source_function(self, energy):
        """
        power law injection
        """

        out = np.zeros(self._n_grid_points)

        idx = (self._gamma_injection <= self._grid) & (self._grid <=
                                                       self._gamma_max)

        out[idx] = np.power(self._grid[idx], self._index)
        return out

    def run(self):

        with progress_bar(int(self._steps), title='cooling electrons') as p:
            for i in range(int(self._steps)):

                self.solve_time_step()

                p.increase()

    def _clean(self):

        lower_bound = min(self._gamma_cool, self._gamma_injection)

        idx = self._grid <= lower_bound

        self._n_current[idx] = 0.


class SynchrotronCoolingWithEscape(SynchrotronCooling):
    def __init__(self,
                 B=10.,
                 index= -2.2,
                 gamma_injection=1E3,
                 gamma_cool=2E3,
                 gamma_max=1E5,
                 n_grid_points=300,
                 max_gamma=1E5,
                 t_esc=1.,
                 initial_distribution = None,
                 store_progress=False):



        self._t_esc = t_esc

        super(SynchrotronCoolingWithEscape, self).__init__(B,index,gamma_injection,gamma_cool,gamma_max,n_grid_points,max_gamma,initial_distribution,store_progress)

    def _escape_function(self,energy):
        print('here')

        return 1./self._t_esc
