from pychangcooper import ChangCooper


class SynchrotronCooling(ChangCooper):
    def __init__(self,
                 B=10.,
                 gamma_injection=1E3,
                 gamma_cool=2E3,
                 gamma_max=1E5,
                 n_grid_points=300,
                 max_gamma=1E5,
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

        self._cool = 1.29234E-9 * B * B

        self._delta_t = sync_cool / (gamma_max)

        super(SynchrotronCooling, self).__init__(n_grid_points, max_gamma,
                                                 store_progress)

    def _define_terms(self):

        self._dispersion_term = np.zeros(self._n_grid_points)

        self._heating_term = -self._cool * self._half_grid2

    def _source_function(self, energy):
        """
        power law injection
        """

        out = np.zeros(self._n_grid_points)

        idx = (self._gamma_injection <= self._grid) & (self._grid <=
                                                       self._gamma_max)

        out[idx] = np.power(self._grid[idx], -2)
        return out

    def run(self):

        for i in xrange(int(self._steps)):

            self.forward_sweep()
            self.back_substitution()

    def _clean(self):

        lower_bound = min(self._gamma_cool, self._gamma_injection)

        idx = self._grid <= lower_bound

        self._n_current[idx] = 0.
