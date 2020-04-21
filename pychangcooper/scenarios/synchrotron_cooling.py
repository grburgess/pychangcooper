import numpy as np

from pychangcooper.chang_cooper import ChangCooper, log_grid_generator
from pychangcooper.photons.photon_emitter import PhotonEmitter
from pychangcooper.photons.synchrotron_emission import (
    SynchrotronEmission,
    synchrotron_cooling_constant,
    synchrotron_cooling_time,
)
from pychangcooper.scenarios.continuous_powerlaw_injection import (
    ContinuousPowerlawInjection,
)
from pychangcooper.scenarios.generic_cooling_component import GenericCoolingComponent


class SynchrotronCoolingComponent(GenericCoolingComponent):
    def __init__(self, B):
        """
        A synchrotron cooling component that implements the proper cooling terms
        
        :param B: magnetic field strength in Gauss
        
        """

        C0 = synchrotron_cooling_constant(B)

        super(SynchrotronCoolingComponent, self).__init__(C0, 2.0)


class SynchrotronCooling_ImpulsivePLInjection(
    SynchrotronCoolingComponent, PhotonEmitter, ChangCooper
):
    def __init__(
        self,
        B=10.0,
        index=-2.2,
        gamma_injection=1e3,
        gamma_cool=2e3,
        gamma_max=1e5,
        n_grid_points=300,
        max_grid=1e7,
        store_progress=False,
    ):
        """
        Synchrotron cooling and radiation from an impulsive power law injection of particles.
        
        :param B: the magnetic field strength in Gaussa
        :param index: the injected electron spectral index
        :param gamma_injection: the electron injection energy
        :param gamma_cool: the cooling energy
        :param gamma_max: the maximum injected energy
        :param n_grid_points: the number of grid point
        :param max_grid: the maximum grid energy
        :param store_progress: to store progress
        """

        self._B = B

        bulk_gamma = 300.0

        # calculate the number of steps to cool the maximum
        # electron energy to gamma cool

        ratio = gamma_max / gamma_cool

        n_steps = np.round(ratio)

        # assign the properties to the calss

        self._gamma_max = gamma_max
        self._gamma_cool = gamma_cool
        self._gamma_injection = gamma_injection
        self._index = index

        # build the initial power law injection

        initial_distribution = np.zeros(n_grid_points)
        tmp_grid, _, _ = log_grid_generator(n_grid_points, max_grid)

        idx = (gamma_injection <= tmp_grid) & (tmp_grid <= gamma_max)

        # normalize the electrons so that N = number of electrons

        norm = (
            np.power(gamma_max, index + 1) - np.power(gamma_injection, index + 1)
        ) / (index + 1)

        initial_distribution[idx] = 1.0 * norm * np.power(tmp_grid[idx], index)

        # set the time step to be that of the cooling time of
        # the most energetic electrons

        delta_t = synchrotron_cooling_time(B, gamma_max)

        # initialize the cooling terms

        SynchrotronCoolingComponent.__init__(self, B)

        # build the solver

        ChangCooper.__init__(
            self, n_grid_points, max_grid, delta_t, initial_distribution, store_progress
        )

        # create the emission kernel for producing radiation

        emission_kernel = SynchrotronEmission(self._grid, self._B)

        # initialize the radiation production

        PhotonEmitter.__init__(self, n_steps, emission_kernel)


class SynchrotronCooling_ContinuousPLInjection(
    SynchrotronCoolingComponent, ContinuousPowerlawInjection, PhotonEmitter, ChangCooper
):
    def __init__(
        self,
        B=10.0,
        index=-2.2,
        gamma_injection=1e3,
        gamma_cool=2e3,
        gamma_max=1e5,
        n_grid_points=300,
        max_grid=1e7,
        store_progress=False,
    ):
        """
        Synchrotron cooling and radiation from continuous power law injection of particles.
        
        :param B: the magnetic field strength in Gaussa
        :param index: the injected electron spectral index
        :param gamma_injection: the electron injection energy
        :param gamma_cool: the cooling energy
        :param gamma_max: the maximum injected energy
        :param n_grid_points: the number of grid point
        :param max_grid: the maximum grid energy
        :param store_progress: to store progress
        """

        self._B = B

        bulk_gamma = 300.0

        # calculate the number of steps to cool the maximum
        # electron energy to gamma cool

        ratio = gamma_max / gamma_cool

        n_steps = np.round(ratio)

        # assign the properties to the calss

        self._gamma_max = gamma_max
        self._gamma_cool = gamma_cool
        self._gamma_injection = gamma_injection
        self._index = index

        # set the time step to be that of the cooling time of
        # the most energetic electrons

        delta_t = synchrotron_cooling_time(B, gamma_max)
        # initialize the cooling terms

        SynchrotronCoolingComponent.__init__(self, B)

        # create the injection
        ContinuousPowerlawInjection.__init__(
            self, gamma_injection, gamma_max, index, N=1.0
        )

        # build the solver

        ChangCooper.__init__(
            self, n_grid_points, max_grid, delta_t, None, store_progress
        )

        # build the synchrotron emission kernel

        emission_kernel = SynchrotronEmission(self._grid, self._B)

        # initialize the photon emission process

        PhotonEmitter.__init__(self, n_steps, emission_kernel)

    def _clean(self):

        if self._iterations <= 1:

            self._idx_max = np.argmax(self._n_current)
            self._sync_max = self._n_current[self._idx_max]

        # now clean anything below

        idx1 = self._grid < self._gamma_injection

        idx2 = self._n_current < self._sync_max

        idx3 = idx1 & idx2

        self._n_current[idx3] = 0.0

        # now handle cooling

        if self._gamma_cool < self._gamma_injection:

            idx = self._grid < self._gamma_cool

            self._n_current[idx] = 0.0

        else:

            idx = self._grid < self._gamma_injection

            self._n_current[idx] = 0.0

        # lower_bound = min(self._gamma_cool, self._gamma_injection)

        # lower_bound = self._gamma_cool


#        idx = self._grid <= lower_bound

# self._n_current[:idx] = 0.
