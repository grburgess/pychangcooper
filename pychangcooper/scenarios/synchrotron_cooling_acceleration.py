import numpy as np

from pychangcooper.chang_cooper import ChangCooper, log_grid_generator
from pychangcooper.scenarios.generic_cooling_acceleration import (
    GenericCoolingAccelerationComponent,
)
from pychangcooper.scenarios.continuous_powerlaw_injection import (
    ContinuousPowerlawInjection,
)
from pychangcooper.photons.photon_emitter import PhotonEmitter
from pychangcooper.scenarios.continuous_powerlaw_injection import (
    ContinuousPowerlawInjection,
)
from pychangcooper.photons.synchrotron_emission import (
    SynchrotronEmission,
    synchrotron_cooling_constant,
    synchrotron_cooling_time,
)


class SynchrotronCoolingAccelerationComponent(GenericCoolingAccelerationComponent):
    def __init__(self, B, t_acc, acceleration_index):
        """
        A synchrotron coooling and acceleration component that must be co-inherited with ChangCooper
        to build a full solver
        
        :param B: the magnetic field strength in Gauss
        :param t_acc: the acceleration time scale
        :param acceleration_index: the acceleration index
        """

        # get the cooling constant
        C0 = synchrotron_cooling_constant(B)

        super(SynchrotronCoolingAccelerationComponent, self).__init__(
            C0, t_acc, cooling_index=2, acceleration_index=acceleration_index
        )


class SynchCoolAccel_ImpulsivePLInjection(
    SynchrotronCoolingAccelerationComponent, PhotonEmitter, ChangCooper
):
    def __init__(
        self,
        B=10.0,
        index=-2.2,
        gamma_injection=1e3,
        gamma_cool=2e3,
        gamma_max=1e5,
        t_acc_fraction=1.0,
        acceleration_index=2,
        n_grid_points=300,
        max_grid=1e7,
        store_progress=False,
    ):

        """
            A synchrotron cooling and acceleration solver and radiation producer with and impulsive
            injection of power law distributed electrons. The acceleration timescale is taken as a 
            fraction of the cooling timescale. The acceleration index expresses the turbulence distribution
            of the magnetic field: 1 for Bohm, 5/3 for Kolmogorov, 2 for Fermi.
            
            
            :param B: the magnetic field strength in Gaussa
            :param index: the injected electron spectral index
            :param gamma_injection: the electron injection energy
            :param gamma_cool: the cooling energy
            :param gamma_max: the maximum injected energy
            :param t_acc_fraction: the acceleration time relative to the cooling time
            :param acceleration_index: the acceleration index
            :param n_grid_points: number of grid points
            :param max_grid: the maximum grid energy
            :param store_progress: to store the progress
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

        # calculate the cooling time of the highest energy electron
        # and set the acceleration time to the fraction of that

        cooling_time = synchrotron_cooling_time(B, gamma_max)

        t_acc = cooling_time * t_acc_fraction

        # initialize the cooling and acceleration terms

        SynchrotronCoolingAccelerationComponent.__init__(
            self, B, t_acc, acceleration_index
        )

        # make the time step the small enough to account for either the cooling
        # or the acceleration

        delta_t = min(t_acc, cooling_time)

        # initialize the solver

        ChangCooper.__init__(
            self, n_grid_points, max_grid, delta_t, initial_distribution, store_progress
        )

        # create the synchrotron emission kernel

        emission_kernel = SynchrotronEmission(self._grid, self._B)

        # initialize the photon emission constructor

        PhotonEmitter.__init__(self, n_steps, emission_kernel)


class SynchCoolAccel_ContinuousPLInjection(
    SynchrotronCoolingAccelerationComponent,
    ContinuousPowerlawInjection,
    PhotonEmitter,
    ChangCooper,
):
    def __init__(
        self,
        B=10.0,
        index=-2.2,
        gamma_injection=1e3,
        gamma_cool=2e3,
        gamma_max=1e5,
        t_acc_fraction=1.0,
        acceleration_index=2,
        n_grid_points=300,
        max_grid=1e7,
        store_progress=False,
    ):
        """
            A synchrotron cooling and acceleration solver and radiation producer with continuous
            injection of power law distributed electrons. The acceleration timescale is taken as a 
            fraction of the cooling timescale. The acceleration index expresses the turbulence distribution
            of the magnetic field: 1 for Bohm, 5/3 for Kolmogorov, 2 for Fermi.


            :param B: the magnetic field strength in Gaussa
            :param index: the injected electron spectral index
            :param gamma_injection: the electron injection energy
            :param gamma_cool: the cooling energy
            :param gamma_max: the maximum injected energy
            :param t_acc_fraction: the acceleration time relative to the cooling time
            :param acceleration_index: the acceleration index
            :param n_grid_points: number of grid points
            :param max_grid: the maximum grid energy
            :param store_progress: to store the progress
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

        # calculate the cooling time of the highest energy electron
        # and set the acceleration time to the fraction of that

        cooling_time = synchrotron_cooling_time(B, gamma_max)

        t_acc = cooling_time * t_acc_fraction

        # initialize the cooling and acceleration terms

        SynchrotronCoolingAccelerationComponent.__init__(
            self, B, t_acc, acceleration_index
        )

        # create the injection
        ContinuousPowerlawInjection.__init__(
            self, gamma_injection, gamma_max, index, N=1.0
        )

        # make the time step the small enough to account for either the cooling
        # or the acceleration

        delta_t = min(t_acc, cooling_time)

        # initialize the solver

        ChangCooper.__init__(
            self, n_grid_points, max_grid, delta_t, None, store_progress
        )

        # create the synchrotron emission kernel

        emission_kernel = SynchrotronEmission(self._grid, self._B)

        # initialize the photon emission constructor

        PhotonEmitter.__init__(self, n_steps, emission_kernel)
