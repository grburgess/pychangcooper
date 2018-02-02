import numpy as np

from pychangcooper.chang_cooper import ChangCooper, log_grid_generator
from pychangcooper.scenarios.generic_cooling_acceleration import GenericCoolingAcceleration
from pychangcooper.scenarios.continuous_powerlaw_injection import ContinuousPowerlawInjection
from pychangcooper.photons.photon_emitter import PhotonEmitter
from pychangcooper.photons.synchrotron_emission import SynchrotronEmission, synchrotron_cooling_constant, synchrotron_cooling_time


class SynchrotronCoolingAccelerationComponent(GenericCoolingAcceleration):

    def __init__(self, B, t_acc, acceleration_index):

        C0 = synchrotron_cooling_constant(B)

        super(SynchrotronCoolingAccelerationComponent, self).__init__(C0, t_acc, cooling_index=2, acceleration_index=acceleration_index)


class SynchCoolAccel_ImpulsivePLInjection(SynchrotronCoolingAccelerationComponent, PhotonEmitter, ChangCooper):

        def __init__(self,
                     B=10.,
                     index=-2.2,
                     gamma_injection=1E3,
                     gamma_cool=2E3,
                     gamma_max=1E5,
                     t_acc_fraction = 1.,
                     acceleration_index=2,
                     n_grid_points=300,
                     max_grid=1E7,
                     store_progress=False):



            self._B = B

            bulk_gamma = 300.

            
            ratio = gamma_max / gamma_cool
            
            n_steps = np.round(ratio)
            
            self._gamma_max = gamma_max
            self._gamma_cool = gamma_cool
            self._gamma_injection = gamma_injection
            self._index = index
            
            
            initial_distribution = np.zeros(n_grid_points)
            tmp_grid, _, _ = log_grid_generator(n_grid_points,max_grid)
        
        
            idx = (gamma_injection <= tmp_grid) & (tmp_grid <= gamma_max)

            norm = (np.power(gamma_max, index+1) - np.power(gamma_injection, index+1)) /(index + 1 )
            
        
            initial_distribution[idx] = 1. * norm  *np.power(tmp_grid[idx], index)


            # calculate the cooling time of the highest energy electron
            # and set the acceleration time to the fraction of that

            cooling_time = synchrotron_cooling_time(B, gamma_max)

            t_acc = cooling_time * t_acc_fraction

            SynchrotronCoolingAccelerationComponent.__init__(self, B, t_acc, acceleration_index)

        
            delta_t = min(t_acc, cooling_time)

        
            ChangCooper.__init__(self, n_grid_points, max_grid, delta_t,
                                 initial_distribution, store_progress)

            emission_kernel = SynchrotronEmission(self._grid, self._B)

            PhotonEmitter.__init__(self, n_steps, emission_kernel)
