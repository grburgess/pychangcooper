import matplotlib.pyplot as plt
import numpy as np

from pychangcooper.chang_cooper import ChangCooper, log_grid_generator



from pychangcooper.photons.synchrotron_emission import SynchrotronEmission, synchrotron_cooling_constant, synchrotron_cooling_time
from pychangcooper.photons.photon_emitter import PhotonEmitter
from pychangcooper.scenarios.generic_cooling_component import GenericCoolingComponent
from pychangcooper.scenarios.continuous_powerlaw_injection import ContinuousPowerlawInjection





class SynchrotronCoolingComponent(GenericCoolingComponent):

    def __init__(self, B):
        """
        A synchrotron cooling component that implements the proper cooling terms
        """

        C0 = synchrotron_cooling_constant(B)


        super(SynchrotronCoolingComponent, self).__init__(C0, 2.)



class SynchrotronCooling_ImpulsivePLInjection(SynchrotronCoolingComponent, PhotonEmitter, ChangCooper):
    def __init__(self,
                 B=10.,
                 index=-2.2,
                 gamma_injection=1E3,
                 gamma_cool=2E3,
                 gamma_max=1E5,
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


        delta_t = synchrotron_cooling_time(B, gamma_max)


        

        SynchrotronCoolingComponent.__init__(self, B)
                

        ChangCooper.__init__(self, n_grid_points, max_grid, delta_t,
                             initial_distribution, store_progress)


        emission_kernel = SynchrotronEmission(self._grid, self._B)


        PhotonEmitter.__init__(self, n_steps, emission_kernel)

#     def _clean(self):

# #        lower_bound = min(self._gamma_cool, self._gamma_injection)

#         lower_bound = self._gamma_cool
        
#         idx = self._grid <= lower_bound

#         self._n_current[idx] = 0.

        
        
class SynchrotronCooling_ContinuousPLInjection(SynchrotronCoolingComponent, ContinuousPowerlawInjection, PhotonEmitter, ChangCooper):
    def __init__(self,
                 B=10.,
                 index=-2.2,
                 gamma_injection=1E3,
                 gamma_cool=2E3,
                 gamma_max=1E5,
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


        delta_t = synchrotron_cooling_time(B, gamma_max)

        SynchrotronCoolingComponent.__init__(self, B)
        ContinuousPowerlawInjection.__init__(self, gamma_injection, gamma_max, index, N=1.)
        

        ChangCooper.__init__(self, n_grid_points, max_grid, delta_t,
                             None, store_progress)


        
        emission_kernel = SynchrotronEmission(self._grid, self._B)


        PhotonEmitter.__init__(self, n_steps, emission_kernel)




#     def _clean(self):

# #        lower_bound = min(self._gamma_cool, self._gamma_injection)

#         lower_bound = self._gamma_cool
        
#         idx = self._grid <= lower_bound

#         self._n_current[idx] = 0.


