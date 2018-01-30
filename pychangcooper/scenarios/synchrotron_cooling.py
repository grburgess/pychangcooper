import numpy as np
import matplotlib.pyplot as plt

from pychangcooper.chang_cooper import ChangCooper
from pychangcooper.synchrotron_emission import SynchrotronEmission
from pychangcooper.utils.progress_bar import progress_bar

from pychangcooper.io.fill_plot import fill_plot_static, fill_plot_animated

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
        self._B = B
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

    def run(self, photon_energies = None):

        with progress_bar(int(self._steps), title='cooling electrons') as p:
            for i in range(int(self._steps)):

                self.solve_time_step()

                p.increase()

        if photon_energies is not None:

            self._compute_synchrotron_spectrum(photon_energies)

            self._photon_energies = photon_energies

    def _clean(self):

        lower_bound = min(self._gamma_cool, self._gamma_injection)

        idx = self._grid <= lower_bound

        self._n_current[idx] = 0.

    def _compute_synchrotron_spectrum(self, photon_energies):
        """

        """

        synchrotron_emitter = SynchrotronEmission(B=self._B,
                                                  photon_energies=photon_energies,
                                                  gamma_grid = self._grid
        )


        self._all_spectra = []

        with progress_bar(int(self._steps), title='computing spectrum') as p:

            for electrons in self.history:

                self._all_spectra.append(synchrotron_emitter.compute_spectrum(electrons))

                p.increase()

        self._all_spectra = np.array(self._all_spectra)


        self._total_spectrum = self._all_spectra.sum(axis=0)
        
        

    @property
    def final_spectrum(self):

        return self._total_spectrum

    @property
    def photon_energies(self):

        return self._photon_energies
        
    def plot_emission(self, cmap='viridis', skip=1, alpha=0.5, ax=None, animate = False):


            
        cumulative_spectrum = (self._all_spectra.cumsum(axis=0))[::skip]

        if not animate:

            fig = fill_plot_static(self._photon_energies, self._photon_energies**2 * cumulative_spectrum, cmap, alpha, ax)

        else:

            fig = fill_plot_animated(self._photon_energies, self._photon_energies**2 * cumulative_spectrum, cmap, alpha, ax)

        if ax is None:

            ax = fig.get_axes()[0]
        
        
        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_xlabel('Energy')
        ax.set_ylabel(r'$\nu F_{\nu}$')

        return fig

    def plot_photons_and_electrons(self, cmap = 'viridis', skip=1, alpha=0.5):

        fig, (ax1, ax2) = plt.subplots(1,2)

        _ = self.plot_evolution(cmap=cmap, skip=skip, ax=ax1, alpha=alpha)
        _ = self.plot_emission(cmap = cmap, skip=skip, ax=ax2, alpha=alpha)
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")


        ax1.set_xlim(left=min(self._gamma_cool,self._gamma_injection)*.5)

        
        fig.tight_layout()
        fig.subplots_adjust(hspace=0,wspace=0)
        
        return fig

        
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


