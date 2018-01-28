import numpy as np

try:
    from pygsl.testing.sf import synchrotron_1

    has_gsl = True
    
except(ImportError):

    has_gsl = False

class SynchrotronEmission(object):

    def __init__(self, photon_energies, gamma_grid, B):
        """
        :param photon_energies: the photon energies
        :param gamma_grid: the electron gamma grid
        :param B: the B-field in units of Gauss
        """


        self._B = B
        self._photon_energies = photon_energies
        self._gamma_grid = gamma_grid

        self._n_photon_energies = len(self._photon_energies)
        self._n_grid_points = len(self._gamma_grid)

        if has_gsl:
            self._build_synchrotron_kernel()
        else:

            RuntimeWarning('There is no GSL, cannot compute')
    
    def _build_synchrotron_kernel(self):
        """
        pre build the synchrotron kernel for the integration
        """
        self._synchrotron_kernel = np.zeros((self._n_photon_energies, self._n_grid_points))
        Bcritical = 4.14E13 # Gauss


        ec = 1.5 * self._B/Bcritical
        
        for i, energy in enumerate(self._photon_energies):
            for j, gamma in enumerate(self._gamma_grid):

                arg = energy  / (ec * gamma * gamma)
                                
                self._synchrotron_kernel[i,j] = synchrotron_1(arg)
                                
    def compute_spectrum(self, electron_distribution):
        """
        

        """

        spectrum = np.zeros_like(self._photon_energies)
        
        for i, energy in enumerate(self._photon_energies):
    
            spectrum[i] = (self._synchrotron_kernel[i, 1:] * electron_distribution[1:] * (self._gamma_grid[1:] - self._gamma_grid[:-1])).sum()/(2.* energy )

        return spectrum
