import numpy as np

from pychangcooper.photons.emission_kernel import EmissionKernel




try:
    from pygsl.testing.sf import synchrotron_1

    has_gsl = True

except(ImportError):

    has_gsl = False


class SynchrotronEmission(EmissionKernel):
    def __init__(self, gamma_grid, B):
        """
        :param photon_energies: the photon energies
        :param gamma_grid: the electron gamma grid
        :param B: the B-field in units of Gauss
        """

        self._B = B
        self._gamma_grid = gamma_grid
        self._n_grid_points = len(gamma_grid)

        
        super(SynchrotronEmission, self).__init__()

        
    def set_photon_energies(self, photon_energies):
        """
        Set the photon energies and build the synchrotron kernel
        """

        super(SynchrotronEmission, self).set_photon_energies(photon_energies)
        
        if has_gsl:

            self._build_synchrotron_kernel()
            
        else:

            # this is a dummy for testing

            self._synchrotron_kernel = np.zeros((self._n_photon_energies, self._n_grid_points))

            RuntimeWarning('There is no GSL, cannot compute')

        
            
    def _build_synchrotron_kernel(self):
        """
        pre build the synchrotron kernel for the integration
        """
        self._synchrotron_kernel = np.zeros((self._n_photon_energies, self._n_grid_points))
        Bcritical = 4.14E13  # Gauss

        ec = 1.5 * self._B / Bcritical

        for i, energy in enumerate(self._photon_energies):
            for j, gamma in enumerate(self._gamma_grid):
                arg = energy / (ec * gamma * gamma)

                self._synchrotron_kernel[i, j] = synchrotron_1(arg)

    def compute_spectrum(self, electron_distribution):
        """
        
        :param electron_distribution: the electron distribution to convolve
        :return: 
        """

        spectrum = np.zeros_like(self._photon_energies)

        # convolve the synchrotron kernel with the electron
        # distribution
        for i, energy in enumerate(self._photon_energies):
            spectrum[i] = (self._synchrotron_kernel[i, 1:] * electron_distribution[1:] * (
            self._gamma_grid[1:] - self._gamma_grid[:-1])).sum() / (2. * energy)

        return spectrum



def synchrotron_cooling_constant(B):
    """
    Compute the characteristic cooling constant for synchrotron cooling

    Define a factor that is dependent on the magnetic field B.
    The cooling time of an electron at energy gamma is
    DT = 6 * pi * me*c / (sigma_T B^2 gamma)
    DT = (1.29234E-9 B^2 gamma)^-1 seconds
    DT = (cool * gamma)^-1 seconds
    where B is in Gauss.

    """


    bulk_gamma = 300.

    const_factor = 1.29234E-9

    C0 = const_factor * B**2

    return C0


def synchrotron_cooling_time(B, gamma):
    """
    Compute the characteristic cooling time of an electron of energy gamma
    for a given magnetic field strength

    """
    # get the cooling constant
    C0 = synchrotron_cooling_constant(B)

    return 1. / (C0 * gamma)
    
