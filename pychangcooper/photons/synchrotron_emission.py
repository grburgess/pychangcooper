import numpy as np

from pychangcooper.photons.emission_kernel import EmissionKernel
from pychangcooper.photons.synchrotron_kernal import synchrotron_kernel


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

        self._build_synchrotron_kernel()

    def _build_synchrotron_kernel(self):
        """
        pre build the synchrotron kernel for the integration
        """
        self._synchrotron_kernel = np.zeros(
            (self._n_photon_energies, self._n_grid_points)
        )
        Bcritical = 4.14e13  # Gauss

        ec = 1.5 * self._B / Bcritical

        for i, energy in enumerate(self._photon_energies):
            for j, gamma in enumerate(self._gamma_grid):
                arg = energy / (ec * gamma * gamma)

                self._synchrotron_kernel[i, j] = synchrotron_kernel(arg)

    def compute_spectrum(self, electron_distribution):
        """
        
        :param electron_distribution: the electron distribution to convolve
        :return: 
        """

        spectrum = np.zeros_like(self._photon_energies)

        # convolve the synchrotron kernel with the electron
        # distribution
        for i, energy in enumerate(self._photon_energies):
            spectrum[i] = (
                self._synchrotron_kernel[i, 1:]
                * electron_distribution[1:]
                * (self._gamma_grid[1:] - self._gamma_grid[:-1])
            ).sum() / (2.0 * energy)

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

    bulk_gamma = 300.0

    const_factor = 1.29234e-9

    C0 = const_factor * B ** 2

    return C0


def synchrotron_cooling_time(B, gamma):
    """
    Compute the characteristic cooling time of an electron of energy gamma
    for a given magnetic field strength

    """
    # get the cooling constant
    C0 = synchrotron_cooling_constant(B)

    return 1.0 / (C0 * gamma)
