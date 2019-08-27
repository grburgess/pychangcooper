import numpy as np


class EmissionKernel(object):
    def __init__(self):

        self._photon_energies = None
        self._n_photon_energies = None

    def set_photon_energies(self, photon_energies):

        self._photon_energies = photon_energies
        self._n_photon_energies = len(self._photon_energies)

    @property
    def photon_energies(self):

        return self._photon_energies

    @property
    def n_photon_energies(self):

        return self._n_photon_energies

    def compute_spectrum(self, electron_distribution):

        RuntimeError("Must be implemented in subclass")
