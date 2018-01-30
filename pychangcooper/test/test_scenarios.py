import pytest
import numpy as np

from pychangcooper.scenarios.synchrotron_cooling import SynchrotronCooling


def test_synchrotron_cooling():

    sc = SynchrotronCooling(store_progress=True)

    sc.run(photon_energies=np.logspace(1,3,10))


    sc.plot_photons_and_electrons()
