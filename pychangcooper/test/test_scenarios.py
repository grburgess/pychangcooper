import pytest
import numpy as np

from pychangcooper.scenarios.synchrotron_cooling import SynchrotronCooling


def test_synchrotron_cooling():

    sc = SynchrotronCooling()

    sc.run()
