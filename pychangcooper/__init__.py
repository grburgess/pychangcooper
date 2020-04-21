from pychangcooper.chang_cooper import ChangCooper
from pychangcooper.scenarios.synchrotron_cooling import (
    SynchrotronCooling_ContinuousPLInjection,
    SynchrotronCooling_ImpulsivePLInjection,
)
from pychangcooper.scenarios.synchrotron_cooling_acceleration import (
    SynchCoolAccel_ImpulsivePLInjection,
    SynchCoolAccel_ContinuousPLInjection,
)

from pychangcooper.scenarios.generic_cooling_acceleration import CoolingAcceleration

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
