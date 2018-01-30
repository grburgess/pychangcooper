import pytest
import numpy as np

from pychangcooper.scenarios.synchrotron_cooling import SynchrotronCooling
from pychangcooper.scenarios.generic_cooling_acceleration import GenericCoolingAcceleration

def test_generic_cool_accel():


    n_grid_points = 300

    init_distribution = np.zeros(n_grid_points)


    for i in range(30):

        init_distribution[i+1] = 1.


    generic_ca = GenericCoolingAcceleration(n_grid_points=n_grid_points,
                                        C0 = 1.,
                                        t_acc= 1E-4,
                                        cooling_index=2.,
                                        acceleration_index=2.,
                                        initial_distribution = init_distribution,
                                        store_progress = True
                                       )


    for i in range(10):

        generic_ca.solve_time_step()




    fig = generic_ca.plot_evolution(skip=2,
                                alpha=.9,
                                cmap='Set1',
                                show_initial=True)

def test_synchrotron_cooling():

    
    synch_cool = SynchrotronCooling(B=1E10,
                                    index=-3.5,
                                    gamma_injection=1E3,
                                    gamma_cool=500,
                                    gamma_max=1E5,
                                    store_progress=True)

    


    synch_cool.run(photon_energies=np.logspace(1,7,50))

    synch_cool.plot_photons_and_electrons(skip=20,alpha=.7,cmap='viridis');
