import pytest
import numpy as np

from pychangcooper.scenarios.synchrotron_cooling import SynchrotronCooling_ContinuousPLInjection
from pychangcooper.scenarios.synchrotron_cooling import SynchrotronCooling_ImpulsivePLInjection
from pychangcooper.scenarios.synchrotron_cooling_acceleration import SynchCoolAccel_ImpulsivePLInjection, SynchCoolAccel_ContinuousPLInjection
from pychangcooper.scenarios.generic_cooling_acceleration import CoolingAcceleration

def test_generic_cool_accel():


    n_grid_points = 300

    init_distribution = np.zeros(n_grid_points)


    for i in range(30):

        init_distribution[i+1] = 1.


    generic_ca = CoolingAcceleration(n_grid_points=n_grid_points,
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

    
    synch_cool = SynchrotronCooling_ContinuousPLInjection(B=1E10,
                                    index=-3.5,
                                    gamma_injection=1E3,
                                    gamma_cool=500,
                                    gamma_max=1E5,
                                    store_progress=True)

    


    synch_cool.run(photon_energies=np.logspace(1,7,50))

    synch_cool.plot_photons_and_electrons(skip=20,alpha=.7,cmap='viridis');


    synch_cool.plot_final_emission()

    
    synch_cool = SynchrotronCooling_ImpulsivePLInjection(B=1E10,
                                index=-3.5,
                                gamma_injection=1E3,
                                gamma_cool=500,
                                gamma_max=1E5,
                                store_progress=True)





    synch_cool.run(photon_energies=np.logspace(1,7,50))

    synch_cool.plot_photons_and_electrons(skip=20,alpha=.7,cmap='viridis');


def test_synchrotron_cooling_acceleration():

    solver_mfc = SynchCoolAccel_ImpulsivePLInjection(n_grid_points=1000,
                            B=5E9,
                            gamma_injection=1E3,
                            gamma_cool=1000.,
                            index=-3.5,
                            acceleration_index=2,
                            t_acc_fraction=1.E2,
                                                 
                            store_progress=True,
                            )

    solver_mfc.run(photon_energies=np.logspace(1,7,100))

    solver_mfc.plot_photons_and_electrons(skip=100);


    solver_mfc = SynchCoolAccel_ContinuousPLInjection(n_grid_points=1000,
                            B=5E9,
                            gamma_injection=1E3,
                            gamma_cool=1000.,
                            index=-3.5,
                            acceleration_index=2,
                            t_acc_fraction=1.E2,
                                                 
                            store_progress=True,
                            )

    solver_mfc.run(photon_energies=np.logspace(1,7,100))

    solver_mfc.plot_photons_and_electrons(skip=100);
