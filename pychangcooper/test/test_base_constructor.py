import pytest
import numpy as np

from pychangcooper import ChangCooper



class DummyChild(ChangCooper):

    def _define_terms(self):

        self._dispersion_term = np.ones(self._n_grid_points)

        self._heating_term = np.ones(self._n_grid_points)

class DummyChildNoDispersion(ChangCooper):

    def _define_terms(self):

        self._dispersion_term = np.zeros(self._n_grid_points)

        self._heating_term = np.ones(self._n_grid_points)




def test_chang_cooper_constructor():

    # check that we cannot build the base class
    with pytest.raises(AttributeError):
        pc = ChangCooper()

    grid_size = 10

    dummy = DummyChild(n_grid_points=grid_size , max_grid = 100., delta_t = 1.)

    assert len(dummy._grid) == grid_size
    assert len(dummy._grid2) == grid_size
    assert len(dummy._half_grid) == grid_size - 1
    assert len(dummy._half_grid2) == grid_size - 1
    assert len(dummy._delta_grid) == grid_size + 1
    assert len(dummy._delta_grid_bar) == grid_size
    
    


    # there was no souce specified
    assert np.all(dummy._source_grid == 0)

    assert np.all(dummy.n == 0)

    assert len(dummy.n) == grid_size
    
    assert len(dummy.history) == 0


    # now with no dispersion

    
    dummy = DummyChildNoDispersion(n_grid_points=grid_size)

    assert np.all(dummy.delta_j == 0)



def test_history():

    grid_size = 10

    dummy = DummyChild(n_grid_points=grid_size , max_grid = 100., delta_t = 1.)

    dummy.solve_time_step()


    assert len(dummy.history) == 0


    dummy = DummyChild(n_grid_points=grid_size , max_grid = 100., delta_t = 1., store_progress=True)

    dummy.solve_time_step()
    

    save_n1 = dummy.n
    
    assert len(dummy.history) == 1


    dummy.solve_time_step()


    save_n2 = dummy.n
    
    assert len(dummy.history) == 2

    assert np.all(save_n1 == dummy.history[0])

    assert np.all(save_n2 == dummy.history[1])

    dummy.plot_evolution()
