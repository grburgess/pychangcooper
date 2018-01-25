import pytest
import numpy as np

from pychangcooper.tridiagonal_solver import TridiagonalSolver


def test_tridiagonal_constructor():

    a = np.ones(3)
    b = np.ones(3)
    c = np.ones(3) 

    b_too_short = np.ones(2)
    c_too_long = np.ones(4)
    a_zero = np.zeros(3)
    
    ts = TridiagonalSolver(a,b,c)

    with pytest.raises(AssertionError):

        ts = TridiagonalSolver(a,b_too_short,c)

    
    with pytest.raises(AssertionError):

        ts = TridiagonalSolver(a,b,c_too_long)

        
    # test a is zero

    ts_zero = TridiagonalSolver(a_zero,b,c)

    assert not ts_zero._a_non_zero


    
    ts_non_zero = TridiagonalSolver(a,b,c)

    assert ts_non_zero._a_non_zero

def test_solving():

    a = np.ones(3)
    b = np.ones(3)
    c = np.ones(3) 

    d = np.ones(3)

    a_zero = np.zeros(3)
    
        
    # test a is zero

    ts_zero = TridiagonalSolver(a_zero,b,c)

    c_prime = ts_zero._cprime

    _ = ts_zero.solve(d)
    # make sure c prime does not change
    assert np.all(ts_zero._cprime == c_prime)

    


    
    ts_non_zero = TridiagonalSolver(a,b,c)

    
