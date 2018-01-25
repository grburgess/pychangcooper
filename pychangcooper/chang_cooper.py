import numpy as np

from pychangcooper.tridiagonal_solver import TridiagonalSolver

class ChangCooper(object):
    def __init__(self,
                 n_grid_points=300,
                 max_grid=1E5,
                 delta_t=1.,
                 initial_distribution=None,
                 store_progress=False):
        """
        Generic Chang and Cooper base class. Currently, the dispersion and heating terms 
        are assumed to be time-independent

        :param n_grid_points: number of grid points on the x-axis
        :param max_grid: the maximum energy of the grid
        """

        self._n_grid_points = n_grid_points
        self._max_grid = max_grid
        self._delta_t = delta_t
        self._iterations = 0
        self._current_time = 0.
        self._store_progress = store_progress
        self._saved_grids = []

        # first build the grid which is independent of the scheme
        self._build_grid()

        if initial_distribution is None:

            # initalize the grid of electrons
            self._n_current = np.zeros(self._n_grid_points)

        else:

            assert len(initial_distribution) == self._n_grid_points

            self._n_current = np.array(initial_distribution)

        # define the heating and dispersion terms
        # must be implemented in the subclasses
        self._define_terms()

        # compute the source function if there is any
        self._compute_source_function()

        # compute the delta_js which control the upwind and downwind scheme
        self._compute_delta_j()

        # now compute the tridiagonal terms
        self._setup_vectors()

        # if there are no dispersion terms we do not need to forward sweep
        self._a_non_zero = ~np.all(self._a == 0)

    def _build_grid(self):
        """
        setup the grid for the calculations and initialize the 
        solution
        """
        # logarithmic grid
        step = np.exp((1. / self._n_grid_points) * np.log(self._max_grid))
        step_plus_one = step + 1.

        # initialize all the grid points
        self._grid = np.zeros(self._n_grid_points)
        self._grid2 = np.zeros(self._n_grid_points)
        self._half_grid = np.zeros(self._n_grid_points)
        self._half_grid2 = np.zeros(self._n_grid_points)

        # now build the grid and the half grid points
        # we also make squared terms just incase
        for i in range(self._n_grid_points):

            self._grid[i] = np.power(step, i)
            self._grid2[i] = self._grid[i] * self._grid[i]

            if (i < self._n_grid_points - 1):

                self._half_grid[i] = 0.5 * self._grid[i] * step_plus_one
                self._half_grid2[i] = self._half_grid[i] * self._half_grid[i]

            else:

                self._half_grid[self._n_grid_points
                                - 1] = 0.5 * self._grid[i] * step_plus_one
                self._half_grid2[self._n_grid_points - 1] = self._half_grid[
                    self._n_grid_points
                    - 1] * self._half_grid[self._n_grid_points - 1]

        # define the delta of the grid
        self._delta_half_grid = np.diff(self._grid)
        self._delta_grid = np.diff(self._half_grid)

        
    def _compute_delta_j(self):
        """
        delta_j controls where the differences are computed. If there are no dispersion
        terms, then delta_j is zero
        """

        self._delta_j = np.zeros(self._n_grid_points)

        for j in range(self._n_grid_points - 1):

            if self._dispersion_term[j] != 0:

                w = (self._delta_half_grid[j] * self._heating_term[j]) / (
                    self._dispersion_term[j])

                if w == 0:
                    self._delta_j[j] = 0.5

                else:

                    self._delta_j[j] = (1. / w) - 1. / (np.exp(w) - 1.)

    def _setup_vectors(self):
        """
        from the specified terms in the subclasses, setup the tridiagonal terms
        
        """

        # initialize everything to zerp

        self._a = np.zeros(self._n_grid_points)
        self._b = np.zeros(self._n_grid_points)
        self._c = np.zeros(self._n_grid_points)

        # walk backwards in j starting from the second to last index
        # then set the end points
        for j_minus_one in range(self._n_grid_points - 2, 0, -1):

            # j is one step ahead
            j = j_minus_one + 1

            # pre compute one over the delta of the grid
            one_over_delta_grid = 1. / self._delta_half_grid[j_minus_one]

            # n_j-1 term
            self._a[j_minus_one] = one_over_delta_grid * (
                one_over_delta_grid * self._dispersion_term[j_minus_one] -
                self._delta_j[j_minus_one] * self._heating_term[j_minus_one])

            # n_j term
            self._b[j_minus_one] = - one_over_delta_grid * (
                one_over_delta_grid *
                (self._dispersion_term[j] + self._dispersion_term[j_minus_one])
                + (1 - self._delta_j[j_minus_one]
                   ) * self._heating_term[j_minus_one] -
                self._delta_j[j] * self._heating_term[j])

            # n_j+1 term
            self._c[j_minus_one] = one_over_delta_grid * (
                (1 - self._delta_j[j]) * self._heating_term[j] +
                one_over_delta_grid * self._dispersion_term[j])

        # now set the end points

        # right boundary
        # j+1/2 = 0

        one_over_delta_grid = 1. / self._delta_half_grid[-1]
        # n_j-1 term
        self._a[-1] = one_over_delta_grid * (
            one_over_delta_grid * self._dispersion_term[-1] -
            self._delta_j[-1] * self._heating_term[-1])

        # n_j term
        self._b[-1] = - one_over_delta_grid * (one_over_delta_grid * (self._dispersion_term[-1]) + (1 - self._delta_j[-1]) * self._heating_term[j_minus_one]) 

        # n_j+1 term
        self._c[-1] = 0

        # left boundary
        # j-1/2 = 0

        one_over_delta_grid = 1. / self._delta_half_grid[0]

        # n_j-1 term
        self._a[0] = 0.

        # n_j term
        self._b[0] = -one_over_delta_grid* ( one_over_delta_grid*(self._dispersion_term[0])\
                                           - self._delta_j[0] * self._heating_term[0]
                                                    ) 
        # n_j+1 term
        self._c[0] = one_over_delta_grid*( ( 1- self._delta_j[0]) * self._heating_term[0] \
                                                    + one_over_delta_grid* self._dispersion_term[0]  )

        # carry terms to the other side
        self._a *= -self._delta_t
        self._c *= -self._delta_t
        self._b = (1 - self._b * self._delta_t)

        # now make a tridiagonal_solver for these terms
        
        self._tridiagonal_solver = TridiagonalSolver(self._a, self._b, self._c)
        
        
    def _compute_source_function(self):
        """
        compute the grid of the source term. This will just be zero if there is nothing 
        added and all will be zero
        """

        self._source_grid = self._source_function(self._grid)

    def _source_function(self, energy):

        return 0.

    def _define_terms(self):

        RuntimeError('Must be implemented in subclass')



    def solve_time_step(self):


        
        if self._store_progress:

            self._saved_grids.append(self._n_current)

        
        d = self._n_current + self._source_grid * self._delta_t


        # set the new solution to the current one
        self._n_current = self._tridiagonal_solver.solve(d)


        # clean any numerical diffusion if needed
        # this must be customized
        self._clean()

        # bump up the iteration number and the time 
        self._iteratate()

    def _clean(self):

        pass

    def _iteratate(self):
        """
        increase the run iterator and the current time
        """

        self._iterations += 1
        self._current_time += self._delta_t

    

    @property
    def current_time(self):

        return self._current_time

    @property
    def n_iterations(self):

        return self._iterations
        
    @property
    def delta_j(self):
        """
        the delta_js 
        """

        return self._delta_j

    @property
    def grid(self):
        """
        The energy grid
        """

        return self._grid

    @property
    def half_grid(self):
        """
        The half energy grid
        """

        return self._half_grid

    @property
    def n(self):

        return self._n_current

    @property
    def history(self):

        return np.array(self._saved_grids)
