import numpy as np
import matplotlib.pyplot as plt

from pychangcooper.io.fill_plot import fill_plot_static
from pychangcooper.tridiagonal_solver import TridiagonalSolver


def log_grid_generator(n_steps, grid_max):
    step = np.exp((1.0 / n_steps) * np.log(grid_max))
    step_plus_one = step + 1.0

    # initialize all the grid points
    grid = np.zeros(n_steps)

    half_grid = np.zeros(n_steps - 1)

    # now build the grid and the half grid points
    # we also make squared terms just incase
    for i in range(n_steps):

        grid[i] = np.power(step, i)

        if i < n_steps - 1:
            half_grid[i] = 0.5 * grid[i] * step_plus_one

    return grid, half_grid, step


class ChangCooper(object):
    def __init__(
        self,
        n_grid_points=300,
        max_grid=1e5,
        delta_t=1.0,
        initial_distribution=None,
        store_progress=False,
    ):
        """
        Generic Chang and Cooper base class. Currently, the dispersion and heating terms 
        are assumed to be time-independent

        :param n_grid_points: number of grid points on the x-axis
        :param max_grid: the maximum energy of the grid
        :param delta_t: the time step in the equation
        :param initial_distribution: an array of an initial electron distribution
        :param store_progress: store the history of the runs
        """

        self._n_grid_points = n_grid_points
        self._dispersion_term = np.zeros(n_grid_points)
        self._heating_term = np.zeros(n_grid_points)

        self._max_grid = max_grid
        self._delta_t = delta_t
        self._iterations = 0
        self._current_time = 0.0
        self._store_progress = store_progress
        self._saved_grids = []

        # first build the grid which is independent of the scheme
        self._build_grid()

        if initial_distribution is None:

            # initalize the grid of electrons
            self._n_current = np.zeros(self._n_grid_points)
            self._initial_distribution = np.zeros(n_grid_points)

        else:

            assert len(initial_distribution) == self._n_grid_points

            self._n_current = np.array(initial_distribution)
            self._initial_distribution = initial_distribution

        # define the heating and dispersion terms
        # must be implemented in the subclasses
        self._define_terms()

        # compute the source/escape function if there is any
        self._compute_source_function_and_escape()

        # compute the delta_js which control the upwind and downwind scheme
        self._compute_delta_j()

        # now compute the tridiagonal terms
        self._setup_vectors()

    def _build_grid(self):
        """
        setup the grid for the calculations and initialize the 
        solution
        """

        # logarithmic grid

        self._grid, self._half_grid, self._step = log_grid_generator(
            self._n_grid_points, self._max_grid
        )
        self._grid2 = self._grid ** 2
        self._half_grid2 = self._half_grid ** 2

        # define the delta of the grid

        # we need to add extra end points to the grid
        # so that we can compute the delta at the boundaries

        first_delta_grid = self._grid[0] * (1 - 1.0 / self._step)
        last_delta_grid = self._grid[-1] * (self._step - 1)

        delta_grid = np.append([first_delta_grid], np.diff(self._grid))
        delta_grid = np.append(delta_grid, [last_delta_grid])

        assert len(delta_grid) == self._n_grid_points + 1

        # delta grid bar is  the average of the grid and applies to
        # to the forward and backwards difference terms
        # When the grid is not uniform, the second derivatives need
        # to be taken in the correct place

        delta_grid_bar = np.mean(np.vstack((delta_grid[:-1], delta_grid[1:])), axis=0)

        assert len(delta_grid_bar) == self._n_grid_points

        self._delta_grid = delta_grid
        self._delta_grid_bar = delta_grid_bar

    def _compute_delta_j(self):
        """
        delta_j controls where the differences are computed. If there are no dispersion
        terms, then delta_j is zero
        """

        # set to zero. note delta_j[n] = 0 by default

        self._delta_j = np.zeros(self._n_grid_points - 1)

        for j in range(self._n_grid_points - 1):

            # if the dispersion term is 0 => delta_j = 0
            if self._dispersion_term[j] != 0:

                w = (
                    self._delta_grid[1:-1][j] * self._heating_term[j]
                ) / self._dispersion_term[j]

                # w asymptotically approaches 1/2, but we need to set it manually
                if w == 0:

                    self._delta_j[j] = 0.5

                # otherwise, we use appropriate bounds
                else:

                    self._delta_j[j] = (1.0 / w) - 1.0 / (np.exp(w) - 1.0)

        # precomoute 1- delta_j
        self._one_minus_delta_j = 1 - self._delta_j

    def _setup_vectors(self):
        """
        from the specified terms in the subclasses, setup the tridiagonal terms
        
        """

        # initialize everything to zero

        a = np.zeros(self._n_grid_points)
        b = np.zeros(self._n_grid_points)
        c = np.zeros(self._n_grid_points)

        # walk backwards in j starting from the second to last index
        # then set the end points
        for k in range(self._n_grid_points - 2, 0, -1):
            # pre compute one over the delta of the grid
            # this is the 1/delta_grid in front of the F_j +/- 1/2.

            one_over_delta_grid_forward = 1.0 / self._delta_grid[k + 1]
            one_over_delta_grid_backward = 1.0 / self._delta_grid[k]

            # this is the delta grid in front of the full equation

            one_over_delta_grid_bar = 1.0 / self._delta_grid_bar[k]

            # The B_j +/- 1/2 from CC
            B_forward = self._heating_term[k]
            B_backward = self._heating_term[k - 1]

            # The C_j +/- 1/2 from CC
            C_forward = self._dispersion_term[k]
            C_backward = self._dispersion_term[k - 1]

            # in order to keep math errors at a minimum, the tridiagonal terms
            # are computed in separate functions so that boundary conditions are
            # set consistently.

            # First we solve (N - N) = F
            # then we will move the terms to form a tridiagonal equation

            # n_j-1 term
            a[k] = _compute_n_j_minus_one_term(
                one_over_delta_grid=one_over_delta_grid_bar,
                one_over_delta_grid_bar_backward=one_over_delta_grid_backward,
                C_backward=C_backward,
                B_backward=B_backward,
                delta_j_minus_one=self._delta_j[k - 1],
            )

            # n_j term
            b[k] = _compute_n_j(
                one_over_delta_grid=one_over_delta_grid_bar,
                one_over_delta_grid_bar_backward=one_over_delta_grid_backward,
                one_over_delta_grid_bar_forward=one_over_delta_grid_forward,
                C_backward=C_backward,
                C_forward=C_forward,
                B_backward=B_backward,
                B_forward=B_forward,
                one_minus_delta_j_minus_one=self._one_minus_delta_j[k - 1],
                delta_j=self._delta_j[k],
            )

            # n_j+1 term
            c[k] = _compute_n_j_plus_one(
                one_over_delta_grid=one_over_delta_grid_bar,
                one_over_delta_grid_bar_forward=one_over_delta_grid_forward,
                C_forward=C_forward,
                B_forward=B_forward,
                one_minus_delta_j=self._one_minus_delta_j[k],
            )

        # now set the end points

        ################
        # right boundary
        # j+1/2 = 0

        one_over_delta_grid_forward = 0.0
        one_over_delta_grid_backward = 1.0 / self._delta_grid[-1]

        one_over_delta_grid_bar = 1.0 / self._delta_grid_bar[-1]

        # n_j-1 term
        a[-1] = _compute_n_j_minus_one_term(
            one_over_delta_grid=one_over_delta_grid_bar,
            one_over_delta_grid_bar_backward=one_over_delta_grid_backward,
            C_backward=self._dispersion_term[-1],
            B_backward=self._heating_term[-1],
            delta_j_minus_one=self._delta_j[-1],
        )

        # n_j term
        b[-1] = _compute_n_j(
            one_over_delta_grid=one_over_delta_grid_bar,
            one_over_delta_grid_bar_backward=one_over_delta_grid_backward,
            one_over_delta_grid_bar_forward=one_over_delta_grid_forward,
            C_backward=self._dispersion_term[-1],
            C_forward=0,
            B_backward=self._heating_term[-1],
            B_forward=0,
            one_minus_delta_j_minus_one=self._one_minus_delta_j[-1],
            delta_j=0,
        )

        # n_j+1 term
        c[-1] = 0

        ###############
        # left boundary
        # j-1/2 = 0

        one_over_delta_grid = 1.0 / (
            self._half_grid[0] - self._grid[0] / np.sqrt(self._step)
        )

        one_over_delta_grid_bar_forward = 1.0 / self._delta_grid_bar[0]
        one_over_delta_grid_bar_backward = 0.0

        one_over_delta_grid_forward = 1.0 / self._delta_grid[0]
        one_over_delta_grid_backward = 0

        one_over_delta_grid_bar = 1.0 / self._delta_grid_bar[0]

        # n_j-1 term
        a[0] = 0.0

        # n_j term
        b[0] = _compute_n_j(
            one_over_delta_grid=one_over_delta_grid_bar,
            one_over_delta_grid_bar_backward=one_over_delta_grid_backward,
            one_over_delta_grid_bar_forward=one_over_delta_grid_forward,
            C_backward=0,
            C_forward=self._dispersion_term[0],
            B_backward=0,
            B_forward=self._heating_term[0],
            one_minus_delta_j_minus_one=0,
            delta_j=self._delta_j[0],
        )
        # n_j+1 term
        c[0] = _compute_n_j_plus_one(
            one_over_delta_grid=one_over_delta_grid_bar,
            one_over_delta_grid_bar_forward=one_over_delta_grid_forward,
            C_forward=self._dispersion_term[0],
            B_forward=self._heating_term[0],
            one_minus_delta_j=self._one_minus_delta_j[0],
        )

        # carry terms to the other side to form a tridiagonal equation
        # the escape term is added on but is zero unless created in
        # a child class

        a *= -self._delta_t
        b = (1 - b * self._delta_t) + self._escape_grid * self._delta_t
        c *= -self._delta_t

        # now make a tridiagonal_solver for these terms

        self._tridiagonal_solver = TridiagonalSolver(a, b, c)

    def _compute_source_function_and_escape(self):
        """
        compute the grid of the source term. This will just be zero if there is nothing 
        added and all will be zero
        """

        self._source_grid = self._source_function(self._grid)
        self._escape_grid = self._escape_function(self._grid)

    def _source_function(self, energy):

        return 0.0

    def _escape_function(self, energy):

        return 0

    def _define_terms(self):

        raise RuntimeError("Must be implemented in subclass")

    def solve_time_step(self):
        """
        Solve for the next time step. 
        """

        # if we are storing the solutions, then append them
        # to the history

        if self._store_progress:
            self._saved_grids.append(self._n_current)

        # set up the right side of the tridiagonal equation.
        # This is the current distribution plus the source
        # unless it is zero

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

        # bump up the iteration number

        self._iterations += 1

        # increase the time

        self._current_time += self._delta_t

    @property
    def current_time(self):
        """
        The current time: delta_t * n_iterations
        """

        return self._current_time

    @property
    def n_iterations(self):
        """
        The number of iterations solved for
        """

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
        """
        The current solution
        """

        return self._n_current

    @property
    def history(self):
        """
        The history of the solution
        """

        return np.array(self._saved_grids)

    def reset(self):
        """
        reset the solver to the initial electron distribution
        
        :return: 
        """

        self._n_current = self._initial_distribution
        self._iterations = 0
        self._current_time = 0.0

    def plot_current_distribution(self, ax=None, **kwargs):

        if ax is None:

            fig, ax = plt.subplots()

            ax.set_xlabel(r"$\gamma$")
            ax.set_ylabel(r"$N(\gamma, t)$")

            ax.set_xscale("log")
            ax.set_yscale("log")

        else:

            fig = ax.get_figure()

        ax.plot(self._grid, self._n_current, **kwargs)

        return fig

    def plot_initial_distribution(self, ax=None, **kwargs):
        """
        plot the initial distribution of the electrons
        
        :param ax: ax to plot to
        :param kwargs: mpl kwargs
        :return: fig
        """

        if ax is None:

            fig, ax = plt.subplots()

            ax.set_xlabel(r"$\gamma$")
            ax.set_ylabel(r"$N(\gamma, t)$")

            ax.set_xscale("log")
            ax.set_yscale("log")

        else:

            fig = ax.get_figure()

        ax.plot(self._grid, self._initial_distribution, **kwargs)

        return fig

    def plot_evolution(
        self,
        cmap="magma",
        skip=1,
        show_legend=False,
        alpha=0.9,
        show_final=False,
        show_initial=False,
        ax=None,
    ):
        """
        plot th evolution of the electrons
        
        
        :param cmap: cmap to use
        :param skip: number of elements to skip
        :param show_legend: show a legend
        :param alpha: the transparency 
        :param show_final: label the final solution
        :param show_initial: show the initial distribution
        :param ax: the ax to plot to
        :return: 
        """

        solutions = self.history[::skip]

        fig = fill_plot_static(self._grid, solutions, cmap, alpha, ax)

        if ax is None:
            ax = fig.get_axes()[0]

        if show_final:
            _ = self.plot_current_distribution(
                ax,
                color="k",
                ls="--",
                zorder=len(solutions) + 1,
                alpha=1,
                label="final distribution",
            )

        if show_initial:
            _ = self.plot_initial_distribution(
                ax,
                color="k",
                ls=":",
                zorder=len(solutions) + 1,
                alpha=1,
                label="initial distribution",
            )

        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.set_xlabel(r"$\gamma$")
        ax.set_ylabel(r"$N(\gamma$,t)")

        if show_legend:
            ax.legend()

        return fig


def _compute_n_j_plus_one(
    one_over_delta_grid,
    one_over_delta_grid_bar_forward,
    C_forward,
    B_forward,
    one_minus_delta_j,
):
    """
    equation for the CC n_j +1 term

    :param one_over_delta_grid: the total change in energy
    :param one_over_delta_grid_bar_backward: the backward change in energy for the second derivative
    :param C_forward: the forward dispersion term
    :param B_forward: the forward heating term
    :param one_minus_delta_j: 1 - delta_j
    """

    return one_over_delta_grid * (
        one_minus_delta_j * B_forward + one_over_delta_grid_bar_forward * C_forward
    )


def _compute_n_j(
    one_over_delta_grid,
    one_over_delta_grid_bar_backward,
    one_over_delta_grid_bar_forward,
    C_backward,
    C_forward,
    B_backward,
    B_forward,
    one_minus_delta_j_minus_one,
    delta_j,
):
    """
    equation for the CC n_j term

    :param one_over_delta_grid: the total change in energy
    :param one_over_delta_grid_bar_backward: the backward change in energy for the second derivative
    :param one_over_delta_grid_bar_forward: the forward change in energy for the second derivative
    :param C_forward: the forward dispersion term
    :param C_backward: the backward dispersion term
    :param B_forward: the forward heating term
    :param B_backward: the backward heating term
    :param one_minus_delta_j_minus_one: 1 - delta_j-1
    """

    return -one_over_delta_grid * (
        (
            one_over_delta_grid_bar_forward * C_forward
            + one_over_delta_grid_bar_backward * C_backward
        )
        + one_minus_delta_j_minus_one * B_backward
        - delta_j * B_forward
    )


def _compute_n_j_minus_one_term(
    one_over_delta_grid,
    one_over_delta_grid_bar_backward,
    C_backward,
    B_backward,
    delta_j_minus_one,
):
    """
    equation for the CC n_j-1 term

    :param one_over_delta_grid: the total change in energy
    :param one_over_delta_grid_bar_forward: the forward change in energy for the second derivative
    :param C_backward: the backward dispersion term
    :param B_backward: the backward heating term
    :param one_minus_delta_j: 1 - delta_j
    """

    return one_over_delta_grid * (
        one_over_delta_grid_bar_backward * C_backward - delta_j_minus_one * B_backward
    )
