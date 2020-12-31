import numpy as np
import matplotlib.pyplot as plt

from pychangcooper.utils.progress_bar import progress_bar
from pychangcooper.io.fill_plot import fill_plot_static


class PhotonEmitter(object):
    def __init__(self, n_steps, emission_kernel):
        """FIXME! briefly describe function

        :param n_steps: 
        :param emission_kernel: 
        :returns: 
        :rtype: 

        """

        self._n_steps = n_steps

        self._emission_kernel = emission_kernel

    def run(self, photon_energies=None):
        """FIXME! briefly describe function

        :param photon_energies: 
        :returns: 
        :rtype: 

        """

        with progress_bar(int(self._n_steps), title="solving electrons electrons") as p:
            for i in range(int(self._n_steps)):
                self.solve_time_step()

                p.increase()

        if photon_energies is not None:

            self._compute_spectrum(photon_energies)

    def _compute_spectrum(self, photon_energies):
        """

        """

        self._emission_kernel.set_photon_energies(photon_energies)

        self._all_spectra = []

        with progress_bar(int(self._n_steps), title="computing spectrum") as p:
            for electrons in self.history:
                self._all_spectra.append(
                    self._emission_kernel.compute_spectrum(electrons)
                )

                p.increase()

        self._all_spectra = np.array(self._all_spectra)

        self._total_spectrum = self._all_spectra.sum(axis=0)

    @property
    def final_spectrum(self):

        return self._total_spectrum

    @property
    def photon_energies(self):

        return self._emission_kernel.photon_energies

    def plot_final_emission(self, ax=None, x_scaling=1.0, y_scaling=1.0, **kwargs):
        """FIXME! briefly describe function

        :param ax: 
        :param x_scaling: 
        :param y_scaling: 
        :returns: 
        :rtype: 

        """

        assert self.photon_energies is not None, "There are no photons!"

        if ax is None:

            fig, ax = plt.subplots()

        else:

            fig = ax.get_figure()

        ax.loglog(
            self.photon_energies * x_scaling,
            self.photon_energies ** 2 * self.final_spectrum * y_scaling,
            **kwargs
        )

        ax.set_xlabel("Energy")
        ax.set_ylabel(r"$vFv$")

        return fig

    def plot_initial_emission(self, ax=None, **kwargs):
        """FIXME! briefly describe function

        :param ax: 
        :returns: 
        :rtype: 

        """

        assert self.photon_energies is not None, "There are no photons!"
        assert np.all(self._all_spectra[0] > 0.0), "There are no photons to start!"

        if ax is None:

            fig, ax = plt.subplots()

        else:

            fig = ax.get_figure()

        ax.loglog(
            self.photon_energies,
            self.photon_energies ** 2 * self._all_spectra[0],
            **kwargs
        )

        ax.set_xlabel("Energy")
        ax.set_ylabel(r"vFv")

        return fig

    def plot_emission(self, cmap="viridis", skip=1, alpha=0.5, ax=None):
        """FIXME! briefly describe function

        :param cmap: 
        :param skip: 
        :param alpha: 
        :param ax: 
        :returns: 
        :rtype: 

        """

        assert self.photon_energies is not None, "There are no photons!"
        cumulative_spectrum = (self._all_spectra.cumsum(axis=0))[::skip]

        fig = fill_plot_static(
            self._emission_kernel.photon_energies,
            self._emission_kernel.photon_energies ** 2 * cumulative_spectrum,
            cmap,
            alpha,
            ax,
        )

        if ax is None:
            ax = fig.get_axes()[0]

        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.set_xlabel("Energy")
        ax.set_ylabel(r"vFv")

        return fig

    def plot_photons_and_electrons(self, cmap="viridis", skip=1, alpha=0.5):
        """FIXME! briefly describe function

        :param cmap: 
        :param skip: 
        :param alpha: 
        :returns: 
        :rtype: 

        """

        fig, (ax1, ax2) = plt.subplots(1, 2)

        _ = self.plot_evolution(cmap=cmap, skip=skip, ax=ax1, alpha=alpha)
        _ = self.plot_emission(cmap=cmap, skip=skip, ax=ax2, alpha=alpha)
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")

        ax1.set_xlim(left=min(self._gamma_cool, self._gamma_injection) * 0.5)

        fig.tight_layout()
        fig.subplots_adjust(hspace=0, wspace=0)

        return fig
