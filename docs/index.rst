.. pychangcooper documentation master file, created by
   sphinx-quickstart on Sat Jan 27 12:17:10 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pychangcooper: a generic Fokker-Planck solver
=========================================
A simple numerical solver for Fokker-Planck style equations of the form:
.. math::
   \frac{\partial N\left(\gamma, t\right)}{\partial t}  = \frac{\partial }{\partial \gamma} \left[ B \left(\gamma, t \right) + C \left(\gamma, t \right) \frac{\partial N\left(\gamma, t\right)}{\partial \gamma}\right]
   
designed with an object-oriented interface to allow for easy problem specification via subclassing.


.. toctree::
   :maxdepth: 5

   notebooks/intro.ipynb
   notebooks/heat_accel.ipynb
   notebooks/synchrotron.ipynb

.. automodule:: pychangcooper
   :members:
