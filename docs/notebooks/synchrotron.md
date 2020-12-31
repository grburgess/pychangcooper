---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.8.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
import numpy as np

%matplotlib inline
import matplotlib.pyplot as plt
from jupyterthemes import jtplot
jtplot.style(context='talk', fscale=1.4, spines=False, gridlines='--')

from pychangcooper import SynchrotronCooling_ContinuousPLInjection

```

<!-- #region -->
# Synchrotron cooling

In the case of simple synchrotron cooling, the diffusion/dispersion term is zero. However, we will have a source term, $Q(\gamma, t)$. Thus we have the simple case of:

$$\frac{\partial N\left(\gamma, t\right)}{\partial t}  = \frac{\partial }{\partial \gamma} B \left(\gamma, t \right)N\left(\gamma, t\right)  + Q(\gamma, t)$$


The synchrotron cooling class computes the cooling time of the electrons and sets the time step of the solver to the cooling time of the highest energy electron which evolves the most.

## Solution
### Setup the solver
First we set the solver with the initial physical parameters.
<!-- #endregion -->

```python
synch_cool = SynchrotronCooling_ContinuousPLInjection(B=1E10,
                                index=-3.5,
                                gamma_injection=1E3,
                                gamma_cool=500,
                                gamma_max=1E5,
                                store_progress=True)
```

### Solving

The solver automatically sets of the number of iterations such that the dynamical time equals the cooling time. If an array of photon energies are also passed, then the photon spectrum that is emitted by the electrons is also computed.

```python
synch_cool.run(photon_energies=np.logspace(1,7,50))
```

```python
synch_cool.plot_photons_and_electrons(skip=20,alpha=.7,cmap='vaporwave');
```

