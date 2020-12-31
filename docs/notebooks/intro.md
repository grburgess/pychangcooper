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
%matplotlib notebook
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from pychangcooper import ChangCooper

```

# Specifying a problem

## Defining heating and dispersion terms.
The ChangCooper class automatically specifies the appropriate difference scheme for any time-independent heating and acceleration terms.  

```python
class MySolver(ChangCooper):
    
    def __init__(self):
        
        # we have no injection, so we must have an
        # initial non-zero distribution function
        
        init_distribution = np.ones(100)
        
        
        # must pass up to the super class so that
        # all terms are setup after initialization
        
        super(MySolver, self).__init__(n_grid_points=100,
                                       delta_t= 1.,
                                       max_grid=1E5,
                                       initial_distribution=init_distribution,
                                       store_progress=True # store each time step
                                      )
        
    def _define_terms(self):
        
        # energy dependent heating and dispersion terms
        # must be evaluated at half grid points.
        
        # These half grid points are automatically
        # calculated about object creation.
        
        
        
        self._heating_term = self._half_grid
        
        self._dispersion_term = self._half_grid2
    
    
    
    
```

To run the solver, simply call the solution method. If the store_progress option has been set, then each solution is stored in the objects history.

```python
solver = MySolver()

# amount of time that has gone by
print(solver.current_time)

# number of 
print(solver.n_iterations)

# current solution
print(solver.n)
```

```python
solver.solve_time_step()


# amount of time that has gone by
print(solver.current_time)

# number of 
print(solver.n_iterations)

# current solution
print(solver.n)
```

We can plot the evolution of the solution if we have been storing it.

```python
for i in range(10):
    
    solver.solve_time_step()

solver.plot_evolution(alpha=.8);
```

## Adding source and escape terms

The general Chang and Cooper scheme does not specify injection and esacpe. But we can easily add them on. In this case, the Fokker-Planck equation reads:

$$\frac{\partial N\left(\gamma, t\right)}{\partial t}  = \frac{\partial }{\partial \gamma} \left[ B \left(\gamma, t \right)N\left(\gamma, t\right) + C \left(\gamma, t \right) \frac{\partial N\left(\gamma, t\right)}{\partial \gamma}\right] - E\left(\gamma, t \right) + Q\left(\gamma, t \right) \text{.}$$

In order ot include these terms, we simply need to define a source and escape function which will be evaluated on the grid at each iteration of the solution.


```python
class MySolver(ChangCooper):
    
    def __init__(self):
        
        # must pass up to the super class so that
        # all terms are setup after initialization
        
        super(MySolver, self).__init__(n_grid_points=100,
                                       delta_t= 1., # the time step of the solution
                                       max_grid=1E5,
                                       initial_distribution=None,
                                       store_progress=True
                                      )
        
    def _define_terms(self):
        
        # energy dependent heating and dispersion terms
        # must be evaluated at half grid points.
        
        # These half grid points are automatically
        # calculated about object creation.
        
        
        
        self._heating_term = self._half_grid
        
        self._dispersion_term = self._half_grid2
    
    
    def _source_function(self, gamma):
        
        # power law injection 
        return gamma**2
    
    def _escape_function(self, gamma):
        
        # constant, energy-independent escape term
        return 0.5 * np.ones_like(gamma)
        
        
```

Upon object creation, the source and escape terms are automatically evaluated.

```python
solver = MySolver()

for i in range(10):
    
    solver.solve_time_step()

solver.plot_evolution(alpha=.8, cmap='winter');
```

```python

```
