#!/usr/bin/env python

import sys
sys.path.append('/home/bij/Projects/fdtd/')
import fdtd
import fdtd.backend as bd
import matplotlib.pyplot as plt


# ## Set Backend
#fdtd.set_backend("numpy")
fdtd.set_backend("torch")


# ## Constants
WAVELENGTH = 1550e-9
#WAVELENGTH2 = 1550e-8
WAVELENGTH2 = 1550e-10
SPEED_LIGHT: float = 299_792_458.0  # [m/s] speed of light


# ## Simulation

# create FDTD Grid

# In[4]:


grid = fdtd.Grid(
    (1.5e-5, 1.5e-5, 1),
    grid_spacing=0.05 * WAVELENGTH,
    permittivity=1.0,
    permeability=1.0,
)
grid.time_step = grid.time_step / 10

print('Grid Shape: ', grid.shape)


# boundaries

# In[5]:


grid[:, :, 0] = fdtd.PeriodicBoundary(name="zbounds")


# sources

# grid[50, 70:75, 0] = fdtd.LineSource(
#     period=WAVELENGTH / SPEED_LIGHT, name="linesource"
# )
# grid[70, 70:75, 0] = fdtd.LineSource(
#     period=WAVELENGTH / SPEED_LIGHT, name="linesource2",
# )

grid[40, 45, 0] = fdtd.PointSource(
    period=WAVELENGTH2 / SPEED_LIGHT, name="pointsource"
)


# detectors

# grid[12e-6, :, 0] = fdtd.LineDetector(name="detector")


# objects

# ## Run simulation

# ## Visualization


grid.visualize(z=0, animate=True)
for i in range(1000000):
    grid.run(10, progress_bar=False)
    grid.visualize(z=0, norm='log', animate=True)
    plt.show()
x = input('type input to end: ')




