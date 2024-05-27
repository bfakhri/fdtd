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
WAVELENGTH1 = 1550e-9
SPEED_LIGHT: float = 299_792_458.0  # [m/s] speed of light


# ## Simulation

# create FDTD Grid

# In[4]:


grid = fdtd.Grid(
    (1.5e-5, 1.5e-5, 1),
    grid_spacing=0.05 * WAVELENGTH1,
    permittivity=1.0,
    permeability=1.0,
)
grid.time_step = grid.time_step / 10

print('Grid Shape: ', grid.shape)


# boundaries

# In[5]:


grid[:, :, 0] = fdtd.PeriodicBoundary(name="zbounds")


# sources


#grid[40, 45, 0] = fdtd.PointSource(
#    period=0.05 * WAVELENGTH1 / SPEED_LIGHT, name="pointsource1"
#)
#grid[40, 45, 0] = fdtd.PointSource(
#    period=0.1 * WAVELENGTH1 / SPEED_LIGHT, name="pointsource2"
#)
#grid[40, 45, 0] = fdtd.PointSource(
#    period=1 * WAVELENGTH1 / SPEED_LIGHT, name="pointsource3"
#)
#grid[40, 45, 0] = fdtd.PointSource(
#    period=10 * WAVELENGTH1 / SPEED_LIGHT, name="pointsource4"
#)

for i in range(8):
    if(i % 2 == 0):
        grid[40, 45, 0] = fdtd.PointSource(
            period=(2**i) * 0.05 * WAVELENGTH1 / SPEED_LIGHT, name="pointsource_" + str(i)
        )




grid.visualize(z=0, animate=True)
for i in range(1000000):
    grid.run(10, progress_bar=False)
    grid.visualize(z=0, norm='log', animate=True, save=True, folder='./sim_frames/', index=i, clean_img=True)
    plt.show()
x = input('type input to end: ')




