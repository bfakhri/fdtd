#!/usr/bin/env python

import sys
sys.path.append('/home/bij/Projects/fdtd/')
import fdtd
import fdtd.backend as bd
import matplotlib.pyplot as plt
from PIL import Image
import torch
import numpy as np

# ## Set Backend
#fdtd.set_backend("numpy")
fdtd.set_backend("torch")


# ## Simulation Constants
WAVELENGTH = 1550e-9 # For resolution purposes.
SPEED_LIGHT: float = 299_792_458.0  # [m/s] speed of light

# Frequency of the brainwaves.
#BRAIN_FREQ = 340000000000.0 # Hz
#BRAIN_FREQ = 5*340000000000.0 # Hz
BRAIN_FREQ = 3400000000000.0 # Hz
#BRAIN_FREQ = 34000000000000.0 # Hz
#BRAIN_FREQ = 340000000000000.0 # Hz
#BRAIN_FREQ = 3400000000000000.0 # Hz
#BRAIN_FREQ = 3400000000.0 # Hz


# ## Simulation

# create FDTD Grid

# In[4]:


grid = fdtd.Grid(
    #(1.5e-5, 1.5e-5, 1),
    #(1.0e-5, 1.0e-5, 1), # Good ratios
    #(1.5e-4, 1.5e-4, 1),
    #(2.5e-5, 2.5e-5, 1),
    #(6.5e-5, 6.5e-5, 1),
    (12.5e-5, 12.5e-5, 1),
    grid_spacing=0.1 * WAVELENGTH,
    permittivity=1.0,
    permeability=1.0,
)

print('Grid Shape: ', grid.shape)


# boundaries

# In[5]:


# grid[0, :, :] = fdtd.PeriodicBoundary(name="xbounds")
grid[0:10, :, :] = fdtd.PML(name="pml_xlow")
grid[-10:, :, :] = fdtd.PML(name="pml_xhigh")

# grid[:, 0, :] = fdtd.PeriodicBoundary(name="ybounds")
grid[:, 0:10, :] = fdtd.PML(name="pml_ylow")
grid[:, -10:, :] = fdtd.PML(name="pml_yhigh")

grid[:, :, 0] = fdtd.PeriodicBoundary(name="zbounds")


# sources
xe = grid.shape[0] - 10
ye = grid.shape[1] - 10
bb = 50

#grid[ bb, bb:xe, 0] = fdtd.LineSource(
#    period=1.0 / BRAIN_FREQ, name="linesource0"
#)
#grid[ye-bb, bb:xe, 0] = fdtd.LineSource(
#    period=1.0 / BRAIN_FREQ, name="linesource1"
#)
#grid[ bb:ye, bb, 0] = fdtd.LineSource(
#    period=1.0 / BRAIN_FREQ, name="linesource2"
#)
grid[bb:ye, xe-bb, 0] = fdtd.LineSource(
    period=1.0 / BRAIN_FREQ, name="linesource3"
)


# detectors

# grid[12e-6, :, 0] = fdtd.LineDetector(name="detector")


# objects

midpoint_y = grid.shape[0]//2
midpoint_x = grid.shape[1]//2
size = 20
#grid[ midpoint_y+0, midpoint_x+70, 0] = fdtd.PointSource(
#    period=1.0 / BRAIN_FREQ, name="pointsource", amplitude=1.0,
#)


grid[40:-40, 40:-40, :] = fdtd.LearnableAnisotropicObject(permittivity=1.0, is_substrate=False, name="cc_substrate")
conv = torch.nn.Conv2d( 2, 3*3, kernel_size=1, stride=1, padding='same')
#grid.objects[0].nonlin_conv = lambda x : torch.ones_like(conv(x))
print(grid.objects[0].inverse_permittivity.shape)
yl, xl = grid.objects[0].Ny, grid.objects[0].Nx

# Import the image
image = Image.open('enlightenment_world2.png')
image = image.resize((yl, xl))

print(image.format)
print(image.size)
print(image.mode)
print(np.asarray(image).shape)
image = np.asarray(image).astype(np.float32) / 255.0
image = np.stack([image]*9, axis=0)
print(image.shape)
print('minmax: ', np.min(image), np.max(image))
#input('klj')

#iimg = torch.ones(9, yl, xl)
#iimg = torch.ones(9, yl, xl) * 100
iimg = torch.from_numpy(image) * 1000 + 1
print(iimg)
#iimg[:, midpoint_y:midpoint_y + size, midpoint_x:midpoint_x + size] = 10
grid.objects[0].seed(1.0/iimg)

# grid[midpoint_y, midpoint_x, 0] = fdtd.PointSource(
#     period=WAVELENGTH2 / SPEED_LIGHT, amplitude=0.001, name="pointsource0"
# )

# ## Run simulation

# ## Visualization
torch.set_grad_enabled(False)



grid.visualize(z=0, animate=True, norm="log")
#vis_steps = 20
vis_steps = 150
step = 0
for i in range(1000000):
    grid.run(vis_steps, progress_bar=False)
    grid.visualize(z=0, norm='log', animate=True, objcolor=(0, 0, 0, 0), objedgecolor=(1,1,1,0), plot_both_fields=False, save=True, folder='./sim_frames_enlight_world7/', index=i, clean_img=True)
    #grid.visualize(z=0, norm='log', animate=True, objcolor=(0, 0, 0, 0), objedgecolor=(1,1,1,0), plot_both_fields=False, save=False, folder='./sim_frames/', index=i, clean_img=True)
    plt.show()
    step += vis_steps
    print('On step: ', step)
x = input('type input to end: ')




