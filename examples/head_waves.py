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
#BRAIN_FREQ = 3400000000000.0 # Hz
BRAIN_FREQ = 4*34000000000000.0 # Hz
#BRAIN_FREQ = 340000000000000.0 # Hz
#BRAIN_FREQ = 3400000000000000.0 # Hz
#BRAIN_FREQ = 3400000000.0 # Hz

# Head model
import trimesh

# Load the 3D model
mesh = trimesh.load('/home/bij/Downloads/3d/source/basicfacerenew7.obj')
#mesh = trimesh.load('/home/bij/Downloads/3d/human_head.glb')
mesh.fill_holes()

# Define the voxel size
voxel_size = 0.01  # Adjust this based on your model's scale and required resolution

# Voxelize the mesh
voxelized = mesh.voxelized(voxel_size)

# Get the filled voxel grid
voxel_matrix = voxelized.matrix

# Convert to NumPy array
voxels_np = np.array(voxel_matrix, dtype=bool)

def flood_fill_3d(grid, start):
    """
    Perform a flood fill on a 3D grid starting from a given point.

    Parameters:
        grid (np.ndarray): The 3D grid to be filled. Non-zero values are considered as borders.
        start (tuple): The starting point (x, y, z) for the flood fill.

    Returns:
        np.ndarray: The grid after performing the flood fill.
    """
    x_max, y_max, z_max = grid.shape
    to_fill = [start]
    target_value = grid[start]

    # If starting point is already a border or out of bounds, return the grid as is
    if target_value != 0:
        return grid

    while to_fill:
        x, y, z = to_fill.pop()

        if grid[x, y, z] == 0:
            # Fill the cell
            grid[x, y, z] = 1

            # Check neighboring cells in all 6 directions
            if x > 0 and grid[x - 1, y, z] == 0:  # Left
                to_fill.append((x - 1, y, z))
            if x < x_max - 1 and grid[x + 1, y, z] == 0:  # Right
                to_fill.append((x + 1, y, z))
            if y > 0 and grid[x, y - 1, z] == 0:  # Up
                to_fill.append((x, y - 1, z))
            if y < y_max - 1 and grid[x, y + 1, z] == 0:  # Down
                to_fill.append((x, y + 1, z))
            if z > 0 and grid[x, y, z - 1] == 0:  # Forward
                to_fill.append((x, y, z - 1))
            if z < z_max - 1 and grid[x, y, z + 1] == 0:  # Backward
                to_fill.append((x, y, z + 1))

    return grid

start_point = tuple(np.array(voxels_np.shape)//2)
voxels_np_filled = flood_fill_3d(voxels_np, start_point)

print('Done filling the voxels: ', voxels_np_filled.shape)



# ## Simulation

# create FDTD Grid

# In[4]:


grid = fdtd.Grid(
    (4.5e-5, 4.5e-5, 4.5e-5),
    grid_spacing=0.1 * WAVELENGTH,
    permittivity=1.0,
    permeability=1.0,
)

print('Grid Shape: ', grid.shape)


# boundaries

# grid[0, :, :] = fdtd.PeriodicBoundary(name="xbounds")
grid[0:10, :, :] = fdtd.PML(name="pml_xlow")
grid[-20:, :, :] = fdtd.PML(name="pml_xhigh")

# grid[:, 0, :] = fdtd.PeriodicBoundary(name="ybounds")
grid[:, 0:10, :] = fdtd.PML(name="pml_ylow")
grid[:, -10:, :] = fdtd.PML(name="pml_yhigh")

grid[:, :, 0] = fdtd.PeriodicBoundary(name="zbounds")


# sources
xe = grid.shape[0] - 10
ye = grid.shape[1] - 10

# objects

midpoint_y = grid.shape[0]//2
midpoint_x = grid.shape[1]//2
size = 20
grid[ midpoint_y+100, midpoint_x-7, 0] = fdtd.PointSource(
    period=10.0 / BRAIN_FREQ, name="pointsource", amplitude=0.0005,
)


grid[10:-10, 10:-10, :] = fdtd.LearnableAnisotropicObject(permittivity=1.0, is_substrate=False, name="cc_substrate")
conv = torch.nn.Conv2d( 2, 3*3, kernel_size=1, stride=1, padding='same')
print(grid.objects[0].inverse_permittivity.shape)
xl, yl, zl = grid.objects[0].Nx, grid.objects[0].Ny, grid.objects[0].Nz

# Import the image
image = Image.open('head_and_ears_self_round.jpg')
print('minmax: ', np.min(image), np.max(image))
image = image.resize((yl, xl))

import numpy as np
from scipy.ndimage import zoom

new_shape = (xl, yl, zl)  # Desired output shape

# Calculate zoom factors
zoom_factors = (new_shape[0] / voxels_np_filled.shape[0],
                new_shape[1] / voxels_np_filled.shape[1],
                new_shape[2] / voxels_np_filled.shape[2])

voxels_np_filled = zoom(arr, zoom_factors)

print(voxels_np_filled.shape)  # Output: (8, 12, 20)

print(np.asarray(image).shape)
image = np.asarray(image).astype(np.float32) / 255.0
image = np.stack([image]*9, axis=0)
print('minmax: ', np.min(image), np.max(image))
#input('klj')

iimg = torch.from_numpy(image) * 100 + 1
print('The real deal: ', grid.objects[0].sm_activations.shape)
grid.objects[0].seed(1.0/iimg)


# ## Run simulation

# ## Visualization
torch.set_grad_enabled(False)



grid.visualize(z=0, animate=True, norm="log")
#vis_steps = 20
vis_steps = 4
step = 0
for i in range(1000000):
    grid.run(vis_steps, progress_bar=False)
    #grid.visualize(z=0, norm='log', animate=True)
    #grid.visualize(z=0, norm='log', animate=True, objcolor=(0, 0, 0, 0), objedgecolor=(1,1,1,1), plot_both_fields=False, save=True, folder='./sim_frames/', index=i)
    grid.visualize(z=0, norm='log', animate=True, objcolor=(0, 0, 0, 0), objedgecolor=(1,1,1,0), plot_both_fields=False, save=True, folder='./sim_frames/', index=i, clean_img=True)
    plt.show()
    step += vis_steps
    print('On step: ', step)
x = input('type input to end: ')




