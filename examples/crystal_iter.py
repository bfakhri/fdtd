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
fdtd.set_backend("torch")


# ## Simulation Constants
WAVELENGTH = 1550e-9 # For resolution purposes.
SPEED_LIGHT: float = 299_792_458.0  # [m/s] speed of light

# Frequency of the brainwaves.
BRAIN_FREQ_START =   340000000000.0 # Hz
BRAIN_FREQ_END =   2.2*34000000000000.0 # Hz

num_samples = 400

for freq_idx, freq in enumerate(np.linspace(BRAIN_FREQ_START, BRAIN_FREQ_END, num_samples)):
    print('Running at freq: {0}'.format(freq))
    grid = fdtd.Grid(
        (4.5e-5, 4.5e-5, 1),
        grid_spacing=0.1 * WAVELENGTH,
        permittivity=1.0,
        permeability=1.0,
    )


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

    grid[ 10:ye, 10, 0] = fdtd.LineSource(
        period=1.0 / freq, name="linesource2"
    )

    grid[40:-40, 40:-40, :] = fdtd.LearnableAnisotropicObject(permittivity=1.0, is_substrate=False, name="cc_substrate")
    yl, xl = grid.objects[0].Ny, grid.objects[0].Nx

    # Import the image
    img_x_freq = 8
    image = Image.open('sinusoidal_grating_f{0}.jpg'.format(img_x_freq))

    # Resize to fit the grid object
    image = image.resize((yl, xl))

    # Calculate the image's wavelength in the grid space
    img_wavelength = xl * grid.grid_spacing / img_x_freq
    print('Object dims: ', yl, xl, grid.grid_spacing, img_wavelength)
    print('Params: {0}\t{1:.2f}'.format(freq_idx, (SPEED_LIGHT/freq) / img_wavelength))

    image = np.asarray(image).astype(np.float32) / 255.0
    image = np.stack([image]*9, axis=0)

    #iimg = torch.from_numpy(image) * 100 + 1
    iimg = torch.from_numpy(image) * 20 + 1
    grid.objects[0].seed(1.0/iimg)


    # ## Visualization
    torch.set_grad_enabled(False)

    grid.visualize(z=0, animate=True, norm="log")
    vis_steps = 1000
    grid.run(vis_steps, progress_bar=False)
    grid.visualize(z=0, norm='log', animate=True, objcolor=(0, 0, 0, 0), objedgecolor=(1,1,1,0), plot_both_fields=False, save=True, folder='./sim_frames_f{0}/'.format(img_x_freq), index=freq_idx, clean_img=True)
    plt.show()
    #x = input('type input to end: ')

    #if(freq_idx > 90):
    #    for i in range(10):
    #        grid.visualize(z=0, animate=True, norm="log")
    #        vis_steps = 100
    #        grid.run(vis_steps, progress_bar=False)
    #        grid.visualize(z=0, norm='log', animate=True, objcolor=(0, 0, 0, 0), objedgecolor=(1,1,1,0), plot_both_fields=False, save=False, folder='./sim_frames/', index=freq_idx, clean_img=True)
    #        plt.show()



