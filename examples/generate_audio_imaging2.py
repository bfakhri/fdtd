import sys
sys.path.append('/home/bij/Projects/fdtd/')
import fdtd
import fdtd.backend as bd
import matplotlib.pyplot as plt
import numpy as np


# ## Set Backend
#fdtd.set_backend("numpy")
fdtd.set_backend("torch")


# ## Constants
WAVELENGTH = 1550e-9
WAVELENGTH2 = 4*1550e-9
SPEED_LIGHT: float = 299_792_458.0  # [m/s] speed of light

def NormalizeEnergy(energy, width=3):
    mean = np.mean(energy.flatten())
    std = np.std(energy.flatten())
    h_cutoff = mean + std*width
    energy_normed = (np.clip(energy, 0, h_cutoff))/(h_cutoff)
    return energy_normed

#for shift in range(0, 143, 1):
period_steps = 90
major_step = 0
directions = [(-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0)]
l_delay = period_steps//2
r_delay = period_steps//2

for l_dir, r_dir in directions:
    for shift in range(0, period_steps//2, 1):
        grid = fdtd.Grid(
            (3.5e-5, 3.5e-5, 1),
            #(1.5e-5, 1.5e-5, 1),
            grid_spacing=0.1 * WAVELENGTH,
            permittivity=1.0,
            permeability=1.0,
        )
        grid[0:10, :, :] = fdtd.PML(name="pml_xlow")
        grid[-10:, :, :] = fdtd.PML(name="pml_xhigh")
        grid[:, 0:10, :] = fdtd.PML(name="pml_ylow")
        grid[:, -10:, :] = fdtd.PML(name="pml_yhigh")
        grid[:, :, 0] = fdtd.PeriodicBoundary(name="zbounds")

        y_mid, x_mid = grid.shape[0]//2, grid.shape[1]//2

        grid[y_mid, 25, 0] = fdtd.PointSource(
            period=WAVELENGTH2 / SPEED_LIGHT, name="left_source", delay=l_delay,
        )
        grid[y_mid, -25, 0] = fdtd.PointSource(
            period=WAVELENGTH2 / SPEED_LIGHT, name="right_source", delay=r_delay,
        )

        l_delay += l_dir
        r_delay += r_dir
        
        print('Step: {0}\tRunning with delays ({1}, {2}) and source period {3}'.format(major_step, l_delay, r_delay, grid.sources[0].period))
        grid.run(1000, progress_bar=False)
        E_energy_avg = bd.numpy(bd.sum(grid.E_avg[:, :, 0, :]**2, axis=-1))
        E_energy_avg_normed = NormalizeEnergy(E_energy_avg)
        plt.axis('off')
        plt.imshow(E_energy_avg_normed)
        plt.savefig('ears/avg_energy_{0:04d}.png'.format(major_step), bbox_inches='tight')
        major_step += 1
