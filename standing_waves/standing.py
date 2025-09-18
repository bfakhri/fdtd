import sys
sys.path.append('/home/bij/Projects/fdtd/')
import matplotlib.pyplot as plt

import fdtd
fdtd.set_backend("torch")

MIN_WAVELENGTH = 1550e-9
SPEED_LIGHT: float = 299_792_458.0  # [m/s] speed of light

grid_y_cells = 60
grid_x_cells = 40
grid_space_ratio = 0.1
grid_spacing = grid_space_ratio * MIN_WAVELENGTH
grid_y_len = grid_y_cells * grid_spacing

# create FDTD Grid
grid = fdtd.Grid(
    (grid_y_cells, grid_x_cells, 1),  # 2D grid
    grid_spacing=grid_spacing,
    permittivity=1,  # same as object
)

wave_len = grid_y_len / 3
period = wave_len / SPEED_LIGHT

# sources
#grid[30, :] = fdtd.LineSource(period=WAVELENGTH / SPEED_LIGHT, name="source")
#grid[10, :] = fdtd.LineSource(period= 1.50 * WAVELENGTH / SPEED_LIGHT, name="source")
grid[10, :] = fdtd.LineSource(period=period, name="source")
#grid[30, :] = fdtd.LineSource(period= 60 / 30 * WAVELENGTH / SPEED_LIGHT, name="source")

# x boundaries
# grid[0, :, :] = fdtd.PeriodicBoundary(name="xbounds")
#grid[0:10, :, :] = fdtd.PML(name="pml_xlow")
grid[0:10, :, :] = fdtd.PML(name="pml_xlow")
#grid[0:10:, :, :] = fdtd.Object(permittivity=1000, name="hard_boundary1")
#grid[-10:, :, :] = fdtd.PML(name="pml_xhigh")
#grid[0:10, :, :] = fdtd.Object(permittivity=100, name="pml_xlow")
grid[-10:, :, :] = fdtd.Object(permittivity=1000, name="hard_boundary2")

# y boundaries
# grid[:, 0, :] = fdtd.PeriodicBoundary(name="ybounds")
grid[:, 0:10, :] = fdtd.PML(name="pml_ylow")
grid[:, -10:, :] = fdtd.PML(name="pml_yhigh")

for i in range(1000):
    grid.run(1, progress_bar=False)
    grid.visualize(z=0, animate=True, norm="log", plot_grid_avg=False, plot_both_fields=True)
    #grid.visualize(z=0, animate=True, plot_grid_avg=True)
    #grid.visualize(z=0, animate=True, norm="log", plot_grid_pow_avg=True)
