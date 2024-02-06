""" The objects to place in the grid.

Objects define all the regions in the grid with a modified update equation,
such as for example regions with anisotropic permittivity etc.

Available Objects:
 - Object
 - AnisotropicObject

"""

## Imports

# typing
from .typing_ import Tensorlike, ListOrSlice

# relative
from .grid import Grid
from .backend import backend as bd
from . import constants as const
import numpy as np
import torch


## Object
class Object:
    """ An object to place in the grid """

    def __init__(self, permittivity: Tensorlike, name: str = None, device=None):
        """
        Args:
            permittivity: permittivity tensor
            name: name of the object (will become available as attribute to the grid)
        """
        self.grid = None
        self.name = name
        self.permittivity = bd.array(permittivity)
        self.device = device

    def _register_grid(
        self, grid: Grid, x: ListOrSlice, y: ListOrSlice, z: ListOrSlice
    ):
        """Register the object to the grid

        Args:
            grid: the grid to register the object into
            x: the x-location of the object in the grid
            y: the y-location of the object in the grid
            z: the z-location of the object in the grid
        """
        self.grid = grid
        self.grid.objects.append(self)
        if self.name is not None:
            if not hasattr(grid, self.name):
                setattr(grid, self.name, self)
            else:
                raise ValueError(
                    f"The grid already has an attribute with name {self.name}"
                )
        self.x = self._handle_slice(x, max_index=self.grid.Nx)
        self.y = self._handle_slice(y, max_index=self.grid.Ny)
        self.z = self._handle_slice(z, max_index=self.grid.Nz)

        self.Nx = abs(self.x.stop - self.x.start)
        self.Ny = abs(self.y.stop - self.y.start)
        self.Nz = abs(self.z.stop - self.z.start)

        # set the permittivity of the object
        if bd.is_array(self.permittivity) and len(self.permittivity.shape) == 3:
            self.permittivity = self.permittivity[:, :, :, None]
        self.inverse_permittivity = (
            bd.ones((self.Nx, self.Ny, self.Nz, 3)) / self.permittivity
        )

        # set the permittivity values of the object at its border to be equal
        # to the grid permittivity. This way, the object is made symmetric.
        if self.Nx > 1:
            self.inverse_permittivity[-1, :, :, 0] = self.grid.inverse_permittivity[
                -1, self.y, self.z, 0
            ]
        if self.Ny > 1:
            self.inverse_permittivity[:, -1, :, 1] = self.grid.inverse_permittivity[
                self.x, -1, self.z, 1
            ]
        if self.Nz > 1:
            self.inverse_permittivity[:, :, -1, 2] = self.grid.inverse_permittivity[
                self.x, self.y, -1, 2
            ]

        self.grid.inverse_permittivity[self.x, self.y, self.z] = 0

    def _handle_slice(self, s: ListOrSlice, max_index: int = None) -> slice:
        if isinstance(s, list):
            if len(s) == 1:
                return slice(s[0], s[0] + 1, None)
            raise IndexError(
                "One can only use slices or single indices to index the grid for an Object"
            )
        if isinstance(s, slice):
            start, stop, step = s.start, s.stop, s.step
            if step is not None and step != 1:
                raise IndexError(
                    "Can only use slices with unit step to index the grid for an Object"
                )
            if start is None:
                start = 0
            if start < 0:
                start = max_index + start
            if stop is None:
                stop = max_index
            if stop < 0:
                stop = max_index + stop
            return slice(start, stop, None)
        raise ValueError("Invalid grid indexing used for object")

    def update_E(self, curl_H):
        """custom update equations for inside the object

        Args:
            curl_H: the curl of magnetic field in the grid.

        """
        loc = (self.x, self.y, self.z)
        self.grid.E[loc] += (
            self.grid.courant_number * self.inverse_permittivity * curl_H[loc]
        )

    def update_H(self, curl_E):
        """custom update equations for inside the object

        Args:
            curl_E: the curl of electric field in the grid.

        """

    def __repr__(self):
        return f"{self.__class__.__name__}(name={repr(self.name)})"

    def __str__(self):
        s = "    " + repr(self) + "\n"

        def _handle_slice(s):
            return (
                str(s)
                .replace("slice(", "")
                .replace(")", "")
                .replace(", ", ":")
                .replace("None", "")
            )

        x = _handle_slice(self.x)
        y = _handle_slice(self.y)
        z = _handle_slice(self.z)
        s += f"        @ x={x}, y={y}, z={z}".replace(":,", ",")
        if s[-1] == ":":
            s = s[:-1]
        return s + "\n"


class AbsorbingObject(Object):
    """ An absorbing object takes conductivity into account """

    def __init__(
        self, permittivity: Tensorlike, conductivity: Tensorlike, name: str = None
    ):
        """
        Args:
            permittivity: permittivity tensor
            conductivity: conductivity tensor (will introduce the loss)
            name: name of the object (will become available as attribute to the grid)
        """
        super().__init__(permittivity, name)
        self.conductivity = conductivity

    def _register_grid(
        self, grid: Grid, x: slice = None, y: slice = None, z: slice = None
    ):
        """Register a grid to the object

        Args:
            grid: the grid to register the object into
            x: the x-location of the object in the grid
            y: the y-location of the object in the grid
            z: the z-location of the object in the grid
        """
        super()._register_grid(grid=grid, x=x, y=y, z=z)

        conductivity = bd.asarray(self.conductivity)
        while conductivity.ndim < self.inverse_permittivity.ndim:
            conductivity = conductivity[..., None]
        self.conductivity = bd.broadcast_to(
            conductivity, self.inverse_permittivity.shape
        )

        self.absorption_factor = (
            0.5
            * self.grid.courant_number
            * self.inverse_permittivity
            * self.conductivity
            * self.grid.grid_spacing
            * const.eta0
        )

    def update_E(self, curl_H):
        """custom update equations for inside the absorbing object

        Args:
            curl_H: the curl of magnetic field in the grid.

        """
        loc = (self.x, self.y, self.z)
        self.grid.E[loc] *= (1 - self.absorption_factor) / (1 + self.absorption_factor)
        self.grid.E[loc] += (
            self.grid.courant_number
            * self.inverse_permittivity
            * curl_H[loc]
            / (1 + self.absorption_factor)
        )

    def update_H(self, curl_E):
        """custom update equations for inside the absorbing object

        Args:
            curl_E: the curl of electric field in the grid.

        """


class AnisotropicObject(Object):
    """ An object with anisotropic permittivity tensor """

    def _register_grid(
        self, grid: Grid, x: slice = None, y: slice = None, z: slice = None
    ):
        """Register a grid to the object

        Args:
            grid: the grid to register the object into
            x: the x-location of the object in the grid
            y: the y-location of the object in the grid
            z: the z-location of the object in the grid
        """
        super()._register_grid(grid=grid, x=x, y=y, z=z)
        eye = bd.zeros((self.Nx * self.Ny * self.Nz, 3, 3))
        eye[:, range(3), range(3)] = 1.0
        self.inverse_permittivity = bd.reshape(
            bd.reshape(self.inverse_permittivity, (-1, 1, 3)) * eye,
            (self.Nx, self.Ny, self.Nz, 3, 3),
        )

    def update_E(self, curl_H):
        """custom update equations for inside the anisotropic object

        Args:
            curl_H: the curl of magnetic field in the grid.

        """
        loc = (self.x, self.y, self.z)
        self.grid.E[loc] += bd.reshape(
            self.grid.courant_number
            * bd.bmm(
                bd.reshape(self.inverse_permittivity, (-1, 3, 3)),
                bd.reshape(curl_H[loc], (-1, 3, 1)),
            ),
            (self.Nx, self.Ny, self.Nz, 3),
        )

    def update_H(self, curl_E):
        """custom update equations for inside the anisotropic object

        Args:
            curl_E: the curl of electric field in the grid.

        """

class LearnableAnisotropicObject(Object):
    """ An object with anisotropic permittivity tensor """

    def __init__(
        self, permittivity: Tensorlike, is_substrate=False, name: str = None, device = None
    ):
        """
        Args:
            permittivity: permittivity tensor
            conductivity: conductivity tensor (will introduce the loss)
            is_substrate: whether or not nonlinearity will be enabled.
            name: name of the object (will become available as attribute to the grid)
        """
        super().__init__(permittivity, name, device)
        # Takes the field energies (E and  H) as input and outputs modifiers for the permitivity.
        self.nonlin_conv = torch.nn.Conv2d( 2, 3*3, kernel_size=1, stride=1, padding='same', device=self.device)
        self.is_substrate = is_substrate
        self.sm_activations = None
        self.min_nonlin_mod = 0.2

    def _register_grid(
        self, grid: Grid, x: slice = None, y: slice = None, z: slice = None
    ):
        """Register a grid to the object

        Args:
            grid: the grid to register the object into
            x: the x-location of the object in the grid
            y: the y-location of the object in the grid
            z: the z-location of the object in the grid
        """
        super()._register_grid(grid=grid, x=x, y=y, z=z)
        eye = bd.zeros((self.Nx * self.Ny * self.Nz, 3, 3))
        eye[:, range(3), range(3)] = 1.0
        self.inverse_permittivity = bd.reshape(
            bd.reshape(self.inverse_permittivity, (-1, 1, 3)) * eye,
            (self.Nx, self.Ny, self.Nz, 3, 3),
        )
        self.inverse_permittivity.requires_grad = True
        self.inverse_permittivity.retain_grad()

    def seed(self, sm_activations):
        '''
        sm_activations: (permitivity_shape, H, W)
        '''
        sm_activations = torch.reshape(sm_activations, (1, 3, 3, self.Nx, self.Ny))
        self.sm_activations = torch.permute(sm_activations, (3, 4, 0, 1, 2))

    def update_E(self, curl_H):
        """custom update equations for inside the anisotropic object

        Args:
            curl_H: the curl of magnetic field in the grid.

        """
        loc = (self.x, self.y, self.z)
        # Nonlinear modifier calculation:
        # Calculate energy of E and H fields.
        if(self.is_substrate):
            with torch.no_grad():
                grid_energy_E = bd.sum(self.grid.E[self.x, self.y, self.z] ** 2, -1)
                grid_energy_H = bd.sum(self.grid.H[self.x, self.y, self.z] ** 2, -1)
                grid_energy = torch.stack([grid_energy_E, grid_energy_H], dim=0)
                grid_energy = torch.permute(grid_energy, (3, 0, 1, 2))
            nonlin_modifier = (1.0-self.min_nonlin_mod)*torch.sigmoid(self.nonlin_conv(grid_energy)) + self.min_nonlin_mod
            s = (1, 3, 3) + tuple(nonlin_modifier.shape[-2:])
            nonlin_modifier = torch.reshape(nonlin_modifier, s)
            nonlin_modifier = torch.permute(nonlin_modifier, (3, 4, 0, 1, 2))
            # Add the substrate modifier activations to nonlin_modifier
            if(self.sm_activations is None):
                print('Error: substrate object must be seeded')
                sys.exit()
            nonlin_modifier = nonlin_modifier * self.sm_activations
        elif(self.sm_activations is not None):
            nonlin_modifier = self.sm_activations
        else:
            nonlin_modifier = torch.tensor([1.0], device=self.device)
        
        
        self.grid.E[loc] += bd.reshape(
            self.grid.courant_number
            * bd.bmm(
                bd.reshape(self.inverse_permittivity * nonlin_modifier, (-1, 3, 3)),
                bd.reshape(curl_H[loc], (-1, 3, 1)),
            ),
            (self.Nx, self.Ny, self.Nz, 3),
        )

    def update_H(self, curl_E):
        """custom update equations for inside the anisotropic object

        Args:
            curl_E: the curl of electric field in the grid.

        """
