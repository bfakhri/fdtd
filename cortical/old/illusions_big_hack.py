#!/usr/bin/env python
import git
import sys
import datetime
from pathlib import Path
sys.path.append('/home/bij/Projects/fdtd/')
import math
import time
import fdtd
import fdtd.backend as bd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torchvision
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from  torch.nn.modules.upsampling import Upsample
from autoencoder import AutoEncoder
import argparse
from os import listdir
from os.path import isfile, join
import cv2
from PIL import Image


parser = argparse.ArgumentParser(description='Process args.')
parser.add_argument('-f', '--load-file', type=str, default=None,
                    help='File to load params from before training starts. Overrides --load-step.')
args = parser.parse_args()

def get_sorted_paths(directory_list, target_ext='.png'):
    path_list = []
    for directory in directory_list:
        paths = [join(directory,f) for f in listdir(directory) if isfile(join(directory, f)) and f.endswith(target_ext)]
        print(f'Found {len(paths)} files in {directory}')
        path_list += paths
    path_list.sort()
    return path_list

img_paths = get_sorted_paths(['./optical_illusions/'])

def norm_img_by_chan(img):
    '''
    Puts each channel into the range [0,1].
    Expects input to be in CHW config.
    '''
    img_flat = torch.reshape(img, (3, -1))
    chan_maxes, _ = torch.max(img_flat, dim=-1, keepdims=True) 
    chan_mins, _  = torch.min(img_flat, dim=-1, keepdims=True) 
    chans_dynamic_range = chan_maxes - chan_mins
    normed_img = (img - chan_mins[...,None])/(chans_dynamic_range[...,None])
    return normed_img 

def get_object_by_name(grid, name):
    for obj in grid.objects:
        if(obj.name == name):
            return obj
def get_source_by_name(grid, name):
    for src in grid.sources:
        if(src.name == name):
            return src 


# Setup tensorboard
tb_parent_dir = './runs/'
repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha
#head = repo.head
local_branch = repo.active_branch.name
run_dir = local_branch + '/' + sha[-3:] + '/' +  datetime.datetime.now().isoformat(timespec='seconds') + '/'
print('TB Log Directory is: ', tb_parent_dir + run_dir)
#writer = SummaryWriter(log_dir=tb_parent_dir + run_dir)

# Setup model saving
model_parent_dir = './model_checkpoints/'
model_checkpoint_dir = model_parent_dir + local_branch + '/'
path = Path(model_checkpoint_dir)
path.mkdir(parents=True, exist_ok=True)

# ## Set Backend
backend_name = "torch"
#backend_name = "torch.cuda.float32"
#backend_name = "torch.cuda.float64"
fdtd.set_backend(backend_name)
if(backend_name.startswith("torch.cuda")):
    device = "cuda"
else:
    device = "cpu"

ih, iw = (600, 600)
image_transform = torchvision.transforms.Compose([torchvision.transforms.Resize((ih, iw)),
                               torchvision.transforms.ToTensor()])


# Physics constants
WAVELENGTH = 1550e-9 # meters
SPEED_LIGHT: float = 299_792_458.0  # [m/s] speed of light
GRID_SPACING = 0.1 * WAVELENGTH # meters


# Size of grid boundary layer
bw = 10
# Create FDTD Grid
grid_h, grid_w = (ih+bw*2, iw+bw*2)
# Boundaries with width bw
grid = fdtd.Grid(
    (grid_h, grid_w, 1),
    grid_spacing=GRID_SPACING,
    permittivity=1.0,
    permeability=1.0,
)

# Slow down the grid!
time_scaler = 1
print('Grid time step: ', grid.time_step)
grid.time_step = grid.time_step / time_scaler
print('Grid time step: ', grid.time_step)

# Calculate how long it takes a wave to cross the entire grid.
grid_diag_cells = math.sqrt(grid_h**2 + grid_w**2)
grid_diag_len = grid_diag_cells * GRID_SPACING
grid_diag_steps = int(grid_diag_len/SPEED_LIGHT/grid.time_step)+1
print('Time Steps to Cover Entire Grid: ', grid_diag_steps)


# Create learnable objects at the boundaries
grid[  0: bw, :, :] = fdtd.LearnableAnisotropicObject(permittivity=2.5, name="xlow")
grid[-bw:   , :, :] = fdtd.LearnableAnisotropicObject(permittivity=2.5, name="xhigh")
grid[:,   0:bw, :] = fdtd.LearnableAnisotropicObject(permittivity=2.5, name="ylow")
grid[:, -bw:  , :] = fdtd.LearnableAnisotropicObject(permittivity=2.5, name="yhigh")
grid[:, :, 0] = fdtd.PeriodicBoundary(name="zbounds")

# Creat the cortical column sources
grid[bw:bw+ih,bw:bw+iw,0] = fdtd.CorticalColumnPlaneSource(
    period = WAVELENGTH / SPEED_LIGHT,
    polarization = 'x', # BS value, polarization is not used.
    name='cc'
)

# Object defining the cortical column substrate 
grid[bw:-bw, bw:-bw, :] = fdtd.LearnableAnisotropicObject(permittivity=2.5, name="cc_substrate")

# List all model checkpoints
checkpoints = [f for f in listdir(model_checkpoint_dir) if(isfile(join(model_checkpoint_dir, f)) and f.endswith('.pt'))]
# Get the latest checkpoint
model = AutoEncoder(grid=grid, num_em_steps=283, input_chans=3, output_chans=3).to(device)
if(args.load_file is not None):
    start_step = int(args.load_file.split('/')[-1].split('_')[-1].split('.')[0])
    print('Loading model {0}. Starting at step {1}.'.format(args.load_file, start_step))
    grid_path = args.load_file.rsplit('.', 1)[0] + '.grd'
    model.load_state_dict(torch.load(args.load_file))
else:
    print('No model provided, exiting...')
    sys.exit()

print('All grid objects: ', [obj.name for obj in grid.objects])
grid_params_to_learn = []
grid_params_to_learn += [get_object_by_name(grid, 'xlow').inverse_permittivity]
grid_params_to_learn += [get_object_by_name(grid, 'xhigh').inverse_permittivity]
grid_params_to_learn += [get_object_by_name(grid, 'ylow').inverse_permittivity]
grid_params_to_learn += [get_object_by_name(grid, 'yhigh').inverse_permittivity]
grid_params_to_learn += [get_object_by_name(grid, 'cc_substrate').inverse_permittivity]
# Nonlinearity weights for the substrate. 
grid_params_to_learn += [get_object_by_name(grid, 'cc_substrate').nonlin_conv.weight]
grid_params_to_learn += [get_object_by_name(grid, 'cc_substrate').nonlin_conv.bias]
# Nonlinearity weights for the cortical columns. 
grid_params_to_learn += [get_source_by_name(grid, 'cc').nonlin_conv.weight]
grid_params_to_learn += [get_source_by_name(grid, 'cc').nonlin_conv.bias]

# Load the grid params from file.
if(grid_path is not None):
    print('Loading grid params...')
    with torch.no_grad():
        load_grid_params_to_learn = torch.load(grid_path)
        for idx, tensor in enumerate(load_grid_params_to_learn):
            try:
                grid_params_to_learn[idx][...] = tensor[...]
            except:
                print('COULD NOT LOAD PARAM: ', idx, ' AS IS. RESIZING.')
                resizer = torchvision.transforms.Resize(tuple(grid_params_to_learn[idx].shape[0:2]), interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
                print('1: ', tensor.shape)
                tensor_tp = torch.permute(tensor[...], (2,3,4,0,1))
                print('2: ', tensor_tp.shape)
                tensor_tp = resizer(tensor_tp)
                print('3: ', tensor_tp.shape)
                tensor_tp = torch.permute(tensor_tp[...], (3,4,0,1,2))
                print('4: ', tensor_tp.shape)
                grid_params_to_learn[idx][...] = tensor_tp
                print('5: ', grid_params_to_learn[idx].shape)

# Optimizer params
mse = torch.nn.MSELoss(reduce=False)
loss_fn = torch.nn.MSELoss()

em_steps = model.num_em_steps


powers = np.exp(np.linspace(0.1, 3.0, 12)) - 1.0

with torch.inference_mode():
    # Add images to tensorboard
    for img_idx, img_file in enumerate(img_paths):
        #img = cv2.imread(img_file)
        img = Image.open(img_file)
        img = image_transform(img)[None, ...]
        print(img.shape)
        # Reset grid
        grid.reset()

        print('\nGenerating image for illusion {0}'.format(img_idx))
        for em_step, (img_hat_em, em_field) in enumerate(model(img, amp_scaler=1.0)):
            print('Generating image for illusion {0} and em step {1}'.format(img_idx, em_step), end='\r')
            # Process outputs
            e_field_img = em_field[0:3,...]
            h_field_img = em_field[3:6,...]
            # Write to TB
            img_grid = torchvision.utils.make_grid([img[0,...], img_hat_em,
                norm_img_by_chan(e_field_img), 
                norm_img_by_chan(h_field_img)])
            #img_grid = torchvision.transforms.functional.resize(img_grid, size=(img_grid.shape[1] * 4, img_grid.shape[2] * 4), interpolation=torchvision.transforms.InterpolationMode.NEAREST)

            #writer.add_image('sample', img_grid, em_step)
            save_image(img_grid, './images/img_p{0}_idx{1}.png'.format('{0:06.3f}'.format(img_idx), str(em_step).zfill(12)))

