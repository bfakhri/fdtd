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
import random
import torch
import torch.optim as optim
import torchvision
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import scipy
from autoencoder import AutoEncoder
import argparse
from os import listdir
from os.path import isfile, join
import util

parser = argparse.ArgumentParser(description='Process args.')
parser.add_argument('-f', '--load-file', type=str, default=None,
                    help='File to load params from before training starts. Overrides --load-step.')
parser.add_argument('-l', '--load-step', type=str, default='0',
                    help='Where to start training. If latest, will start at the latest checkpoint.')
parser.add_argument('-s', '--save-steps', type=int, default='1000',
                    help='How often to save the model.')
parser.add_argument('-c', '--coverage-ratio', type=float, default=1.0,
                    help='How much distance a wave can cover as a proportion of the diagonal length of the sim.')
parser.add_argument('-m', '--max-steps', type=int, default='1000000000000000',
                    help='How many steps to train.')
parser.add_argument('-d', '--dry-run', type=bool, default=False,
                    help='If true, does not save model checkpoint.')
parser.add_argument('-rog', '--reset-grid-optim', default=False, action='store_true',
                    help='If true, loads completely new params for the grid and optimizer.')
parser.add_argument('-is', '--image-size', type=int, default=40,
                    help='Size of each side of the image. Determines grid size.')
parser.add_argument('-sc', '--image-scaler', type=int, default=1,
                    help='How much to scale the entire simulation by (changes the dimensions of the model).')
parser.add_argument('-oc', '--old-scaler', type=int, default=1,
                    help='If the loaded file was scaled, that scaler value.')
parser.add_argument('-bem', '--bypass-em', default=False, action='store_true',
                    help='If set, will disable the EM component of the model.')
parser.add_argument('-gray', '--grayscale', default=False, action='store_true',
                    help='If set, will force the input and output images to be grayscale.')
parser.add_argument('-thw', '--target-half-way', default=False, action='store_true',
                    help='If set, will only produce loss for the timestep halfway up the coverage ratio.')
parser.add_argument('-sds', '--source-down-scaler', type=int, default=1, 
                    help='How much to stride sources in the cortical substrate.')
parser.add_argument('-sgd', '--use-sgd', default=False, action='store_true',
                    help='If set, will switch to SGD instead of Adam. Useful for finetuning.')
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

# Setup tensorboard
tb_parent_dir = './runs/'
repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha
#head = repo.head
local_branch = repo.active_branch.name
run_dir = 'test_outputs'
print('TB Log Directory is: ', tb_parent_dir + run_dir)
writer = SummaryWriter(log_dir=tb_parent_dir + run_dir)

# Setup model saving
model_parent_dir = './model_checkpoints/'
model_checkpoint_dir = model_parent_dir + local_branch + '/'
path = Path(model_checkpoint_dir)
path.mkdir(parents=True, exist_ok=True)

# ## Set Backend
backend_name = "torch"
#backend_name = "torch.float32"
#backend_name = "torch.float16"
#backend_name = "torch.cuda.float32"
#backend_name = "torch.cuda.float64"
fdtd.set_backend(backend_name)
if(backend_name.startswith("torch.cuda")):
    device = "cuda"
else:
    device = "cpu"

image_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.RandomVerticalFlip(p=0.5),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    #torchvision.transforms.RandomRotation(degrees=[0, 360], expand=True),
    util.RandomRot90(),
    torchvision.transforms.ColorJitter(brightness=0.5, hue=0.3),
    torchvision.transforms.RandomInvert(p=0.5),
    torchvision.transforms.Resize((args.image_size*args.image_scaler, args.image_size*args.image_scaler))])
train_dataset = torchvision.datasets.Flowers102('flowers102/', 
                                           split='train',
                                           download=True,
                                           transform=image_transform)
# Note - turn SHUFFLE back to TRUE for training on multiple images.
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=1, 
                                           shuffle=True)


print('Grayscale: ', args.grayscale)
#sample = util.get_sample_img(train_loader, device, color=not args.grayscale)
img_file = img_paths[6]
img = Image.open(img_file)
img = image_transform(img)[None, ...]
if(args.grayscale):
    img = torchvision.transforms.Grayscale()(img)[None, 0, ...]
else:
    img = torchvision.transforms.Grayscale()(img)
print('Image shape: ', img.shape)
ih, iw = tuple(img.shape[2:4])
print('ih, iw: ', ih, iw)
testing = util.get_sample_img(train_loader, device, color=not args.grayscale)
print('testing: ', testing.shape)

# Physics constants
WAVELENGTH = 1550e-9 # meters
SPEED_LIGHT: float = 299_792_458.0  # [m/s] speed of light
GRID_SPACING = 0.1 * WAVELENGTH # meters



# Size of grid boundary layer
bw = 10*args.image_scaler
# Create FDTD Grid
grid_h, grid_w = (ih+bw*2, iw+bw*2)
print('Grid height and width: ', grid_h, grid_w)
# Boundaries with width bw
grid = fdtd.Grid(
    (grid_h, grid_w, 1),
    grid_spacing=GRID_SPACING,
    permittivity=1.0,
    permeability=1.0,
)

# Calculate how long it takes a wave to cross the entire grid.
grid_diag_cells = math.sqrt(grid_h**2 + grid_w**2)
grid_diag_len = grid_diag_cells * GRID_SPACING
grid_diag_steps = int(grid_diag_len/SPEED_LIGHT/grid.time_step)+1
print('Time steps to cover entire grid: ', grid_diag_steps)
# The number of steps is based on the coverage ratio.
if(args.target_half_way):
    em_steps = int(grid_diag_steps*args.coverage_ratio/2)
else:
    em_steps = int(grid_diag_steps*args.coverage_ratio)
print('Time steps the grid will run for: ', em_steps)


# Create learnable objects at the boundaries
grid[  0: bw, :, :] = fdtd.LearnableAnisotropicObject(permittivity=2.5, name="xlow", device=device)
grid[-bw:   , :, :] = fdtd.LearnableAnisotropicObject(permittivity=2.5, name="xhigh", device=device)
grid[:,   0:bw, :] = fdtd.LearnableAnisotropicObject(permittivity=2.5, name="ylow", device=device)
grid[:, -bw:  , :] = fdtd.LearnableAnisotropicObject(permittivity=2.5, name="yhigh", device=device)
grid[:, :, 0] = fdtd.PeriodicBoundary(name="zbounds")

# Creat the cortical column sources
grid[bw:bw+ih,bw:bw+iw,0] = fdtd.CorticalColumnPlaneSource(
    period = WAVELENGTH / SPEED_LIGHT,
    polarization = 'x', # BS value, polarization is not used.
    name = 'cc',
    source_stride = args.source_down_scaler,
)

# Object defining the cortical column substrate 
grid[bw:-bw, bw:-bw, :] = fdtd.LearnableAnisotropicObject(permittivity=2.5, is_substrate=True, name="cc_substrate", device=device)
# List all model checkpoints
checkpoints = [f for f in listdir(model_checkpoint_dir) if(isfile(join(model_checkpoint_dir, f)) and f.endswith('.pt'))]

torch.autograd.set_detect_anomaly(True)
# The weights for the reconstruction loss at each em time step. 
loss_step_weights = torch.ones(em_steps, device=device)/em_steps
#loss_step_weights = torch.nn.Parameter(torch.reshape(loss_step_weights, (-1, 1, 1, 1, 1)))
loss_step_weights.requires_grad = True
softmax = torch.nn.Softmax(dim=0)

# Initialize the model and grid with default params.
if(args.grayscale):
    chans = 1
else:
    chans = 3
model = AutoEncoder(num_em_steps=em_steps, grid=grid, input_chans=chans, output_chans=chans, source_stride=args.source_down_scaler, bypass_em=args.bypass_em).to(device)
print('All grid objects: ', [obj.name for obj in grid.objects])
grid_params_to_learn = []
grid_params_to_learn += [util.get_object_by_name(grid, 'xlow').inverse_permittivity]
grid_params_to_learn += [util.get_object_by_name(grid, 'xhigh').inverse_permittivity]
grid_params_to_learn += [util.get_object_by_name(grid, 'ylow').inverse_permittivity]
grid_params_to_learn += [util.get_object_by_name(grid, 'yhigh').inverse_permittivity]
grid_params_to_learn += [util.get_object_by_name(grid, 'cc_substrate').inverse_permittivity]
# Nonlinearity weights for the substrate. 
grid_params_to_learn += [util.get_object_by_name(grid, 'cc_substrate').nonlin_conv.weight]
grid_params_to_learn += [util.get_object_by_name(grid, 'cc_substrate').nonlin_conv.bias]
# Nonlinearity weights for the cortical columns. 
grid_params_to_learn += [util.get_source_by_name(grid, 'cc').nonlin_conv.weight]
grid_params_to_learn += [util.get_source_by_name(grid, 'cc').nonlin_conv.bias]
# The weights for the loss.
grid_params_to_learn += [loss_step_weights]

# Load saved params for model and optimizer.
checkpoint_steps = [int(cf.split('_')[-1].split('.')[0]) for cf in checkpoints]
if(args.load_file is not None):
    start_step = int(args.load_file.split('/')[-1].split('_')[-1].split('.')[0])
    print('Loading model {0}. Starting at step {1}.'.format(args.load_file, start_step))
    optimizer_path = args.load_file.rsplit('.', 1)[0] + '.opt'
    grid_path = args.load_file.rsplit('.', 1)[0] + '.grd'
    model.load_state_dict(torch.load(args.load_file))
else:
    if(args.load_step == 'latest'):
        if(len(checkpoint_steps) > 0):
            latest_idx = np.argmax(checkpoint_steps)
            start_step = checkpoint_steps[latest_idx]
            model_dict_path = model_checkpoint_dir + checkpoints[latest_idx]
            optimizer_path = model_dict_path.rsplit('.', 1)[0] + '.opt'
            grid_path = model_dict_path.rsplit('.', 1)[0] + '.grd'
            print('Loading model {0} with optimizer {1} and grid {2}.'.format(model_dict_path, optimizer_path, grid_path))
            model.load_state_dict(torch.load(model_dict_path))
        else:
            start_step = 0
    elif(int(args.load_step) != 0):
        if(int(args.load_step) not in checkpoint_steps):
            print('Checkpoint {0} not found in {1}'.format(args.load_step, model_checkpoint_dir))
            sys.exit()
        start_step = int(args.load_step)
        model_idx = np.where(np.array(checkpoint_steps) == start_step)[0][0]
        model_dict_path = model_checkpoint_dir + checkpoints[model_idx]
        optimizer_path = model_dict_path.rsplit('.', 1)[0] + '.opt'
        grid_path = model_dict_path.rsplit('.', 1)[0] + '.grd'
        print('Loading model {0} with optimizer {1} and grid {2}.'.format(model_dict_path, optimizer_path, grid_path))
        model.load_state_dict(torch.load(model_dict_path))
    else:
        print('Starting model at step 0')
        start_step = 0
        optimizer_path = None
        grid_path = None

reset_optimizer = False
if((grid_path is not None) and (not args.reset_grid_optim)):
    print('Loading grid params...')
    with torch.no_grad():
        load_grid_params_to_learn = torch.load(grid_path)
        for idx, tensor in enumerate(load_grid_params_to_learn):
            if(tensor.shape == grid_params_to_learn[idx][...].shape):
                grid_params_to_learn[idx][...] = tensor[...]
            else:
                if(idx == len(load_grid_params_to_learn) - 1):
                    tensor = torch.squeeze(tensor)
                # Interpolate the thing....
                print('INFO: Shapes are mismatched: {0} vs {1}'.format(tensor[...].shape, grid_params_to_learn[idx][...].shape))
                
                # If this is a grid param, expand it over the spatial dims.
                if(len(tensor.shape) > 1):
                    reps = np.ones(len(tensor.shape), dtype=int)
                    for i in range(len(reps)):
                        reps[i] = int(args.image_scaler / args.old_scaler)
                        if(i >= 1):
                            break

                    tensor_np_interp = scipy.ndimage.zoom(tensor.detach().numpy(), reps, order=1)
                    grid_params_to_learn[idx][...] = torch.from_numpy(tensor_np_interp)
                    print('INFO: Grid object scaled to shape: ', grid_params_to_learn[idx][...].shape)
                # If this is the loss step weights, scale it linearly to fit the new size.
                else:
                    tensor_interp = torch.nn.functional.interpolate(tensor[None, None, ...], grid_params_to_learn[idx][...].shape, mode='linear')
                    grid_params_to_learn[idx][...] = tensor_interp
                    print('INFO: EM Step loss object scaled to shape: ', grid_params_to_learn[idx][...].shape)
                    print('EM Steps: ', em_steps)
                    print('Loss step weights: ', loss_step_weights.shape)

                # Since parameter shapes have changed, the optimizer weights are obsolete.
                reset_optimizer = True

# Combine grid and model params and register them with the optimizer.
params_to_learn = [*model.parameters()] + grid_params_to_learn

mse = torch.nn.MSELoss(reduce=False)
loss_fn = torch.nn.MSELoss()

grid.H.requires_grad = True
grid.H.retain_grad()
grid.E.requires_grad = True
grid.E.retain_grad()

# For timing steps
stopwatch = time.time()

# Generate a new image
#img = util.get_sample_img(train_loader, device, color=not args.grayscale)
#img = torch.zeros_like(img)
#img[:,:,0:40,0:40] = 1
#print(img.max(), img.min(), img.mean())
#sys.exit()

# Reset grid and optimizer
grid.reset()

loss_list = []
E_energy_jump_list = []
H_energy_jump_list = []
energy_loss = 0
last_energy_E = 0
last_energy_H = 0 
# Get sample from training data
em_step_loss_weight_dist = softmax(loss_step_weights)
argmax_step = torch.argmax(torch.squeeze(loss_step_weights))
for em_step, (img_hat_em, em_field) in enumerate(model(img, summary_writer=writer, train_step=0)):
    print('On em step: ', em_step, em_steps)
    # Calculate clipped metabolic loss. This keeps energy addition small.
    energy_E, energy_H = util.calculate_em_energy(em_field)
    # Normalize the energy by the size of the field.
    energy_E, energy_H = (energy_E/(em_field.numel()*em_steps), energy_H/(em_field.numel()*em_steps))
    energy_jump_E = energy_E - last_energy_E
    energy_jump_H = energy_H - last_energy_H
    E_energy_jump_list += [energy_jump_E]
    H_energy_jump_list += [energy_jump_H]
    energy_E_loss = torch.relu(energy_jump_E - 0.1)
    energy_H_loss = torch.relu(energy_jump_H - 0.1)
    energy_loss += energy_E_loss + energy_H_loss
    last_energy_E = energy_E
    last_energy_H = energy_H
    # Only add the last step to the loss if we half target_half_way enabled.
    if(args.target_half_way and em_step == em_steps-1):
        loss_list += [loss_fn(img_hat_em[None, ...], img)]
    else:
        loss_list += [loss_fn(img_hat_em[None, ...], img)]

    # Save the em field and energy at the target step.
    e_field_img = em_field[0:3,...]
    h_field_img = em_field[3:6,...]
    img_hat_em_save = img_hat_em
    e_field_energy_save = energy_E
    h_field_energy_save = energy_H

    loss_per_step = torch.stack(loss_list)
    E_energy_jump_per_step = torch.stack(E_energy_jump_list)
    H_energy_jump_per_step = torch.stack(H_energy_jump_list)

    # Add the argmaxxed images to tensorboard
    if(args.grayscale):
        img =  img.expand(-1, 3, -1, -1)
        img_hat_em_save =  img_hat_em_save.expand(3, -1, -1)
    img_grid = torchvision.utils.make_grid([img[0,...], img_hat_em_save,
        util.norm_img_by_chan(e_field_img), 
        util.norm_img_by_chan(h_field_img)])
    writer.add_image('sample', img_grid, em_step)
    save_image(img_grid, './images/img_idx{0}.png'.format(str(em_step).zfill(12)))

writer.close()
