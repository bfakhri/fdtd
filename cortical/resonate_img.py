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
from torch.utils.tensorboard import SummaryWriter
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


# Setup tensorboard
tb_parent_dir = './runs/'
repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha
#head = repo.head
local_branch = repo.active_branch.name
run_dir = local_branch + '/' + sha[-3:] + '/' +  datetime.datetime.now().isoformat(timespec='seconds') + '/'
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
#train_dataset = torchvision.datasets.Flowers102('flowers102/', 
#                                           split='train',
#                                           download=True,
#                                           transform=image_transform)
train_dataset = torchvision.datasets.CelebA('CelebA/', 
                                           split='train',
                                           download=False,
                                           transform=image_transform)
# Note - turn SHUFFLE back to TRUE for training on multiple images.
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=1, 
                                           shuffle=True)


print('Grayscale: ', args.grayscale)
sample = util.get_sample_img(train_loader, device, color=not args.grayscale)
print('Image shape: ', sample.shape)
ih, iw = tuple(sample.shape[2:4])

# Physics constants
WAVELENGTH = 1550e-9 # meters
SPEED_LIGHT: float = 299_792_458.0  # [m/s] speed of light
GRID_SPACING = 0.1 * WAVELENGTH # meters


# Size of grid boundary layer
bw = 10
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


# Create PML boundaries
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
# The weights for the loss.
grid_params_to_learn += [loss_step_weights]
# Nonlinearity weights for the substrate.
grid_params_to_learn += [util.get_object_by_name(grid, 'cc_substrate').nonlin_conv.weight]
grid_params_to_learn += [util.get_object_by_name(grid, 'cc_substrate').nonlin_conv.bias]

# Load saved params for model and optimizer.
reset_optimizer = False
if(args.load_file is not None):
    start_step = int(args.load_file.split('/')[-1].split('_')[-1].split('.')[0])
    print('Loading model {0}. Starting at step {1}.'.format(args.load_file, start_step))
    optimizer_path = args.load_file.rsplit('.', 1)[0] + '.opt'
    grid_path = args.load_file.rsplit('.', 1)[0] + '.grd'
    model.load_state_dict(torch.load(args.load_file))
    # Load grid params and interpolate them if necessary
    with torch.no_grad():
        load_grid_params_to_learn = torch.load(grid_path)
        for idx, tensor in enumerate(load_grid_params_to_learn):
            if(tensor.shape == grid_params_to_learn[idx][...].shape):
                grid_params_to_learn[idx][...] = tensor[...]
                print('Loaded grid param {0} with shape: {1}'.format(idx, tensor.shape)) 
            else:
                # Interpolate the thing....
                print('INFO: Shapes are mismatched: {0} vs {1}'.format(tensor[...].shape, grid_params_to_learn[idx][...].shape))
                reset_optimizer = True
else:
    print('Starting model at step 0')
    start_step = 0
    optimizer_path = None
    grid_path = None

# Combine grid and model params and register them with the optimizer.
params_to_learn = [*model.parameters()] + grid_params_to_learn
if(args.use_sgd):
    optimizer = optim.SGD(params_to_learn, lr=0.01)
else:
    optimizer = optim.AdamW(params_to_learn, lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
if((optimizer_path is not None) and (not reset_optimizer) and (not args.reset_grid_optim)):
    print('INFO: Loading saved optimizer params...')
    optimizer.load_state_dict(torch.load(optimizer_path))
else:
    print('INFO: Starting with a fresh optimizer.')

mse = torch.nn.MSELoss(reduce=False)
loss_fn = torch.nn.MSELoss()

grid.H.requires_grad = True
grid.H.retain_grad()
grid.E.requires_grad = True
grid.E.retain_grad()

# For timing steps
stopwatch = time.time()

# Train the weights
for train_step in range(start_step + 1, start_step + args.max_steps):
    # Generate a new image
    img = util.get_sample_img(train_loader, device, color=not args.grayscale)

    # Reset grid and optimizer
    grid.reset()
    optimizer.zero_grad()

    loss_list = []
    E_energy_jump_list = []
    H_energy_jump_list = []
    energy_loss = 0
    last_energy_E = 0
    last_energy_H = 0 
    # Get sample from training data
    em_step_loss_weight_dist = softmax(loss_step_weights)
    argmax_step = torch.argmax(torch.squeeze(loss_step_weights))
    for em_step, (img_hat_em, em_field) in enumerate(model(img, summary_writer=writer, train_step=train_step)):
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
        if(em_step == argmax_step):
            e_field_img = em_field[0:3,...]
            h_field_img = em_field[3:6,...]
            img_hat_em_save = img_hat_em
            e_field_energy_save = energy_E
            h_field_energy_save = energy_H
    loss_per_step = torch.stack(loss_list)
    E_energy_jump_per_step = torch.stack(E_energy_jump_list)
    H_energy_jump_per_step = torch.stack(H_energy_jump_list)
    # Bypass em_step_loss_weight_dist if target_half_way enabled.
    if(args.target_half_way):
        weighted_loss_per_step = loss_per_step
    else:
        weighted_loss_per_step = loss_per_step * em_step_loss_weight_dist

    reconstruction_loss = torch.sum(weighted_loss_per_step) 

    # Calculate the total loss
    loss = reconstruction_loss + energy_loss

    # Add the argmaxxed images to tensorboard
    if(args.grayscale):
        img =  img.expand(-1, 3, -1, -1)
        img_hat_em_save =  img_hat_em_save.expand(3, -1, -1)
    img_grid = torchvision.utils.make_grid([img[0,...], img_hat_em_save,
        util.norm_img_by_chan(e_field_img), 
        util.norm_img_by_chan(h_field_img)])
    writer.add_image('sample', img_grid, train_step)

    writer.add_scalar('Total Loss', loss, train_step)
    writer.add_scalar('Reconstruction Loss', reconstruction_loss, train_step)
    writer.add_scalar('Metabolic Loss', energy_loss, train_step)
    writer.add_scalar('Max E Energy Jump', bd.max(E_energy_jump_per_step), train_step)
    writer.add_scalar('Max H Energy Jump', bd.max(H_energy_jump_per_step), train_step)
    writer.add_scalar('Max E Energy Step', torch.argmax(E_energy_jump_per_step), train_step)
    writer.add_scalar('Max H Energy Step', torch.argmax(H_energy_jump_per_step), train_step)
    writer.add_scalar('Sample E Energy', e_field_energy_save, train_step)
    writer.add_scalar('Sample H Energy', h_field_energy_save, train_step)
    writer.add_histogram('Loss Per Step', loss_per_step, train_step)
    writer.add_histogram('Weighted Loss Per Step', weighted_loss_per_step, train_step)
    writer.add_histogram('Loss EM Step Weights', loss_step_weights, train_step)
    writer.add_histogram('ccsubstrate_nonlin_w', util.get_object_by_name(grid, 'cc_substrate').nonlin_conv.weight, train_step)
    writer.add_histogram('ccsubstrate_nonlin_b', util.get_object_by_name(grid, 'cc_substrate').nonlin_conv.bias, train_step)
    writer.add_scalar('em_steps', em_steps, train_step)
    writer.add_scalar('Argmax EM Step', argmax_step, train_step)
    writer.add_scalar('Argmax EM Step Ratio', argmax_step/em_steps, train_step)

    print('Step: ', train_step, '\tTime: ', grid.time_steps_passed, '\tLoss: ', loss)

    # Tensorboard
    writer.add_histogram('sm_conv_lin', model.sm_conv_linear.weight, train_step)
    writer.add_histogram('e_field', e_field_img, train_step)
    writer.add_histogram('h_field', h_field_img, train_step)
    writer.add_histogram('e_field_energy_jumps', E_energy_jump_per_step, train_step)
    writer.add_histogram('h_field_energy_jumps', H_energy_jump_per_step, train_step)

    optimizer.zero_grad()
    # Backprop
    loss.backward(retain_graph=True)
    optimizer.step()

    # Save model 
    if(not args.dry_run):
        if((train_step % args.save_steps == 0) and (train_step > 0)):
            torch.save(model.state_dict(), model_checkpoint_dir + 'md_'+str(train_step).zfill(12)+'.pt')
            torch.save(optimizer.state_dict(), model_checkpoint_dir + 'md_'+str(train_step).zfill(12)+'.opt')
            torch.save(grid_params_to_learn, model_checkpoint_dir + 'md_'+str(train_step).zfill(12)+'.grd')

    # Profile performance
    seconds_per_step = time.time() - stopwatch 
    writer.add_scalar('seconds_per_step', torch.tensor(seconds_per_step), train_step)
    stopwatch = time.time()

writer.close()
