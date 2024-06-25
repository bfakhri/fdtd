#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn
import numpy as np
import cv2
import ipywidgets as widgets
import asyncio
import matplotlib.pyplot as plt
import skimage
import git
import sys
import datetime
from pathlib import Path
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter
import scipy
import argparse
from os import listdir
from os.path import isfile, join


# In[2]:


#screen_h, screen_w = 1440, 2560
#screen_h, screen_w = 1440//4, 2560//4
#screen_h, screen_w = 200, 200
screen_h, screen_w = 100, 100


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, using CPU instead")


def NormalizeEnergy(energy, width=3):
    mean = torch.mean(energy.flatten())
    std = torch.std(energy.flatten())
    h_cutoff = mean + std*width
    energy_normed = (torch.clip(energy, 0, h_cutoff))/(h_cutoff)
    return energy_normed

def NormalizeSine(arr):
    return (arr + 1.0)/2.0
    
def Colorize(arr):
    '''
    Assumes a tensor of shape (h, w) and outputs 
    (h, w, (rgb))
    '''
    # First convert to CIELAB space
    lab = np.stack([60*np.ones_like(arr), 128*np.cos(arr), 128*np.sin(arr)], axis=-1)
    rgb = skimage.color.lab2rgb(lab)
    return rgb


# In[3]:


# RGB --> Init Phase, Coupling Constants foreach scale
# (H, W, 3) --> (H, W, 1), (H, W, N_convs)

class AutoEncoder(nn.Module):
    def __init__(self, input_chans=1, num_scales=3, coupling_scale=20):
        super(AutoEncoder, self).__init__()
        ic = input_chans
        self.num_scales = num_scales
        self.coupling_scale = coupling_scale
        # Convolutions for common feature extractor
        self.conv1 = nn.Conv2d(ic,  8, kernel_size=5, stride=1, padding='same')
        self.conv2 = nn.Conv2d( 8, 16, kernel_size=5, stride=1, padding='same')
        self.conv3 = nn.Conv2d(16,  8, kernel_size=5, stride=1, padding='same')
        self.conv4 = nn.Conv2d( 8,  8, kernel_size=5, stride=1, padding='same')
        self.conv_phases = nn.Conv2d( 8,  1, kernel_size=5, stride=1, padding='same')
        self.conv_couple = nn.Conv2d( 8,  self.num_scales, kernel_size=5, stride=1, padding='same')
        
    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = torch.relu(x)
        x = self.conv4(x)
        x = torch.relu(x)
        z_phases = self.conv_phases(x)
        z_couple = self.coupling_scale*torch.sin(self.conv_couple(x))
        
        return z_phases, z_couple
        


# In[4]:


# Instantiate convolution dict
coupling_convs = {}
num_scales = 3
size = 3
grow = 3
for i in range(num_scales):
    conv = torch.nn.Conv2d(1, 1, size, bias=False, padding='same', padding_mode='circular', device=device)
    if(i == 0):
        k = torch.ones((size, size)).to(device)
        k = k /(torch.sum(k) - 1.0)
    else:
        #k = torch.zeros((last_size, last_size)).to(device)
        k = torch.zeros((size, size)).to(device)
        #k = torch.nn.functional.pad(k, (grow, grow, grow, grow), value=1.0)
        k[0, :] = 1  # First row
        k[-1, :] = 1  # Last row
        k[:, 0] = 1  # First column
        k[:, -1] = 1  # Last column
        k = k /(torch.sum(k))
    
    
    k[k.shape[0]//2, k.shape[0]//2] = -1
    
    conv.weight = torch.nn.Parameter(k[None, None, ...])
    coupling_convs[k.shape[0]] = conv
    print(size, k.shape, conv.weight.shape, k.sum())
    #print(k)
    last_size = size
    size += grow*2


# In[5]:


def run_sim(phases, coupling_constants, coupling_convs, frequency = 0.1, num_steps = 500, show_sim=False):
    img_mean = torch.zeros_like(phases).cuda()
    for t in range(num_steps):
        # Calculate the oscillator's positions
        img_out = torch.sin(frequency*t + phases)
        # Calculate the errors
        error_sum = torch.zeros_like(img_out)
        for k_idx, k_key in enumerate(coupling_convs):
           error = torch.sin(coupling_convs[k_key](phases))
           error_sum -= error*coupling_constants[0, k_idx]
           #print(t, torch.abs(error_sum).mean())
           break
        phases += error_sum
        
        #img_mean += NormalizeSine(torch.sin(phases))
        #img_mean += phases
        img_mean = img_mean + (torch.sin(phases)+1.0)/2.0
        if(((t % 10) == 0) and show_sim):
            img_out_waves = Colorize(img_out[0,0,...].cpu().detach().numpy())
            img_out_phases = Colorize(phases[0,0,...].cpu().detach().numpy())
            cv2.imshow('Waves', img_out_waves)
            cv2.imshow('Phases', img_out_phases)
            cv2.waitKey(1)
    return img_mean / num_steps


#run_sim(torch.rand((1, 1, screen_h, screen_w)).cuda(), torch.ones((1, 1, screen_h, screen_w, num_scales)).cuda(), coupling_convs)


# In[6]:


# Generate image
# Put through encoder
# Put through simulator --> generating an average of output
# Generate loss


# In[7]:


def get_sample_img(img_loader, device, color=True):
    _, (example_datas, labels) = next(enumerate(img_loader))
    if(color):
        sample = example_datas[0]
        sample = sample.to(device)[None, :]
    else:
        sample = example_datas[0]
        sample = torchvision.transforms.Grayscale()(sample)[0]
        sample = sample.to(device)[None, None, :]
    return sample


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

image_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.RandomVerticalFlip(p=0.5),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    #torchvision.transforms.RandomRotation(degrees=[0, 360], expand=True),
    torchvision.transforms.ColorJitter(brightness=0.5, hue=0.3),
    torchvision.transforms.RandomInvert(p=0.5),
    torchvision.transforms.Resize((screen_h, screen_w))])

train_dataset = torchvision.datasets.Flowers102('flowers102/', 
                                          split='train',
                                          download=True,
                                          transform=image_transform)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=1, 
                                           shuffle=True)


grayscale = True
print('Grayscale: ', grayscale)
sample = get_sample_img(train_loader, device, color=not grayscale)
print('Image shape: ', sample.shape)
ih, iw = tuple(sample.shape[2:4])


# In[8]:


def detect_nan_gradients(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            if torch.isnan(param.grad).any():
                print(f"NaN gradient detected in parameter: {name}")
                return True
    return False


# In[9]:


mse = torch.nn.MSELoss(reduce=False)
loss_fn = torch.nn.MSELoss()
model = AutoEncoder(num_scales=len(coupling_convs.keys())).cuda()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# In[10]:


with torch.autograd.set_detect_anomaly(True): 
    start_step = 0
    max_steps = 100000
    for train_step in range(start_step + 1, start_step + max_steps):
        # Generate a new image
        img = get_sample_img(train_loader, device, color=not grayscale)

        print('img: ', img.shape)
        

        # Run model
        phases, coupling_constants = model(img)
        print(phases.shape, coupling_constants.shape)
        img_hat = run_sim(phases, coupling_constants, coupling_convs, show_sim = bool(train_step % 10 == 1))
        print('img_hat: ', img_hat.shape)
        loss = loss_fn(img_hat, img)
        
        optimizer.zero_grad()
        loss.backward()
        #loss.backward(retain_graph=True)

        
#         if detect_nan_gradients(model):
#            print("NaN gradients detected. Skipping update.")
#            optimizer.zero_grad()  # Reset gradients to prevent propagation of NaNs
#            continue
        
        optimizer.step()
        print('Step {0} with loss: {1}'.format(train_step, loss.detach().cpu()))
        
        cv2.imshow('X', img[0,0].detach().cpu().numpy())
        cv2.imshow('X_hat', img_hat[0,0].detach().cpu().numpy())
        cv2.imshow('Phases_init', phases[0,0].detach().cpu().numpy())
        cv2.imshow('CCs_init', coupling_constants[0,0].detach().cpu().numpy())


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




