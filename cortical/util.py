
import sys
import datetime
from pathlib import Path
sys.path.append('/home/bij/Projects/fdtd/')
import math
import time
import fdtd
import fdtd.backend as bd
import numpy as np
import random
import torch
import torch.optim as optim
import torchvision
import scipy

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

def get_object_by_name(grid, name):
    for obj in grid.objects:
        if(obj.name == name):
            return obj
    else:
        raise Exception('Could not find object: {0}'.format(name))
        sys.exit()
def get_source_by_name(grid, name):
    for src in grid.sources:
        if(src.name == name):
            return src 
    else:
        raise Exception('Could not find object: {0}'.format(name))
        sys.exit()

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

def gauss_norm_img_by_chan(img, s=6):
    '''
    Puts each channel into the range [0,1], clipping everything outside of s standard deviations.
    Expects input to be in CHW config.
    '''
    img_flat = torch.reshape(img, (3, -1))
    chan_means = torch.mean(img_flat, dim=-1)[...,None] 
    chan_stddevs = torch.std(img_flat, dim=-1)[...,None] 
    normed_img = (img - chan_means[...,None])/(chan_stddevs[...,None]) + 0.5
    normed_img = torch.clip(normed_img, min=0, max=1)
    return normed_img 

class RandomRot90:
    # Randomly rotates the image by multiples of 90 degrees.
    def __init__(self):
        pass

    def __call__(self, sample):
        return torch.rot90(sample, k=random.randrange(4), dims=[1, 2])

def toy_img(img):
    img = torch.zeros_like(img)
    x, y, b, s = np.random.rand(4)
    max_size = 14
    min_size =  6
    max_b = 1.0
    min_b = 0.5
    x = int(x*(img.shape[-1] - max_size))
    y = int(y*(img.shape[-2] - max_size))
    b = float(min_b + b*(max_b - min_b))
    s = int(min_size + s*(max_size - min_size))
    img[..., x:x+s, y:y+s] = b
    return bd.array(img[:,0,...])

def calculate_em_energy(em_field):
    " Returns the energy of the total electric and magnetic fields "
    e_field = em_field[0:3,...]
    h_field = em_field[3:6,...]
    with torch.no_grad():
        grid_energy_E = bd.sum(bd.sum(e_field ** 2, -1))
        grid_energy_H = bd.sum(bd.sum(h_field ** 2, -1))
    return grid_energy_E, grid_energy_H


