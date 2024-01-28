import numpy as np
import torch
import torchvision
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
#import cv2
import torchvision.transforms.functional as F
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm_notebook as tqdm
import util

plt.rcParams["savefig.bbox"] = 'tight'


## Then define the model class
class AutoEncoder(nn.Module):
    def __init__(self, grid, num_em_steps, input_chans=3, output_chans=3, max_freq=100, source_stride=8, bypass_em=False):
        super(AutoEncoder, self).__init__()
        self.em_grid = grid
        self.num_em_steps = num_em_steps
        ic = input_chans
        oc = output_chans
        self.bypass_em = bypass_em
        # Convolutions for common feature extractor
        self.conv1 = nn.Conv2d(ic,  8, kernel_size=5, stride=1, padding='same')
        self.conv2 = nn.Conv2d( 8, 16, kernel_size=5, stride=1, padding='same')
        self.conv3 = nn.Conv2d(16,  8, kernel_size=5, stride=1, padding='same')
        self.conv4 = nn.Conv2d( 8,  8, kernel_size=5, stride=1, padding='same')
        self.conv5 = nn.Conv2d( 8,  8, kernel_size=5, stride=1, padding='same')
        # Convs for CC activations [0,1) scaled (amp, freq, phase)
        self.conv6 = nn.Conv2d( 8,  8, kernel_size=5, stride=1, padding='same')
        self.conv7 = nn.Conv2d( 8,  3, kernel_size=5, stride=1, padding='same')
        self.conv_downscaler = nn.Conv2d( 3,  3, kernel_size=source_stride, stride=source_stride)
        # Convs for substrate manipulation
        self.sm_conv_linear = nn.Conv2d(3, 9, kernel_size=1, stride=1, padding='same')
        # Converts E and H fields back into an image with a linear transformation
        if(bypass_em):
            self.conv_linear = nn.Conv2d(16, oc, kernel_size=1, stride=1, padding='same')
        else:
            self.conv_linear = nn.Conv2d( 6, oc, kernel_size=1, stride=1, padding='same')

        # The base frequency that is scaled via the cc_activations (learnable).
        self.max_freq = torch.nn.Parameter(torch.Tensor(max_freq))

    def get_em_plane(self):
        ' Extracts a slice along the image plane from the EM field. '
        em_plane = torch.cat([self.em_grid.E, self.em_grid.H], axis=-1)
        em_plane = em_plane[self.em_grid.sources[0].x, self.em_grid.sources[0].y]
        em_plane = torch.permute(torch.squeeze(em_plane), (2,0,1))
        return em_plane

    def forward(self, x, em_steps=None, amp_scaler=1.0, summary_writer=None, train_step=None, output_scaler=None):
        ## 1 - Extract features
        # Convert image into amplitude, frequency, and phase shift for our CCs.
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = torch.relu(x)
        x = self.conv4(x)
        x = torch.relu(x)
        x = self.conv5(x)
        x = torch.relu(x)
        x = self.conv6(x)
        x = torch.relu(x)
        x = self.conv7(x)
        x = torch.relu(x)
        # Get activations as a downscaled version of the activations.
        cc_activations = self.conv_downscaler(x)
        # Scale the activations
        # Amplitude (-1, 1)
        cc_activations[:,0] = 2*torch.sigmoid(cc_activations[:, 0]) - 1.0
        # Frequency scaler (0, 1)
        cc_activations[:,1] = torch.sigmoid(cc_activations[:, 1])
        # Phase scaler (0, 1) (already in range).
        cc_activations[:,2] = torch.sigmoid(cc_activations[:, 2])

        # Branch to substrate manipulation. Compress it to (0.1, 1.0) for stability.
        sm_activations = 0.9*torch.sigmoid(self.sm_conv_linear(x)) + 0.1

        # Write out summary if the writer is present
        if(summary_writer is not None):
            summary_writer.add_histogram('cc_amplitudes', cc_activations[0, 0], train_step)
            summary_writer.add_histogram('cc_freqs', cc_activations[0, 1], train_step)
            summary_writer.add_histogram('cc_phases', cc_activations[0, 2], train_step)
            summary_writer.add_histogram('sm_activations', sm_activations, train_step)


        if(self.bypass_em):
            x_hat_em = torch.sigmoid(self.conv_linear(cc_activations))
            em_plane = self.get_em_plane()
            yield torch.squeeze(x_hat_em), em_plane
        else:
            ## 2 - Seed the cc grid source
            #TODO reference this one by name like #3
            self.em_grid.sources[0].seed(cc_activations, amp_scaler)

            ## 3 - Seed the substrate
            util.get_object_by_name(self.em_grid, 'cc_substrate').seed(sm_activations)

            # 3 - Run the grid and generate output
            if(em_steps is None or em_steps == 0):
                em_steps = self.num_em_steps

            for em_step in range(em_steps):
                self.em_grid.run(1 , progress_bar=False)
                em_plane = self.get_em_plane()
                x_hat_em = torch.sigmoid(self.conv_linear(em_plane))
                if(output_scaler):
                    x_hat_em = torchvision.transforms.functional.resize(x_hat_em, size=(x_hat_em.shape[1] * output_scaler, x_hat_em.shape[2] * output_scaler), interpolation=torchvision.transforms.InterpolationMode.NEAREST)
                    em_plane = torchvision.transforms.functional.resize(em_plane, size=(em_plane.shape[1] * output_scaler, em_plane.shape[2] * output_scaler), interpolation=torchvision.transforms.InterpolationMode.NEAREST)
                    yield x_hat_em, em_plane
                else:
                    yield x_hat_em, em_plane
