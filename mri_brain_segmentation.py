import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torchvision.transforms as tvt
from torch.utils.data import DataLoader

class ConvolutionBlock(nn.Module):
    """
    ConvolutionBlock class constructs the most simple convolutional block on the U-Net architecture.
    It applies one convolution to a tensor with input_channels, and outputs another tensor with
    output_channels. The convolution is applied with a kernel_size of 3 as default. A ReLU activation
    function is applied, and another convolution to the tensor, without changing the number of channels.
    """
    def __init__(self, input_channels, output_channels, kernel_size=3):
        # Initialize convolution operations
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size, stride=1, padding=0)
    
    def forward(self, im):
        # Apply ConvolutionBlock to input tensor "im"
        im = self.conv1(im)
        im = self.relu(im)
        im = self.conv2(im)
        return im


class Encoder(nn.Module):
    """
    This class constructs the encoder part of the U-Net architecture. It pulls from the ConvolutionBloc constructor
    to encode the input image, applying a convolution block, as defined at "ConvolutionBlock" and a max pool layer
    with kernel size of 2x2 as default. At the end of the forward method, it returns the intermediate ouputs generated
    during the encoding process, that will be used for concatenation, during the decoding process.
    """
    def __init__(self, channels=(3, 8, 16, 32, 64, 128)):
        super().__init__()
        # Initialize a list of blocks, defined by the number of channels.
        convolutions = list()
        for i in range(len(channels) - 1):
            convolutions.append(ConvolutionBlock(channels[i], channels[i+1]))
        self.encoder_convolutions = nn.ModuleList(convolutions)
        # Initialize max pool layer with 2x2 kernel
        self.max_pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, im):
        # Apply initialized convolution blocks and max pool layer, to input mri image
        intermediate_outputs = list()
        for convolution_block in self.encoder_convolutions:
            im = convolution_block(im)
            intermediate_outputs.append(im)
            im = self.max_pool(im)

        return intermediate_outputs

if __name__ == "__main__":
    # block = ConvolutionBlock
    # encoder = Encoder
    pass

