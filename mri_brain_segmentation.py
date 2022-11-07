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
            intermediate_outputs.insert(0, im)
            im = self.max_pool(im)

        return intermediate_outputs

class Decoder(nn.Module):
    """
    This class constructs the decoder part of the U-Net model architecture. It performs upconvolutions
    to increase the size the running tensor, and it concatenates it with the previously stored intermediate
    tensors.
    """
    def __init__(self, channels=(3, 8, 16, 32, 64, 128)):
        super().__init__()
        self.channels = channels[::-1][:-1]
        # Initialize list of upconvolutions and convolution blocks
        upconvolutions = list()
        convolutions = list()
        for i in range(len(channels)):
            upconvolutions.append(nn.ConvTranspose2d(channels[i],channels[i+1],2,2))
            convolutions.append(ConvolutionBlock(channels[i], channels[i+1]))
        self.upconvolutions = nn.ModuleList(upconvolutions)
        self.decoder_convolutions = nn.ModuleList(convolutions)

    def forward(self, im, encoder_intermediate):
        for i in range(self.channels):
            # Perform upconvolution on running tensor
            im = self.upconvolutions[i](im)
            # Crop encoder intermediate tensor to match the running tensor H and W dimension
            intermediate_im = self.tensor_crop(im, encoder_intermediate[i])
            # Concatenate the running tensor with the previously obatined intermediate tensor
            im = torch.cat([im, intermediate_im], dim=1)
            # Apply decoder convolution to the concatenated tensor
            im = self.decoder_convolutions[i](im)
        
        return im

    def tensor_crop(self, im, intermediate_im):
        # Crop tensor intermediate_im to match H and W of im tensor
        (_, _, H, W) = im.size()
        intermediate_im = tvt.CenterCrop(size=[H, W])(intermediate_im)
        return intermediate_im


class UNetMRI(nn.Module):
    """
    The UNetMRI class initializes an Encoder and Decoder constructors, to build the U-Net arhcitecture.
    It takes as input the number of channels to which the model will encode. With this, we can control
    how deep the actual model is. It finally applies one last convolution to match the number of channels
    of the mask (1) as well as the H and W.
    """
    def __init__(self, channels=(3, 8, 16, 32, 64, 128)):
        self.maskChannels=1
        self.output_dimension = (128, 128)    # (H, W)
        super().__init__()
        # Initialize Encoder and Decoder constructors. Initialize last convolution to reduce number of channel to 1.
        self.encoder_block = Encoder(channels)
        self.decoder_block = Decoder(channels)
        self.segmentation_convolution = nn.Conv2d(channels[1], self.maskChannels, kernel_size=1)

    def forward(self, im):
        # Obtain intermediate encoding tensors of the input im
        intermediate_outputs = self.encoder_block(im)
        # Decode the intermediate outputs
        im = self.decoder_block(intermediate_outputs[0], intermediate_outputs[1:])
        # Apply final convolution layer to reduce number of channels
        mri_segmentation = self.segmentation_convolution(im)
        # Interpolate output to match mask dimensions H and W
        mri_segmentation = nn.functional.interpolate(mri_segmentation, self.output_dimension)
    
        return mri_segmentation








if __name__ == "__main__":
    # block = ConvolutionBlock
    # encoder = Encoder
    pass

