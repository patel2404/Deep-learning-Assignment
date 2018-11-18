"""SegmentationNN"""
import torch
import torch.nn as nn
import torchvision.models as models


class SegmentationNN(nn.Module):

    def __init__(self, height=240, width=240, num_classes=23):
        super(SegmentationNN, self).__init__()

        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################
        self.features = models.vgg16(pretrained=True).features

        for param in self.features.parameters():
            param.requires_grad = False

        self.conv_to24 = nn.Conv2d(512, num_classes, 1)
        self.upsample = nn.Upsample(size=(height, width), mode='bilinear')
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################
        final = nn.Sequential(self.features, self.conv_to24, self.upsample)

        scores = final(x)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return scores

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)