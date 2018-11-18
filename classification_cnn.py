"""ClassificationCNN"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ClassificationCNN(nn.Module):
    """
    A PyTorch implementation of a three-layer convolutional network
    with the following architecture:

    conv - relu - 2x2 max pool - fc - dropout - relu - fc

    The network operates on mini-batches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels  """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, kernel_size=7,
                 stride_conv=1, weight_scale=0.001, pool=2, stride_pool=2, hidden_dim=100,
                 num_classes=10, dropout=0.0):
        """
        Initialize a new network.
        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data.
        - num_filters: Number of filters to use in the convolutional layer.
        - kernel_size: Size of filters to use in the convolutional layer.
        - hidden_dim: Number of units to use in the fully-connected hidden layer-
        - num_classes: Number of scores to produce from the final affine layer.
        - stride_conv: Stride for the convolution layer.
        - stride_pool: Stride for the max pooling layer.
        - weight_scale: Scale for the convolution weights initialization
        - pool: The size of the max pooling window.
        - dropout: Probability of an element to be zeroed.
        """
        super(ClassificationCNN, self).__init__()
        self.channels, self.height, self.width = input_dim

        ########################################################################
        # TODO: Initialize the necessary trainable layers to resemble the      #
        # ClassificationCNN architecture  from the class docstring.            #
        #                                                                      #
        # In- and output features should not be hard coded which demands some  #
        # calculations especially for the input of the first fully             #
        # convolutional layer.                                                 #
        #                                                                      #
        # The convolution should use "same" padding which can be derived from  #
        # the kernel size and its weights should be scaled. Layers should have #
        # a bias if possible.                                                  #
        #                                                                      #
        # Note: Avoid using any of PyTorch's random functions or your output   #
        # will not coincide with the Jupyter notebook cell.                    #
        ########################################################################
       
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride_conv = stride_conv
        self.pool = pool
        self.stride_pool = stride_pool
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout =  dropout
        self.weight_scale = weight_scale
        self.padding = (self.kernel_size -1) // 2  # for 'SAME' padding

       # self.conv_layer = 
        #conv_layer.weight = conv_layer.weight * self.weight_scale 
        self.conv_layer = nn.Conv2d(self.channels,self.num_filters, self.kernel_size, stride_conv,self.padding,bias=True)
        self.conv_layer.weight.data.mul_(weight_scale) 

        #layer1 ===== conv - relu - 2x2 max pool
        #print(input_dim)
        self.layer1 = nn.Sequential(
            self.conv_layer,
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = self.pool, stride=self.stride_pool)
        )
        #print(self.layer1.size())
        #self.conv_height_out = math.ceil(1 + (self.height - self.kernel_size  + 2 * self.padding)/self.stride_conv)
        #self.conv_width_out = math.ceil(1 + (self.width - self.kernel_size  + 2 * self.padding)/self.stride_conv)
        
       
        self.op_height = (((self.height - self.pool)//self.stride_pool) + 1)
        self.op_width = (((self.width - self.pool)//self.stride_pool) + 1)
        self.size_output_layer1 = self.num_filters * self.op_height * self.op_width
        print(self.op_height)
        print(self.op_width)
        

        self.layer2 = nn.Sequential(
            nn.Linear(self.size_output_layer1,self.hidden_dim,bias=True),
            torch.nn.Dropout(p=self.dropout),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_classes)
        )
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def forward(self, x):
        """
        Forward pass of the neural network. Should not be called manually but by
        calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """

        ########################################################################
        # TODO: Chain our previously initialized fully-connected neural        #
        # network layers to resemble the architecture drafted in the class     #
        # docstring. Have a look at the Variable.view function to make the     #
        # transition from the spatial input image to the flat fully connected  #
        # layers.                                                              #
        ########################################################################
        # conv - relu - 2x2 max pool - fc - dropout - relu - fc
        
        #out = nn.Sequential(self.conv1(x),nn.ReLU(),nn.MaxPool2d(self.kernel_size, self.stride_pool))
        #out = nn.Sequential(self.fc1(out))
        #out = nn.Sequential(nn.Dropout(self.dropout),nn.ReLU())
        #out = nn.Linear(out)
       # print (x.size())
        out = self.layer1(x)
        #print(out.size())
        out = out.view(out.size()[0],-1)
        ##print(x.size())
        x = self.layer2(out)
                   

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return x

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
