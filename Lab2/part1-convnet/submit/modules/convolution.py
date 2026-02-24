"""
2d Convolution Module.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def hello_do_you_copy():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Roger that from convolution.py!")

class Conv2D:
    '''
    An implementation of the convolutional layer. We convolve the input with out_channels different filters
    and each filter spans all channels in the input.
    '''

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        """
        :param in_channels: the number of channels of the input data
        :param out_channels: the number of channels of the output(aka the number of filters applied in the layer)
        :param kernel_size: the specified size of the kernel(both height and width)
        :param stride: the stride of convolution
        :param padding: the size of padding. Pad zeros to the input with padding size.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        np.random.seed(1024)
        self.weight = 1e-3 * np.random.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        self.bias = np.zeros(self.out_channels)

        self.dx = None
        self.dw = None
        self.db = None

    def forward(self, x):
        """
        The forward pass of convolution
        :param x: input data of shape (N, C, H, W)
        :return: output data of shape (N, self.out_channels, H', W') where H' and W' are determined by the convolution
                 parameters. Save necessary variables in self.cache for backward pass
        """
        out = None
        #############################################################################
        # TODO: Implement the convolution forward pass.                             #
        # Hint: 1) You may use np.pad for padding.                                  #
        #       2) You may implement the convolution with loops                     #
        #############################################################################
        p = self.padding
        sw_p = np.pad(x, pad_width=((0, 0), (0, 0), (p, p), (p, p)), constant_values=0)
        sw = sliding_window_view(sw_p, window_shape=(self.kernel_size, self.kernel_size), axis=(2, 3))
        sw = sw[:, :, ::self.stride, ::self.stride, :, :]
        # sw -> (n, c, hp, wp, k, k)

        #print("stride: ", p)
        #print("sw shape: ", sw.shape)
        #print("w shape: ", self.weight.shape)

        out = np.tensordot(sw, self.weight, axes=([1, 4, 5], [1, 2, 3]))
        # sw 0, 2, 3 and w 0 are the end result
        # out -> (n, hp, wp, o)
        # set to (n, o, h, w)
        out  = np.transpose(out, (0,3,1,2))

        # add bias per output channel
        out = out + self.bias[None, :, None, None]

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = x
        return out

    def backward(self, dout):
        """
        The backward pass of convolution
        :param dout: upstream gradients
        :return: nothing but dx, dw, and db of self should be updated
        """
        x = self.cache

        #############################################################################
        # TODO: Implement the convolution backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the convolution with loops                     #
        #       2) don't forget padding when computing dx                           #
        #############################################################################

        # keep track of real indexes via _indx matrices

        p = self.padding
        x_pad = np.pad(x, pad_width=((0, 0), (0, 0), (p, p), (p, p)), constant_values=0)

        x_indx = np.arange(np.prod(x_pad.shape)).reshape(x_pad.shape)
        k_size = self.kernel_size
        sw_indx = sliding_window_view(x_indx, window_shape=(k_size, k_size), axis=(2, 3))
        s = self.stride
        sw_indx = sw_indx[:, :, ::s, ::s, :, :]

        sw = sliding_window_view(x_pad, window_shape=(k_size, k_size), axis=(2, 3))
        sw = sw[:, :, ::s, ::s, :, :]


        # dw calculation:

        dw = np.tensordot(sw, dout, axes=([0, 2, 3], [0, 2, 3]))
        self.dw  = np.transpose(dw, (3,0,1,2))


        # dx calculation:

        # dx = dout . weight, dot product on common axis
        dxm = np.tensordot(dout, self.weight, axes=([1], [0]))
        # dx shape (4, 5, 5, 3, 3, 3)

        # to match sw_indx shape
        dxm = np.transpose(dxm, (0, 3, 1, 2, 4, 5))

        #create a index matrix for dx that matches x
        sw = sw.flatten()
        dxm = dxm.flatten()
        sw_indx = sw_indx.flatten()

        self.dx = np.zeros(x_pad.shape).flatten()
        for i in range(dxm.shape[0]):
            indx = sw_indx[i]
            self.dx[indx] += dxm[i]

        # remove pad
        self.dx = self.dx.reshape(x_pad.shape)
        if p > 0:
            self.dx = self.dx[:, :, p:-p, p:-p]


        # db calculation:

        self.db = np.sum(dout, axis=(0, 2, 3))

        #print("dout shape", dout.shape)
        #print("weight shape", self.weight.shape)
        #print("x shape", x.shape)
        #print("bias shape", self.bias.shape)
        #print("sw shape", sw.shape)
        #print("sw_indx shape", sw_indx.shape)
        #print("dw shape", self.dw.shape)
        #print("sw_indx shape", sw_indx.shape)
        #print("dxm shape", dxm.shape)
        #print("sw_indx shape", sw_indx.shape)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
