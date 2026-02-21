"""
2d Max Pooling Module.  (c) 2021 Georgia Tech

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
    print("Roger that from max_pool.py!")

class MaxPooling:
    """
    Max Pooling of input
    """

    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache = None
        self.dx = None

    def forward(self, x):
        """
        Forward pass of max pooling
        :param x: input, (N, C, H, W)
        :return: The output by max pooling with kernel_size and stride
        """
        out = None
        #############################################################################
        # TODO: Implement the max pooling forward pass.                             #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                         #
        #############################################################################
        # axis should be the last two in x's shape
        sw = sliding_window_view(x, window_shape=(self.kernel_size, self.kernel_size), axis=(2, 3))
        # sw -> (n, c, hp, wp, k, k)

        # hp, wp
        H_out = sw.shape[2]
        W_out = sw.shape[3]

        sw = sw[:, :, ::self.stride, ::self.stride, :, :]

        # stride over k, k, the last two
        out = np.max(sw, axis=(4, 5))

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = (x, H_out, W_out)
        return out

    def backward(self, dout):
        """
        Backward pass of max pooling
        :param dout: Upstream derivatives
        :return: nothing, but self.dx should be updated
        """
        x, H_out, W_out = self.cache
        #############################################################################
        # TODO: Implement the max pooling backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                         #
        #       2) You may find np.unravel_index useful                             #
        #############################################################################

        # keep track of real indexes via _indx matrices

        x_indx = np.arange(np.prod(x.shape)).reshape(x.shape)
        k_size = self.kernel_size
        sw_indx = sliding_window_view(x_indx, window_shape=(k_size, k_size), axis=(2, 3))
        s = self.stride
        sw_indx = sw_indx[:, :, ::s, ::s, :, :]

        sw = sliding_window_view(x, window_shape=(k_size, k_size), axis=(2, 3))
        sw = sw[:, :, ::s, ::s, :, :]

        #print("stride: ", s)
        #print("sw shape: ", sw.shape)
        #print("sw_indx shape: ", sw_indx.shape)
        #print("sw: ", sw)
        #print("sw_indx: ", sw_indx)

        total = np.prod(sw.shape)
        m = sw.reshape((int(total/(k_size**2)), k_size**2))
        m_indx = sw_indx.reshape((int(total/(k_size**2)), k_size**2))

        #print("m shape: ", m.shape)
        #print("m_indx shape: ", m_indx.shape)
        #print("m_indx: ", m_indx)

        # receptive field argmax per row
        amax = np.argmax(m, axis=1)
        #print("amax: ", amax)
        rows = np.arange(m.shape[0])
        amax_indx = m_indx[rows, amax]
        #print("amax_indx: ", amax_indx)
        z = np.zeros(np.prod(x.shape))

        dout = dout.flatten()
        for i, d in enumerate(dout):
            z[amax_indx[i]] += d

        self.dx = z.reshape(x.shape)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
