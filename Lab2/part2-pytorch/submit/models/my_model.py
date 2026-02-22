"""
MyModel model.  (c) 2021 Georgia Tech

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

import torch
import torch.nn as nn


def hello_do_you_copy():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Roger that from my_model.py!")


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        #############################################################################
        # TODO: Initialize the network weights                                      #
        #############################################################################
        # 2 VGG
        # 1: 3 -> 64 ch (32x32 -> 16x16)
        # 2: 64 -> 128 ch (16x16 -> 8x8)

        self.relu = nn.ReLU()

        # segment 1
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv2_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # segment 2
        self.conv1_2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # classify
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        outs = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################

        # segment 1
        outs = self.conv1_1(x)
        outs = self.bn1_1(outs)
        outs = self.relu(outs)

        outs = self.conv2_1(outs)
        outs = self.bn2_1(outs)
        outs = self.relu(outs)

        outs = self.pool1(outs)  # 32x32 -> 16x16

        # segment 2
        outs = self.conv1_2(outs)
        outs = self.bn1_2(outs)
        outs = self.relu(outs)

        outs = self.conv2_2(outs)
        outs = self.bn2_2(outs)
        outs = self.relu(outs)

        outs = self.pool2(outs)  # 16x16 -> 8x8

        # classify
        outs = outs.reshape(outs.shape[0], -1)

        outs = self.fc1(outs)
        outs = self.relu(outs)
        outs = self.dropout(outs)
        outs = self.fc2(outs)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return outs
