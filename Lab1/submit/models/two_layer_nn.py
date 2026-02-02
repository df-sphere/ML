"""
MLP Model.  (c) 2021 Georgia Tech

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

# Do not use packages that are not in standard distribution of python
import numpy as np

np.random.seed(1024)
from ._base_network import _baseNetwork


class TwoLayerNet(_baseNetwork):
    def __init__(self, input_size=28 * 28, num_classes=10, hidden_size=128):
        super().__init__(input_size, num_classes)

        self.hidden_size = hidden_size
        self._weight_init()

    def _weight_init(self):
        """
        initialize weights of the network
        :return: None; self.weights is filled based on method
        - W1: The weight matrix of the first layer of shape (num_features, hidden_size)
        - b1: The bias term of the first layer of shape (hidden_size,)
        - W2: The weight matrix of the second layer of shape (hidden_size, num_classes)
        - b2: The bias term of the second layer of shape (num_classes,)
        """

        # initialize weights
        self.weights['b1'] = np.zeros(self.hidden_size)
        self.weights['b2'] = np.zeros(self.num_classes)
        np.random.seed(1024)
        self.weights['W1'] = 0.001 * np.random.randn(self.input_size, self.hidden_size)
        np.random.seed(1024)
        self.weights['W2'] = 0.001 * np.random.randn(self.hidden_size, self.num_classes)

        # initialize gradients to zeros
        self.gradients['W1'] = np.zeros((self.input_size, self.hidden_size))
        self.gradients['b1'] = np.zeros(self.hidden_size)
        self.gradients['W2'] = np.zeros((self.hidden_size, self.num_classes))
        self.gradients['b2'] = np.zeros(self.num_classes)

    def forward(self, X, y, mode='train'):
        """
        The forward pass of the two-layer net. The activation function used in between the two layers is sigmoid, which
        is to be implemented in self.,sigmoid.
        The method forward should compute the loss of input batch X and gradients of each weights.
        Further, it should also compute the accuracy of given batch. The loss and
        accuracy are returned by the method and gradients are stored in self.gradients

        :param X: a batch of images (N, input_size)
        :param y: labels of images in the batch (N,)
        :param mode: if mode is training, compute and update gradients;else, just return the loss and accuracy
        :return:
            loss: the loss associated with the batch
            accuracy: the accuracy of the batch
            self.gradients: gradients are not explicitly returned but rather updated in the class member self.gradients
        """
        loss = None
        accuracy = None
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the forward process:                                      #
        #        1) Call sigmoid function between the two layers for non-linearity  #
        #        2) The output of the second layer should be passed to softmax      #
        #        function before computing the cross entropy loss                   #
        #    2) Compute Cross-Entropy Loss and batch accuracy based on network      #
        #       outputs                                                             #
        #############################################################################
        z1 = X@self.weights['W1'] + self.weights['b1']
        a1 = self.sigmoid(z1)
        z2 = a1@self.weights['W2'] + self.weights['b2']
        a2 = self.softmax(z2)

        loss = self.cross_entropy_loss(a2, y)
        accuracy = self.compute_accuracy(a2, y)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the backward process:                                     #
        #        1) Compute gradients of each weight and bias by chain rule         #
        #        2) Store the gradients in self.gradients                           #
        #    HINT: You will need to compute gradients backwards, i.e, compute       #
        #          gradients of W2 and b2 first, then compute it for W1 and b1      #
        #          You may also want to implement the analytical derivative of      #
        #          the sigmoid function in self.sigmoid_dev first                   #
        #############################################################################
        dloss_dz2 = self.softmax(z2) - self.y_one_hot[y]
        dz2_da1 = self.weights['W2']
        da1_dz1 = self.sigmoid_dev(z1)
        dz1_dw1 = X

        """
        Shapes:
        X: (32, 784), (b, i)
        y: (32,), (b, )
        a1: (32, 128), (b, h)
        W1: (784, 128), (i, h)
        b1: (128,), (h, )
        W2: (128, 10), (h, o)
        b2: (10,), (o, )
        dloss_dz2: (32, 10), (b, o)
        dz2_da1: (128, 10),  (h, o)
        da1_dz1: (32, 128),  (b, h)
        dz1_dw1: (32, 784),  (b, i)
        input = 784
        hidden  = 128
        output = 10
        batch = 32
        """

        #              b,o       (h, o)    (b, h)   (b, i)
        #              [(b,o)@(o,h)]*(b, h)->T->(h, b)@(b, i)->T->(i, h)
        dloss_dw1 = (((dloss_dz2@dz2_da1.T)*da1_dz1).T@dz1_dw1).T

        # (i, h)
        self.gradients['W1'] = dloss_dw1/X.shape[0]

        # (b, h)
        dz2_dw2 = a1

        #            (b, o)  (b, h)
        #            [(o, b)@(b, h)]->T->(h, o)
        dloss_dw2 = (dloss_dz2.T@dz2_dw2).T

        # (h, o)
        self.gradients['W2'] = dloss_dw2/X.shape[0]

        dz1_db1 = np.ones((X.shape[0], 1))

        #            (b, o)   (h, o)   (b, h) (?)
        #           [(b,o)@(o,h)]*(b, h)->T->(h,b)@(b, 1)->(h, 1)
        dloss_db1 = ((dloss_dz2@dz2_da1.T)*da1_dz1).T@dz1_db1
        dloss_db1 = dloss_db1.reshape(dloss_db1.shape[0],)

        # (h, )
        self.gradients['b1'] = dloss_db1/X.shape[0]

        dz2_db2 = np.ones((X.shape[0], 1))

        #            (b, o)
        #            (o, b)@(b, 1)->(o, 1)
        dloss_db2 = dloss_dz2.T@dz2_db2
        dloss_db2 = dloss_db2.reshape(dloss_db2.shape[0],)

        # (o, )
        self.gradients['b2'] = dloss_db2/X.shape[0]

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return loss, accuracy
