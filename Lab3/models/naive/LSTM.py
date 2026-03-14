import numpy as np
import torch
import torch.nn as nn


class LSTM(nn.Module):
    # You will need to complete the class init function, and forward function

    def __init__(self, input_size, hidden_size):
        """ Init function for LSTM class
            Args:
                input_size (int): the number of features in the inputs.
                hidden_size (int): the size of the hidden layer
            Returns:
                None
        """
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        ################################################################################
        # TODO:                                                                        #
        #   Declare LSTM weights and attributes in order specified below to pass GS.   #
        #   You should include weights and biases regarding using nn.Parameter:        #
        #       1) i_t: input gate                                                     #
        #       2) f_t: forget gate                                                    #
        #       3) g_t: cell gate, or the tilded cell state                            #
        #       4) o_t: output gate                                                    #
        #   for each equation above, initialize the weights,biases for input prior     #
        #   to weights, biases for hidden.                                             #
        #   when initializing the weights consider that in forward method you          #
        #   should NOT transpose the weights.                                          #
        #   You also need to include correct activation functions                      #
        ################################################################################
        xs, hs = self.input_size, self.hidden_size
        #print("xs: ", xs, " hs: ", hs)

        # i_t: input gate
        self.wx_i = nn.Parameter(torch.Tensor(xs, hs))
        self.bx_i = nn.Parameter(torch.Tensor(hs))
        self.wh_i = nn.Parameter(torch.Tensor(hs, hs))
        self.bh_i = nn.Parameter(torch.Tensor(hs))

        # f_t: the forget gate
        self.wx_f = nn.Parameter(torch.Tensor(xs, hs))
        self.bx_f = nn.Parameter(torch.Tensor(hs))
        self.wh_f = nn.Parameter(torch.Tensor(hs, hs))
        self.bh_f = nn.Parameter(torch.Tensor(hs))

        # g_t: the cell gate
        self.wx_g = nn.Parameter(torch.Tensor(xs, hs))
        self.bx_g = nn.Parameter(torch.Tensor(hs))
        self.wh_g = nn.Parameter(torch.Tensor(hs, hs))
        self.bh_g = nn.Parameter(torch.Tensor(hs))

        # o_t: the output gate
        self.wx_o = nn.Parameter(torch.Tensor(xs, hs))
        self.bx_o = nn.Parameter(torch.Tensor(hs))
        self.wh_o = nn.Parameter(torch.Tensor(hs, hs))
        self.bh_o = nn.Parameter(torch.Tensor(hs))

        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        self.init_hidden()

    def init_hidden(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x: torch.Tensor):
        """Assumes x is of shape (batch, sequence, feature)"""

        ################################################################################
        # TODO:                                                                        #
        #   Implement the forward pass of LSTM. Please refer to the equations in the   #
        #   corresponding section of jupyter notebook. Iterate through all the time    #
        #   steps and return only the hidden and cell state, h_t and c_t.              #
        #   h_t and c_t should be initialized to zeros.                                #
        #   Note that this time you are also iterating over all of the time steps.     #
        ################################################################################
        ht, ct = None, None
        bs, ts, xs = x.shape
        #print("x.shape: ", x.shape)
        #print("wx.shape: ", self.wx_i.shape)
        #print("bx.shape: ", self.bx_i)
        #print("wh.shape: ", self.wh_i.shape)
        #print("bh.shape: ", self.bh_i.shape)
        ht_1 = torch.zeros(bs, self.hidden_size)
        ct_1 = torch.zeros(bs, self.hidden_size)

        for t in range(ts):
            xt = x[:,t,:]
            #print("xt.shape: ", xt.shape)
            #print("ht_1.shape: ", ht_1.shape)
            it = torch.sigmoid(xt @ self.wx_i + self.bx_i + ht_1 @ self.wh_i + self.bh_i)
            ft = torch.sigmoid(xt @ self.wx_f + self.bx_f + ht_1 @ self.wh_f + self.bh_f)
            gt = torch.tanh(xt @ self.wx_g  + self.bx_g + ht_1 @ self.wh_g + self.bh_g)
            ot = torch.sigmoid(xt @ self.wx_o + self.bx_o + ht_1 @ self.wh_o + self.bh_o)
            ct = ft*ct_1 + it*gt
            ht = ot*torch.tanh(ct)
            ht_1 = ht
            ct_1 = ct

        h_t, c_t = ht, ct

        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        return (h_t, c_t)
