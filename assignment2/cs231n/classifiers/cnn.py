from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        C, H, W = input_dim
        F, C, HH, WW = num_filters, input_dim[0], filter_size, filter_size
        S = 1
        P = 1

        assert (H + 2*P - HH) % S == 0
        assert (W + 2*P - WW) % S == 0
        
        H_ = int(1 + (H + 2*P - HH) / S) # danger
        W_ = int(1 + (W + 2*P - WW) / S) # danger

        assert H_ % 2 == 0
        assert W_ % 2 == 0
        
        self.params['W1'] = np.random.normal(0, weight_scale, F*C*HH*WW).reshape(F, C, HH, WW)
        self.params['b1'] = np.zeros(F)

        self.params['W2'] = np.random.normal(0, weight_scale, F*H//2*W//2*hidden_dim).reshape(F*H//2*W//2, hidden_dim)
        self.params['b2'] = np.zeros(hidden_dim)

        self.params['W3'] = np.random.normal(0, weight_scale, hidden_dim*num_classes).reshape(hidden_dim, num_classes)
        self.params['b3'] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        N, C, H, W = X.shape
        P = conv_param['pad']
        S_conv = conv_param['stride']
        S_pool = pool_param['stride']

        cache = {}
        act = {}

        conv_out, cache['conv'] = conv_forward_fast(X, W1, b1, conv_param)
        relu1_out, cache['relu1'] = relu_forward(conv_out)
        maxpool_out, cache['maxpool'] = max_pool_forward_fast(relu1_out, pool_param)

        FC1_out, cache['fc1'] = affine_forward(maxpool_out, W2, b2)
        relu2_out, cache['relu2'] = relu_forward(FC1_out)

        scores, cache['fc2'] = affine_forward(relu2_out, W3, b3)
        softmax_act, dout = softmax_loss(scores, y)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss += softmax_act
        loss += 0.5 * self.reg * (np.sum(W1*W1) + np.sum(W2*W2) + np.sum(W3*W3))

        FC2_dout, grads['W3'], grads['b3'] = affine_backward(dout, cache['fc2'])
        grads['W3'] += self.reg * self.params['W3']

        relu2_dout = relu_backward(FC2_dout, cache['relu2'])

        FC1_dout, grads['W2'], grads['b2'] = affine_backward(relu2_dout, cache['fc1'])
        grads['W2'] += self.reg * self.params['W2']
        
        maxpool_dout = max_pool_backward_fast(FC1_dout, cache['maxpool'])

        relu1_dout = relu_backward(maxpool_dout, cache['relu1'])

        dx, grads['W1'], grads['b1'] = conv_backward_fast(relu1_dout, cache['conv'])
        grads['W1'] += self.reg * self.params['W1']
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
