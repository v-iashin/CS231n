from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    N = x.shape[0]
    D = np.prod(x.shape[1:])
    x_ = x.reshape(N, D)
    out = x_.dot(w) + b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    N = x.shape[0]
    D = np.prod(x.shape[1:])

    x_ = x.reshape(N, D)
    dx = dout.dot(w.T).reshape(x.shape)
    
    dw = x_.T.dot(dout)

    db = dout.sum(axis=0) * np.ones_like(b)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = np.maximum(0, x)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    relu_x = np.maximum(0, x)
    drelu_x = np.zeros_like(relu_x)
    drelu_x[relu_x > 0] = 1
    dx = drelu_x * dout
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #######################################################################
        # Step 1: (D,)
        mean = x.sum(axis=0) / N
        # Step 2: (N, D)
        x_mean = x - mean
        # Step 3: (N, D)
        var_ = x_mean ** 2
        # Step 4: (D,)
        var = var_.sum(axis=0) / N
        # Step 5: (D,)
        sqrtvar = np.sqrt(var + eps)
        # Step 6: (D,)
        invsqrtvar = 1 / sqrtvar
        # Step 7: (N, D)
        x_hat = x_mean * invsqrtvar
        # Step 8: (N, D):
        gamma_xhat = gamma * x_hat
        # Step 9: (N, D)
        out = gamma_xhat + beta

        running_mean = momentum*running_mean + (1-momentum)*mean
        running_var = momentum*running_var + (1-momentum)*var

        cache = bn_param, gamma, x, x_hat, invsqrtvar, x_mean, sqrtvar, var, var_
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        mean = running_mean
        var = running_var
        x_hat = (x - mean) / np.sqrt(var + eps)
        out = gamma * x_hat + beta

        cache = bn_param, gamma, x, x_hat, var
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    ###########################################################################
    bn_param, gamma, x, x_hat, invsqrtvar, x_mean, sqrtvar, var, var_ = cache
    N, D = dout.shape
    eps = bn_param.get('eps', 1e-5)

    # BStep 9: out = gamma_xhat + beta
    dbeta = dout.sum(axis=0)
    dgamma_xhat = dout
    # BStep 8: gamma_xhat = gamma * x_hat
    dgamma = (dgamma_xhat * x_hat).sum(axis=0)
    dx_hat = dgamma_xhat * gamma
    # BStep 7: x_hat = x_mean * invsqrtvar
    dx_mean = dx_hat * invsqrtvar # N, D
    dinvsqrtvar = (dx_hat * x_mean).sum(axis=0) # D,
    # BStep 6: invsqrtvar = 1 / sqrtvar
    dsqrtvar = dinvsqrtvar * (-1/(sqrtvar**2)) # D,
    # BStep 5: sqrtvar = np.sqrt(var + eps)
    dvar = dsqrtvar * 0.5 / np.sqrt(var + eps) # D,
    # BStep 4: var = var_.sum(axis=0) / N
    dvar_ = dvar * np.ones_like(var_) / N # N, D
    # BStep 3: var_ = x_mean ** 2
    dx_mean += dvar_ * 2 * x_mean # N, D
    # BStep 2: x_mean = x - mean
    dx = dx_mean # N, D
    dmean = -dx_mean.sum(axis=0) # D,
    # BStep 1: mean = x.sum(axis=0) / N
    dx += dmean * np.ones_like(x) / N
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        mask = (np.random.rand(*x.shape) > p) / p
        out = x * mask
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = dout * mask
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    pad = conv_param['pad']
    stride = conv_param['stride']

    assert (H + 2 * pad - HH) % stride == 0
    assert (W + 2 * pad - WW) % stride == 0

    H_pad = H + pad*2
    W_pad = W + pad*2

    x_pad = np.zeros((N, C, H_pad, W_pad))

    for n in range(N):
        for c in range(C):
            x_pad[n, c, :, :] = np.pad(x[n, c], pad, 'constant')
            
    H_out = int(1 + (H + 2 * pad - HH) / stride) # assertion
    W_out = int(1 + (W + 2 * pad - WW) / stride) # assertion
    out = np.zeros((N, F, H_out, W_out))

    for n in range(N):
        h_out = 0
        for h_ in range(0, H_pad, stride):
            w_out = 0
            if h_+HH <= H_pad:
                for w_ in range(0, W_pad, stride):
                    if w_+WW <= W_pad:
                        x_local = x_pad[n, :, h_:h_+HH, w_:w_+WW]
                        out[n, :, h_out, w_out] = np.sum(x_local * w, axis=(1, 2, 3)) + b
                    w_out += 1
            h_out += 1
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    N, F, H_, W_ = dout.shape
    pad = conv_param['pad']
    stride = conv_param['stride']

    H_pad = H + pad*2
    W_pad = W + pad*2

    x_pad = np.zeros((N, C, H_pad, W_pad))

    for n in range(N):
        for c in range(C):
            x_pad[n, c, :, :] = np.pad(x[n, c], pad, 'constant')

    dw = np.zeros_like(w)

    for f in range(F):
        for c in range(C):
            dw_ = np.zeros((1, 1, HH, WW))
            for n in range(N):
                x_h = 0
                for h_ in range(H_):
                    x_w = 0
                    for w_ in range(W_):
                        dw_ += dout[n, f, h_, w_] * x_pad[n, c, x_h:x_h+HH, x_w:x_w+WW]
                        x_w += stride
                    x_h += stride
            dw[f, c, :, :] = dw_

    dx_pad = np.zeros((N, C, H_pad, W_pad))

    for n in range(N):
        for c in range(C):
            x_h = 0
            for h_ in range(H_):
                x_w = 0
                for w_ in range(W_):
                    for f in range(F):
                        dx_pad[n, c, x_h:x_h+HH, x_w:x_w+WW] += dout[n, f, h_, w_] * w[f, c, :, :]
                    x_w += stride
                x_h += stride
    dx = dx_pad[:, :, pad:H_pad-pad, pad:W_pad-pad]

    # db
    db = np.sum(dout, axis=(0, 2, 3))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################
    N, C, H, W = x.shape
    
    stride = pool_param['stride']
    H_pool = pool_param['pool_height']
    W_pool = pool_param['pool_width']

    H_out = round((H - H_pool) / stride + 1) # danger
    W_out = round((W - W_pool) / stride + 1) # danger
    # print(H_out, W_out)
    out = np.zeros((N, C, H_out, W_out))

    h_out = 0
    for h in range(0, H, stride):
        w_out = 0
        if h+H_pool <= H:
            for w in range(0, W, stride):
                if w+W_pool <= W:
                    out[:, :, h_out, w_out] = np.max(x[:, :, h:h+H_pool, w:w+W_pool], axis=(2, 3))
                w_out += 1
        h_out += 1
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################
    x, pool_param = cache
    N, C, H, W = x.shape

    stride = pool_param['stride']
    H_pool = pool_param['pool_height']
    W_pool = pool_param['pool_width']

    x_maxes = {}

    for n in range(N):
        for c in range(C):
            for h in range(0, H, stride):
                if h+H_pool <= H:
                    for w in range(0, W, stride):
                        if w+W_pool <= W:
                            mat = x[n, c, h:h+H_pool, w:w+W_pool]
                            x_maxes[n, c, h, w] = np.unravel_index(mat.argmax(), mat.shape)

    dx = np.zeros_like(x)
    
    for n in range(N):
        for c in range(C):
            h_out = 0
            for h in range(0, H, stride):
                if h+H_pool <= H:
                    w_out = 0
                    for w in range(0, W, stride):
                        if w+W_pool <= W:
                            max_loc_h, max_loc_w = x_maxes[n, c, h, w]
                            dx[n, c, max_loc_h+h, max_loc_w+w] += dout[n, c, h_out, w_out]
                        w_out += 1
                h_out += 1
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################

    # the implementation seems to exceed the 'five lines' limit
    # though it is done for a reason. Most of the lines could be made as
    # oneliners (V. Y.)

    N, C, H, W = x.shape
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)
    running_mean = bn_param.get('running_mean', np.zeros(C, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(C, dtype=x.dtype))

    if bn_param['mode'] == 'train':
        # Step 1: (C,)
        mean = x.sum(axis=(0, 2, 3)) / (N*H*W)
        # Step 2: (N, C, H, W)
        x_mean = x - mean.reshape(1, C, 1, 1)
        # Step 3: (N, C, H, W)
        var_ = x_mean ** 2
        # Step 4: (C,)
        var = var_.sum(axis=(0, 2, 3)) / (N*H*W)
        # Step 5: (C, H, W,)
        sqrtvar = np.sqrt(var + eps)
        # Step 6: (C, H, W,)
        invsqrtvar = 1 / sqrtvar
        # Step 7: (N, C, H, W)
        x_hat = x_mean * invsqrtvar.reshape(1, C, 1, 1)
        # Step 8: (N, C, H, W):
        gamma_xhat = gamma.reshape(1, C, 1, 1) * x_hat
        # Step 9: (N, D)
        out = gamma_xhat + beta.reshape(1, C, 1, 1)

        # print(running_mean.shape, mean.shape)

        bn_param['running_mean'] = momentum*running_mean + (1-momentum)*mean
        bn_param['running_var'] = momentum*running_var + (1-momentum)*var

        cache = bn_param, gamma, x, x_hat, invsqrtvar, x_mean, sqrtvar, var, var_

    elif bn_param['mode'] == 'test':
        mean = running_mean.reshape(1, C, 1, 1)
        var = running_var.reshape(1, C, 1, 1)
        x_hat = (x - mean) / np.sqrt(var + eps)
        out = gamma.reshape(1, C, 1, 1) * x_hat + beta.reshape(1, C, 1, 1)

        cache = bn_param, gamma, x, x_hat, var

    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################

    # the implementation seems to exceed the 'five lines' limit
    # though it is done for a reason. Most of the lines could be made as
    # oneliners (V. Y.)

    bn_param, gamma, x, x_hat, invsqrtvar, x_mean, sqrtvar, var, var_ = cache
    N, C, H, W = dout.shape
    eps = bn_param.get('eps', 1e-5)

    # BStep 9: out = gamma_xhat + beta.reshape(1, C, 1, 1)
    dbeta = dout.sum(axis=(0, 2, 3))
    dgamma_xhat = dout
    # BStep 8: gamma_xhat = gamma.reshape(1, C, 1, 1) * x_hat
    dgamma = (dgamma_xhat * x_hat).sum(axis=(0, 2, 3))
    dx_hat = dgamma_xhat * gamma.reshape(1, C, 1, 1)
    # BStep 7: x_hat = x_mean * invsqrtvar.reshape(1, C, 1, 1)
    dx_mean = dx_hat * invsqrtvar.reshape(1, C, 1, 1) # N, C, H, W
    dinvsqrtvar = (dx_hat * x_mean).sum(axis=(0, 2, 3)) # C, H, W
    # BStep 6: invsqrtvar = 1 / sqrtvar
    dsqrtvar = dinvsqrtvar * (-1/(sqrtvar**2)) # C, H, W
    # BStep 5: sqrtvar = np.sqrt(var + eps)
    dvar = dsqrtvar * 0.5 / np.sqrt(var + eps) # C, H, W
    # BStep 4: var = var_.sum(axis=0) / N
    dvar_ = dvar / (N*H*W) # N, C, H, W
    # BStep 3: var_ = x_mean ** 2
    dx_mean += dvar_.reshape(1, C, 1, 1) * 2 * x_mean # N, C, H, W
    # BStep 2: x_mean = x - mean.reshape(1, C, 1, 1)
    dx = dx_mean # N, C, H, W
    dmean = -dx_mean.sum(axis=(0, 2, 3)) # C, H, W
    # BStep 1: mean = x.sum(axis=(0, 2, 3)) / (N*H*W)
    dx += dmean.reshape(1, C, 1, 1) / (N*H*W)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
