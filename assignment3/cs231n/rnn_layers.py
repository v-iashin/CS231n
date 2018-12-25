from __future__ import print_function, division
from builtins import range
import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """
    next_h, cache = None, None
    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
    # hidden state and any values you need for the backward pass in the next_h   #
    # and cache variables respectively.                                          #
    ##############################################################################
    # Step 1: (N, H) dot (H, H) and (N, D) dot (D, H)
    hWh = prev_h.dot(Wh)
    xWx = x.dot(Wx)

    # Step 2: (N, H)
    hWh_p_xWx = hWh + xWx

    # Step 3: (N, H)
    hWh_p_xWx_p_b = hWh_p_xWx + b

    # Step 4: (N, H)
    next_h = np.tanh(hWh_p_xWx_p_b)

    cache = (x, prev_h, Wx, Wh, b, hWh_p_xWx_p_b)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state
    - cache: Cache object from the forward pass

    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
    #                                                                            #
    # HINT: For the tanh function, you can compute the local derivative in terms #
    # of the output value from tanh.                                             #
    ##############################################################################
    x, prev_h, Wx, Wh, b, hWh_p_xWx_p_b = cache

    # Step 4: next_h = np.tanh(hWh_p_xWx_p_b)
    dhWh_p_xWx_p_b = dnext_h * (1 - np.tanh(hWh_p_xWx_p_b)**2) # (N, H) (dot) (N, H)

    # Step 3: hWh_p_xWx_p_b = hWh_p_xWx + b
    dhWh_p_xWx = dhWh_p_xWx_p_b # (N, H)
    db = np.sum(dhWh_p_xWx_p_b, axis=0) # (H,)

    # Step 2: hWh_p_xWx = hWh + xWx
    dhWh = dhWh_p_xWx # (N, H)
    dxWx = dhWh_p_xWx # (N, H)

    # Step 1: hWh = prev_h.dot(Wh)     and     xWx = x.dot(Wx)
    dprev_h = dhWh.dot(Wh.T) # WHY SHOULD IT BE TRANSPOSED?!
    dWh = prev_h.T.dot(dhWh)
    dx = dxWx.dot(Wx.T)
    # print(dxWx.shape, Wx.T.shape)
    dWx = x.T.dot(dxWx)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    h, cache = None, None
    ##############################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above. You can use a for loop to help compute the forward pass.            #
    ##############################################################################
    N, T, D = x.shape
    D, H = Wx.shape
    h_shape = (N, T, H)
    h = np.zeros(h_shape)
    hWh_p_xWx_p_b = np.zeros(h_shape)

    for t in range(T):

        if t == 0:
            next_h, cache_ = rnn_step_forward(x[:, t, :], h0, Wx, Wh, b)
            x_, prev_h_, Wx_, Wh_, b_, hWh_p_xWx_p_b_ = cache_
            hWh_p_xWx_p_b[:, t, :] = hWh_p_xWx_p_b_

        else:
            next_h, cache_ = rnn_step_forward(x[:, t, :], h[:, t-1, :], Wx, Wh, b)
            x_, prev_h_, Wx_, Wh_, b_, hWh_p_xWx_p_b_ = cache_
            hWh_p_xWx_p_b[:, t, :] = hWh_p_xWx_p_b_

        h[:, t, :] = next_h

    cache = (x, h, h0, Wx, Wh, b, hWh_p_xWx_p_b)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H)

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire      #
    # sequence of data. You should use the rnn_step_backward function that you   #
    # defined above. You can use a for loop to help compute the backward pass.   #
    ##############################################################################
    x, h, h0, Wx, Wh, b, hWh_p_xWx_p_b = cache
    N, T, H = dh.shape
    N, T, D = x.shape
    dh0 = np.zeros((N, H))
    dx = np.zeros_like(x)
    dWx = np.zeros_like(Wx)
    dWh = np.zeros_like(Wh)
    db = np.zeros_like(b)
    dprev_h = np.zeros_like(dh0)

    for t in range(T, 0, -1):

        if t != 1:
            cache = x[:, t-1, :], h[:, t-2, :], Wx, Wh, b, hWh_p_xWx_p_b[:, t-1, :]
            dx_, dprev_h, dWx_, dWh_, db_ = rnn_step_backward(dh[:, t-1, :] + dprev_h, cache)

        elif t == 1:
            cache = x[:, t-1, :], h0, Wx, Wh, b, hWh_p_xWx_p_b[:, t-1, :]
            dx_, dprev_h, dWx_, dWh_, db_ = rnn_step_backward(dh[:, t-1, :] + dprev_h, cache)
            dh0 = dprev_h

        dx[:, t-1, :] = dx_
        dWx += dWx_
        dWh += dWh_
        db += db_
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings. We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    to a vector of dimension D.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x muxt be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    ##############################################################################
    # TODO: Implement the forward pass for word embeddings.                      #
    #                                                                            #
    # HINT: This can be done in one line using NumPy's array indexing.           #
    ##############################################################################
    N, T = x.shape
    V, D = W.shape

    out = (np.arange(V) == x[..., None]).astype(int).dot(W)
    cache = (x, W)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return out, cache


def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings. We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.

    HINT: Look up the function np.add.at

    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass

    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D).
    """
    dW = None
    ##############################################################################
    # TODO: Implement the backward pass for word embeddings.                     #
    #                                                                            #
    # Note that Words can appear more than once in a sequence.                   #
    # HINT: Look up the function np.add.at                                       #
    ##############################################################################
    x, W = cache
    dW = np.zeros_like(W)
    N, T, D = dout.shape
    V, D = W.shape

    xohe = (np.arange(V) == x[..., None]).astype(int) # (N, T, V)

    # for n in range(N):
    #     dW += xohe[n].T.dot(dout[n])

    dW = np.sum(np.einsum('NTV, NTD -> NVD', xohe, dout), axis=0)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dW


def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    next_h, next_c, cache = None, None, None
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    #############################################################################
    N, H = prev_h.shape

    # Step 1: 
    hWh = prev_h.dot(Wh) # (N, 4H) <- (N, H) x (H, 4H)
    xWx = x.dot(Wx) # (N, 4H) <- (N, D) x (D, 4H)

    # Step 2:
    A = hWh + xWx + b # (N, 4H)

    # Step 3:
    Ai = A[:, 0*H:1*H] # (N, H)
    Af = A[:, 1*H:2*H] # (N, H)
    Ao = A[:, 2*H:3*H] # (N, H)
    Ag = A[:, 3*H:4*H] # (N, H)

    i = sigmoid(Ai)
    f = sigmoid(Af)
    o = sigmoid(Ao)
    g = np.tanh(Ag)

    # Step 4:
    fprev_c = f * prev_c # (N, H)
    ig = i * g # (N, H)

    # Step 5:
    next_c = fprev_c + ig # (N, H)

    # Step 6:
    tanh_next_c = np.tanh(next_c) # (N, H)

    # Step 7:
    next_h = o * tanh_next_c  # (N, H)

    cache = (next_c, x, prev_h, prev_c, Wx, Wh, b, Ai, Af, Ao, Ag, i, f, o, g)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.

    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh, dc, dWx, dWh, db = None, None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for a single timestep of an LSTM.       #
    #                                                                           #
    # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
    # the output value from the nonlinearity.                                   #
    #############################################################################
    (next_c, x, prev_h, prev_c, Wx, Wh, b, Ai, Af, Ao, Ag, i, f, o, g) = cache

    # Step 7: next_h = o * tanh_next_c (N, H)
    do = np.tanh(next_c) * dnext_h
    dtanh_next_c = o * dnext_h

    # Step 6: tanh_next_c = np.tanh(next_c)
    dnext_c_from_h = (1 - np.tanh(next_c)**2) * dtanh_next_c

    # Step 5: next_c = fprev_c + ig
    dfprev_c = dnext_c_from_h + dnext_c # (N, H)
    dig = dnext_c_from_h + dnext_c # (N, H)

    # Step 4: 
    # fprev_c = f * prev_c
    df = prev_c * dfprev_c 
    dprev_c = f * dfprev_c
    # ig = i * g
    di = g * dig
    dg = i * dig

    # Step 3: 
    # g = np.tanh(Ag)
    dAg = (1 - np.tanh(Ag)**2) * dg
    # o = sigmoid(Ao)
    dAo = sigmoid(Ao) * (1 - sigmoid(Ao)) * do
    # f = sigmoid(Af)
    dAf = sigmoid(Af) * (1 - sigmoid(Af)) * df
    # i = sigmoid(Ai)
    dAi = sigmoid(Ai) * (1 - sigmoid(Ai)) * di

    dA = np.column_stack([dAi, dAf, dAo, dAg])

    # Step 2: A = hWh + xWx + b # (N, 4H)
    dhWh = dA
    dxWx = dA
    db = np.sum(dA, axis=0)

    # Step 1: (N, H) <- dhWh(N, 4H) Wh(H, 4H)
    # hWh = prev_h.dot(Wh) 
    dprev_h = dhWh.dot(Wh.T) # (N, H) <- dhWh(N, 4H) Wh(H, 4H)
    dWh = prev_h.T.dot(dhWh) # (H, 4H) <- prev_h(N, H) dhWh(N, 4H)
    # xWx = x.dot(Wx)
    dx = dxWx.dot(Wx.T) # (N, D) <- dxWx(N, 4H) Wx(D, 4H)
    dWx = x.T.dot(dxWx) # (D, 4H) <- x(N, D) dxWx(N, 4H)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.

    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.

    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    - cache: Values needed for the backward pass.
    """
    h, cache = None, None
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    #############################################################################
    N, T, D = x.shape
    N, H = h0.shape
    h_shape = (N, T, H)
    c_shape = (N, T, H)
    h = np.zeros(h_shape)
    c = np.zeros(c_shape)
    c0 = c[:, 0, :].copy() ##### `.copy()` IS IMPORTANT (SON OF A BITCH)
    Aj_shape = (N, T, H)
    Ai_log = np.zeros(Aj_shape)
    Af_log = np.zeros(Aj_shape)
    Ao_log = np.zeros(Aj_shape)
    Ag_log = np.zeros(Aj_shape)

    for t in range(T):

        if t == 0:
            next_h, next_c, cache = lstm_step_forward(x[:, t, :], h0, c0, Wx, Wh, b)

        else:
            next_h, next_c, cache = lstm_step_forward(x[:, t, :], h[:, t-1, :], c[:, t-1, :], Wx, Wh, b)

        (next_c, x_, prev_h, prev_c, Wx, Wh, b, Ai, Af, Ao, Ag, i, f, o, g) = cache

        h[:, t, :] = next_h
        c[:, t, :] = next_c
        Ai_log[:, t, :] = Ai
        Af_log[:, t, :] = Af
        Ao_log[:, t, :] = Ao
        Ag_log[:, t, :] = Ag
    
    cache = (x, h, c, h0, c0, Wx, Wh, b, Ai_log, Af_log, Ao_log, Ag_log)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return h, cache


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.]

    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
    # You should use the lstm_step_backward function that you just defined.     #
    #############################################################################
    x, h, c, h0, c0, Wx, Wh, b, Ai_log, Af_log, Ao_log, Ag_log = cache
    N, T, D = dh.shape
    dx = np.zeros_like(x)
    dWx = np.zeros_like(Wx)
    dWh = np.zeros_like(Wh)
    db = np.zeros_like(b)
    dprev_h = np.zeros_like(h0)
    dprev_c = np.zeros_like(c0)

    for t in range(T, 0, -1):
        Ai = Ai_log[:, t-1, :]
        Af = Af_log[:, t-1, :]
        Ao = Ao_log[:, t-1, :]
        Ag = Ag_log[:, t-1, :]
        i = sigmoid(Ai)
        f = sigmoid(Af)
        o = sigmoid(Ao)
        g = np.tanh(Ag)

        if t != 1:
            cache = (c[:, t-1, :], x[:, t-1, :], h[:, t-2, :], c[:, t-2, :], 
                Wx, Wh, b, Ai, Af, Ao, Ag, i, f, o, g)

            (dx_, dprev_h, dprev_c, dWx_, dWh_, db_
                ) = lstm_step_backward(dh[:, t-1, :] + dprev_h, dprev_c, cache)

        elif t == 1:
            cache = (c[:, t-1, :], x[:, t-1, :], h0, c0, Wx, Wh, b, Ai, Af, Ao, Ag, 
                i, f, o, g)

            (dx_, dprev_h, dprev_c, dWx_, dWh_, db_
                ) = lstm_step_backward(dh[:, t-1, :] + dprev_h, dprev_c, cache)
            dh0 = dprev_h.copy()

        dx[:, t-1, :] = dx_
        dWx += dWx_
        dWh += dWh_
        db += db_
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
    """
    Forward pass for a temporal affine layer. The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M) [D, H]
    - b: Biases of shape (M,) [H]

    Returns a tuple of:
    - out: Output data of shape (N, T, M) [N, T, H]
    - cache: Values needed for the backward pass
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache


def temporal_affine_backward(dout, cache):
    """
    Backward pass for temporal affine layer.

    Input:
    - dout: Upstream gradients of shape (N, T, M) [N, T, H]
    - cache: Values from forward pass

    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M) [D, H]
    - db: Gradient of biases, of shape (M,) [H]
    """
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, summing the loss over all timesteps and averaging across the
    minibatch.

    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional mask argument tells us
    which elements should contribute to the loss.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose: print('dx_flat: ', dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

    return loss, dx
