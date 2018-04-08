import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
    
  for i in range(num_train):
    scores = X[i].dot(W) # shape C,
    scores += -np.max(scores) 
    correct_class_score = scores[y[i]] # shape 1,
    loss += -correct_class_score
    sum_class_scores = 0
    dWi = np.zeros_like(W)
    
    for j in range(num_classes):
      # it seems that it is not necessary to exclude the correct class score
      sum_class_scores += np.exp(scores[j])
      if y[i] == j:
        dWi[:, j] += X[i] * (np.exp(scores[j]) / np.sum(np.exp(scores)) - 1)
      else:
        dWi[:, j] += X[i] * np.exp(scores[j]) / np.sum(np.exp(scores))
    
    dW += dWi
    loss += np.log(sum_class_scores)

  loss += reg * np.sum(W * W)
  loss /= num_train
  dW += 2 * reg * W
  dW /= num_train
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
    
  scores = X.dot(W)
  scores += -np.max(scores).reshape(-1, 1)
  true_class_scores = np.choose(y, scores.T)
  true_class_idx = np.vstack([np.arange(y.shape[0]), y]).T

  loss = np.sum(-scores[true_class_idx[:, 0], true_class_idx[:, 1]] + np.log(np.sum(np.exp(scores), axis=1)))
  loss += reg * np.sum(W * W)
  loss /= num_train
 
  dW_parentheses = np.exp(scores) / np.sum(np.exp(scores), axis=1).reshape(-1, 1)
  dW_parentheses[true_class_idx[:, 0], true_class_idx[:, 1]] -= 1
  dW = X.T.dot(dW_parentheses)
  dW += 2 * reg * W
  dW /= num_train
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

