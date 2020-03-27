import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, XX, y, reg):
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
  X = np.copy(XX)
  num_samples = len(X)

  X /= np.max(X)

  for i in range(len(X)):

    scores = X[i].dot(W)
    scores_sum = 0

    for j in range(len(scores)):

      scores_sum += np.exp(scores[j])
      if j == y[i]:

        dW[:,j] -= X[i]

    softmax = np.exp(scores[y[i]]) / scores_sum
    softmax_j = np.exp(scores[j]) / scores_sum
    loss += -np.log(softmax)

    dW_t = np.zeros_like(dW.T)
    dW_t += X[i] * softmax_j
    dW += dW_t.T
    
  loss /= num_samples 
  dW /= num_samples 

  dW += reg*2*W
  loss += reg * np.sum(W*W)


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, XX, y, reg):
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
  X = np.copy(XX)
  #normalize
  X /= np.max(X)
  num_samples = len(X)
  scores = np.matmul(X, W)
  correct_scores = scores[np.arange(len(scores)), y]
  scores_sums = np.sum(np.exp(scores), axis=1)
  loss = - np.log( np.exp(correct_scores) / scores_sums)

  softmax_matrix = (np.exp(scores).T / np.sum(np.exp(scores), axis=1)).T

  subtract_matrix = np.zeros_like(softmax_matrix)
  subtract_matrix[np.arange(num_samples), y] = 1


  softmax_matrix -= subtract_matrix

  dW = np.matmul(X.T, softmax_matrix) / num_samples

  loss = np.sum(loss) / num_samples + reg*np.sum(W*W)
  dW += reg*2*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  
  return loss, dW

