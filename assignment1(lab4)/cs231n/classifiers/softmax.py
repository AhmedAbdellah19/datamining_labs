import numpy as np
from random import shuffle

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


  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  N, D = X.shape
  C = W.shape[1]
  for i in range(N):
    score = X[i].dot(W)
    score -= np.max(score)

    dom = sum(np.exp(score))
    loss += np.log(dom) - score[y[i]]

    for c in range(C):
        delta = np.exp(score[c])/dom
        if y[i] == c:
            delta -= 1
        dW[:,c] += delta * X[i]

  loss /= N
  loss += 0.5 * reg * np.sum(W * W)
  dW /= N
  dW += reg * W
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
  N, D = X.shape
  C = W.shape[0]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  score = X.dot(W)
  score[:] -= np.max(score,axis=1,keepdims=True)
  exp_score = np.exp(score)

  dom = [0]*N
  dom[:] = np.sum(exp_score,axis=1,keepdims=True)

  loss = (np.sum(np.log(dom)) - np.sum(score[range(N), y])) / N
  loss += 0.5 * reg * np.sum(W * W)

  delta = np.exp(score) / np.sum(np.exp(score),axis=1,keepdims=True)
  delta[range(N),y] -= 1
  dW = X.T.dot(delta) / N
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
