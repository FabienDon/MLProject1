# -*- coding: utf-8 -*-
"""Problem Sheet 2.

Stochastic Gradient Descent
"""
import numpy as np
from helpers import batch_iter
from costs import compute_loss


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient at w from just few examples n and their corresponding y_n labels.

    Args:
        y: shape=(N, )
        tx: shape=(N,d)
        w: shape=(d, ). The vector of model parameters.

    Returns:
        An array of shape (d, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """

    B = y.shape[0]
    e = y - tx@w

    grad = -1/B*np.transpose(tx)@e
    return grad


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: shape=(N, )
        tx: shape=(N,d)
        initial_w: shape=(d, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        w: optimal weights
        loss: final loss value 
    """

    batch_size = 1
    N = y.shape[0]
    w = initial_w

    for n_iter in range(max_iters):

        batch_indices = np.random.choice(N, batch_size, replace=False)
        y_batch = y[batch_indices]
        tx_batch = tx[batch_indices]

        grad = compute_stoch_gradient(y_batch, tx_batch, w)
        loss = compute_loss(y, tx, w)

        w = w - gamma * grad

    return w, loss
