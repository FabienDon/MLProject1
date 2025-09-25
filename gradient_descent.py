# -*- coding: utf-8 -*-
"""Problem Sheet 2.

Gradient Descent
"""
import numpy as np
from costs import compute_loss


def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: shape=(N, )
        tx: shape=(N,d)
        w: shape=(d, ). The vector of model parameters.

    Returns:
        An array of shape (d, ) (same shape as w), containing the gradient of the loss at w.
    """
    N = y.shape[0]
    e = y - tx@w

    grad = -1/N*np.transpose(tx)@e
    return grad


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: shape=(N, )
        tx: shape=(N,d)
        initial_w: shape=(d, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        w: optimal weights
        loss: final loss value 
    """
    w = initial_w
    w_opt = w
    loss_opt = 0
    for n_iter in range(max_iters):

        grad = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        
        w = w - gamma*grad

        if loss < loss_opt:
            w_opt = w 
            loss_opt = loss

    return w_opt, loss_opt
