import numpy as np
from Ridge_reg import ridge_regression

def build_poly(x, degree):
    """
    polynomial basis functions for input data x, for j=0 up to j=degree.

    Parameters:
    
    x : numpy array
        Input data, shape (N,).
    degree : int
        Degree of the polynomial.

    Returns:
    
    numpy array of shape (N, degree+1) with polynomial features.
    """
    tx = np.hstack([x**j for j in range(degree + 1)])
    return tx


def build_k_indices(y, k_fold, seed=1):
    """build k indices for k-fold.

    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

    >>> build_k_indices(np.array([1., 2., 3., 4.]), 2, 1)
    array([[3, 2],
           [0, 1]])
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, k_indices, k, model_fn, degree, loss_fn, **kwargs):
    """
    Generic k-th fold cross-validation.
    
    Args:
        y: shape=(N, )
        x: shape=(N, )
        k_indices: output of build_k_indices
        k: the k-th fold
        model_fn: function that returns predicted y, e.g., ridge_regression, least_squares, etc.
        degree: polynomial degree
        loss_fn: function to compute loss, e.g., rmse, mse
        kwargs: additional parameters for the model

    Returns:
        loss_tr: loss on training data
        loss_te: loss on test data
    """

    # split train/test indices
    test_idx = k_indices[k]
    train_idx = k_indices[np.arange(k_indices.shape[0]) != k].reshape(-1)

    x_train, y_train = x[train_idx], y[train_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    # polynomial expansion
    tx_train = build_poly(x_train, degree)
    tx_test = build_poly(x_test, degree)

    # fit model
    w = model_fn(y_train, tx_train, **kwargs)

    # predict
    y_pred_train = tx_train @ w
    y_pred_test = tx_test @ w

    # compute losses
    loss_tr = loss_fn(y_train, y_pred_train)
    loss_te = loss_fn(y_test, y_pred_test)

    return loss_tr, loss_te