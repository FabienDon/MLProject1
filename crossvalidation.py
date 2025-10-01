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

def cross_validation(y, x, k_fold, model_fn, degree, loss_fn, **kwargs):
    """
    Perform k-fold cross-validation and report mean + standard deviation of the loss.

    Args:
        y:         numpy array, shape=(N,)
        x:         numpy array, shape=(N,)
        k_fold:    int, number of folds
        model_fn:  function(y_train, tx_train, **kwargs) -> w
        degree:    int, polynomial degree for build_poly
        loss_fn:   function(y, y_pred) -> scalar loss
        kwargs:    additional args for model_fn (e.g., lambda_ for ridge)

    Returns:
        mean_train_loss, std_train_loss, mean_test_loss, std_test_loss
    """

    k_indices = build_k_indices(y, k_fold, seed=1)

    train_losses = []
    test_losses = []

    for k in range(k_fold):
        # Split train/test indices
        test_idx = k_indices[k]
        train_idx = k_indices[np.arange(k_fold) != k].reshape(-1)

        x_train, y_train = x[train_idx], y[train_idx]
        x_test, y_test   = x[test_idx], y[test_idx]

        # Polynomial expansion
        tx_train = build_poly(x_train, degree)
        tx_test  = build_poly(x_test, degree)

        # Train model
        w = model_fn(y_train, tx_train, **kwargs)

        # Predictions
        y_pred_train = tx_train @ w
        y_pred_test  = tx_test @ w

        # Compute loss
        train_losses.append(loss_fn(y_train, y_pred_train))
        test_losses.append(loss_fn(y_test, y_pred_test))

    # Compute mean and standard deviation
    mean_train_loss = np.mean(train_losses)
    std_train_loss  = np.std(train_losses)
    mean_test_loss  = np.mean(test_losses)
    std_test_loss   = np.std(test_losses)

    return mean_train_loss, std_train_loss, mean_test_loss, std_test_loss
