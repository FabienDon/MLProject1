import numpy as np

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


import numpy as np

def cross_validation(y, x, k_indices, k, lambda_, degree):
    """Return the train and test RMSE of ridge regression for a fold."""
    
    # 1. split data
    test_idx = k_indices[k]
    train_idx = k_indices[np.arange(len(k_indices)) != k].reshape(-1)
    
    y_train, y_test = y[train_idx], y[test_idx]
    x_train, x_test = x[train_idx], x[test_idx]
    
    # 2. form polynomial features
    tx_train = build_poly(x_train, degree)
    tx_test = build_poly(x_test, degree)
    
    # 3. ridge regression
    w, mse_tr = ridge_regression(y_train, tx_train, lambda_)
    
    # 4. compute loss (RMSE)
    e_test = y_test - tx_test @ w
    mse_te = (e_test.T @ e_test) / (2 * len(y_test))
    
    rmse_tr = np.sqrt(2 * mse_tr)
    rmse_te = np.sqrt(2 * mse_te)
    
    return rmse_tr, rmse_te

