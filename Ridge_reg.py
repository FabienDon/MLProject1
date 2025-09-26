import numpy as np

def  ridge_regression(y,tx,lambda_):
    """Computes the ridge regression solution.

    Args:
        y: shape=(N, )
        tx: shape=(N,d)
        lamda: float

    Returns:
        w: shape=(d,), the optimal weights
    """
    N = y.shape[0]
    d = tx.shape[1]

    XTX = tx.transpose()@tx
    lambda_prime = lambda_/(2*N)

    w = np.linalg.solve(XTX + lambda_prime*np.identity(d), tx.transpose()@y)
    return w