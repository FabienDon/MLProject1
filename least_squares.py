import numpy as np

def least_squares(y, tx):
    """Computes the least-squares solution.

    Args:
        y: shape=(N, )
        tx: shape=(N,d)

    Returns:
        w: shape=(d,), the optimal weights that minimize the MSE loss.
    """
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    return w
