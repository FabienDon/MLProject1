import numpy as np

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

def precision(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp)

def recall(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn + 1e-8)

def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * (p * r) / (p + r + 1e-8)

def cross_validate(model, X, y, k=5, random_state=100):
    """
    Perform k-fold cross-validation.

    Parameters:
    ----------
    model : object with .fit(X, y) and .predict(X)
        Custom ML model (must implement fit and predict).
    X : numpy array
        Feature matrix.
    y : numpy array
        Target vector.
    k : int
        Number of folds.
    random_state : int
        Seed for shuffling.

    Returns:
    -------
    dict with mean accuracy, precision, recall, f1
    """

    np.random.seed(random_state)
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    
    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[:n_samples % k] += 1
    
    current = 0
    accuracies, precisions, recalls, f1s = [], [], [], []
    
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        val_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        # train and predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        # collect metrics
        accuracies.append(accuracy(y_val, y_pred))
        precisions.append(precision(y_val, y_pred))
        recalls.append(recall(y_val, y_pred))
        f1s.append(f1_score(y_val, y_pred))
        
        current = stop
    
    return {
        "accuracy": np.mean(accuracies),
        "precision": np.mean(precisions),
        "recall": np.mean(recalls),
        "f1": np.mean(f1s)
    }
