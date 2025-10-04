"""Some helper functions for project 1."""

import csv
import numpy as np
import os
import matplotlib.pyplot as plt



def load_csv_data(data_path, sub_sample=False, keep_cols=None):
    """
    Load CSV data and return:
      - dict of training features (feature_name -> np.array)
      - dict of test features (feature_name -> np.array)
      - y_train labels
      - train_ids
      - test_ids
    """
    # --- Read header (for feature names) ---
    with open(os.path.join(data_path, "x_train.csv"), "r") as f:
        header = f.readline().strip().split(",")
    feature_names = header[1:]  # drop the first column ("Id")

    # --- Load arrays ---
    y_train = np.genfromtxt(
        os.path.join(data_path, "y_train.csv"),
        delimiter=",",
        skip_header=1,
        dtype=int,
        usecols=1,
    )
    x_train = np.genfromtxt(
        os.path.join(data_path, "x_train.csv"), delimiter=",", skip_header=1
    )
    x_test = np.genfromtxt(
        os.path.join(data_path, "x_test.csv"), delimiter=",", skip_header=1
    )

    train_ids = x_train[:, 0].astype(int)
    test_ids = x_test[:, 0].astype(int)
    x_train = x_train[:, 1:]
    x_test = x_test[:, 1:]

    # --- Keep only selected columns (if specified) ---
    if keep_cols is not None:
        x_train = x_train[:, keep_cols]
        x_test = x_test[:, keep_cols]
        feature_names = [feature_names[i] for i in keep_cols]

    # --- Sub-sample ---
    if sub_sample:
        y_train = y_train[::50]
        x_train = x_train[::50]
        train_ids = train_ids[::50]

    # --- Build dictionaries ---
    train_dict = {name: x_train[:, i] for i, name in enumerate(feature_names)}
    test_dict = {name: x_test[:, i] for i, name in enumerate(feature_names)}

    return train_dict, test_dict, y_train, train_ids, test_ids

def filter_features_by_nan(train_dict, test_dict, threshold=70.0):
    """
    Drop features from dictionary if percentage of NaNs exceeds threshold.
    Returns a new dictionary with kept features.
    """
    n_rows = len(next(iter(train_dict.values())))  # number of samples
    keep = []

    for name, values in train_dict.items():
        nan_pct = np.isnan(values).sum() / n_rows * 100
        if nan_pct <= threshold:
            keep.append(name)

        train_clean_dict = {k: train_dict[k] for k in keep if k in train_dict}
        test_clean_dict = {k: test_dict[k] for k in keep if k in train_dict}

    print("Keeping", len(keep), "features")
    return train_clean_dict, test_clean_dict

def drop_and_keep_features(train_dict, test_dict, fields_to_drop):
    """
    Remove unwanted features and keep only the intersection
    of keep_names - fields_to_drop.
    """
    # Final set of features to keep

    keep_names = train_dict.keys()

    final_keep = [name for name in keep_names if name not in fields_to_drop]

    # Build cleaned dictionaries
    train_clean = {name: train_dict[name] for name in final_keep}
    test_clean  = {name: test_dict[name] for name in final_keep}

    return train_clean, test_clean


def replace_invalid_with_nan_inplace(dicts, invalid_dict):
    """
    Replace invalid values with np.nan for features listed in invalid_dict,
    applied to one or more feature dictionaries.
    
    Parameters
    ----------
    dicts : list[dict[str, np.ndarray]] or dict[str, np.ndarray]
        One or more dictionaries mapping feature name -> numpy array of values.
    invalid_dict : dict[tuple[int], list[str]]
        Mapping of invalid values -> list of feature names.
    """
    # allow single dict input
    if isinstance(dicts, dict):
        dicts = [dicts]

    for invalid_values, feature_list in invalid_dict.items():
        for name in feature_list:
            for d in dicts:
                if name in d:
                    values = d[name]
                    mask = np.isin(values, invalid_values)
                    values[mask] = np.nan
                    d[name] = values



def create_csv_submission(ids, y_pred, name):
    """
    This function creates a csv file named 'name' in the format required for a submission in Kaggle or AIcrowd.
    The file will contain two columns the first with 'ids' and the second with 'y_pred'.
    y_pred must be a list or np.array of 1 and -1 otherwise the function will raise a ValueError.

    Args:
        ids (list,np.array): indices
        y_pred (list,np.array): predictions on data correspondent to indices
        name (str): name of the file to be created
    """
    # Check that y_pred only contains -1 and 1
    if not all(i in [-1, 1] for i in y_pred):
        raise ValueError("y_pred can only contain values -1, 1")

    with open(name, "w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})

def distribution_plotter(y_train, train_dict):

    # count how many samples in each class
    n_pos = np.sum(y_train == 1)
    idx_pos = np.where(y_train == 1)[0]
    idx_neg = np.where(y_train == -1)[0]

    # random subsample negatives
    np.random.seed(42)
    idx_neg_sampled = np.random.choice(idx_neg, size=n_pos, replace=False)

    # number of features
    n_features = len(train_dict)
    n_cols = 4
    n_rows = int(np.ceil(n_features / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*4))

    for i, (feature_name, values) in enumerate(train_dict.items()):
        ax = axes.flat[i]
    
        vals_pos = values[idx_pos]
        vals_neg = values[idx_neg_sampled]
    
        ax.hist(vals_pos, bins='auto', alpha=0.5, label="y = 1")
        ax.hist(vals_neg, bins='auto', alpha=0.5, label="y = -1 (sampled)")
    
        ax.set_title(feature_name)
        ax.set_xlabel(feature_name)
        ax.set_ylabel("Count")
        ax.legend()

    # Hide empty subplots (if number of features not multiple of 4)
    for j in range(i+1, n_rows*n_cols):
        fig.delaxes(axes.flat[j])

    plt.tight_layout()
    plt.show()

def winsorizor(data_dict,features,percentile = 95):

    # winsorizor for continuous feautures

    lower = 100 - percentile
    upper = percentile
    
    for feature in features:
        if feature in data_dict:
            arr = data_dict[feature]
            lo = np.nanpercentile(arr, lower)
            hi = np.nanpercentile(arr, upper)
            arr = np.clip(arr, lo, hi)
            data_dict[feature] = arr

def categorical_nan_filler(data_dict, continuous_features):

    new_data_dict = {}

    for feature, arr in data_dict.items():
        if feature in continuous_features:
            # keep continuous as is
            new_data_dict[feature] = arr
            continue

        # convert to object for safe handling
        arr = arr.astype(object)

        # replace NaN with a string "NaN" to keep as category
        arr_safe = np.array(
            ["NaN" if (x is None or (isinstance(x, float) and np.isnan(x))) else str(x)
             for x in arr],
            dtype=object
        )

        # unique categories
        values = np.unique(arr_safe)

        # create one-hot columns
        for val in values:
            new_key = f"{feature}_{val}"
            new_data_dict[new_key] = (arr_safe == val).astype(int)

    return new_data_dict


def continuous_nan_filler(data_dict, continuous_feautures):

    print('xxx')

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

def cross_validation(y, x, k_fold,method):

    k_indices = build_k_indices(y, k_fold, seed=1)

    train_losses = []
    test_losses = []

    for k in range(k_fold):
        # Split train/test indices
        test_idx = k_indices[k]
        train_idx = k_indices[np.arange(k_fold) != k].reshape(-1)

        x_train, y_train = x[train_idx], y[train_idx]

        minority_idx = np.where(y_train == 1)[0]
        majority_idx = np.where(y_train == -1)[0]

        if method == 'undersample':

            n_minority = len(minority_idx)

            # Randomly undersample majority to match minority size
            sampled_majority = np.random.choice(majority_idx, size=n_minority, replace=False)

            # Combine balanced indices
            balanced_idx = np.concatenate([minority_idx, sampled_majority])

            x_train, y_train = x_train[balanced_idx], y_train[balanced_idx]

        elif method == 'oversample':

            n_minority = len(minority_idx)
            n_majority = len(majority_idx)

            sampled_minority= np.random.choice(minority_idx, size=n_majority, replace=True)

            balanced_idx = np.concatenate([majority_idx, sampled_minority])
            x_train_balanced, y_train_balanced = x_train[balanced_idx], y_train[balanced_idx]

        else:

            # need to do for balanced logistic regression #
            print('xxx')

        x_test, y_test   = x[test_idx], y[test_idx]


