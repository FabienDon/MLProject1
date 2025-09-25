"""Some helper functions for project 1."""

import csv
import numpy as np
import os

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

def filter_features_by_nan(features_dict, threshold=70.0):
    """
    Drop features from dictionary if percentage of NaNs exceeds threshold.
    Returns a new dictionary with kept features.
    """
    n_rows = len(next(iter(features_dict.values())))  # number of samples
    keep = {}

    for name, values in features_dict.items():
        nan_pct = np.isnan(values).sum() / n_rows * 100
        if nan_pct <= threshold:
            keep[name] = values

    print("Keeping", len(keep), "features")
    return keep

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

    return train_clean, test_clean, final_keep


def replace_invalid_with_nan_inplace(features_dict, invalid_dict):
    """
    Replace invalid values with np.nan for features listed in invalid_dict.
    
    Parameters
    ----------
    features_dict : dict[str, np.ndarray]
        Dictionary mapping feature name -> numpy array of values.
    invalid_dict : dict[tuple[int], list[str]]
        Mapping of invalid values -> list of feature names.
    """
    for invalid_values, feature_list in invalid_dict.items():
        for name in feature_list:
            if name in features_dict:
                values = features_dict[name]
                # make sure it's float so it can hold NaN
                if not np.issubdtype(values.dtype, np.floating):
                    values = values.astype(float, copy=False)
                mask = np.isin(values, invalid_values)
                values[mask] = np.nan
                features_dict[name] = values



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
