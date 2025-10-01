import numpy as np

X_data = np.genfromtxt("./data/dataset/x_train.csv", delimiter=",", skip_header=1, dtype=np.float32)
with open("./data/dataset/x_train.csv", "r") as f:
    feature_names = f.readline().strip().split(",")

y_raw = np.genfromtxt("./data/dataset/y_train.csv", delimiter=",", skip_header=1, dtype=np.float32)
y_data = y_raw[:, 1].flatten()  # second column = labels
y_data = np.where(y_data == -1, 0, y_data)  # convert -1 -> 0

def corr_ignore_nan(x, y):
    """
    Computes correlation between x and y, ignoring NaN values in x.
    y is assumed to have no NaNs.
    """
    mask = ~np.isnan(x)
    x_valid = x[mask]
    y_valid = y[mask]
    
    if len(x_valid) == 0:
        return np.nan
    x_mean = np.mean(x_valid)
    y_mean = np.mean(y_valid)
    numerator = np.sum((x_valid - x_mean) * (y_valid - y_mean))
    denominator = np.sqrt(np.sum((x_valid - x_mean)**2) * np.sum((y_valid - y_mean)**2))
    if denominator == 0:
        return 0
    return numerator / denominator

correlations = {}
for i, name in enumerate(feature_names):
    col = X_data[:, i]
    correlations[name] = corr_ignore_nan(col, y_data)

with open("./correlations.txt", "w") as f:
    f.write("Feature\tCorrelation\n")
    for name, c in correlations.items():
        f.write(f"{name}\t{c:.4f}\n")

print("Correlations saved to correlations.txt")
