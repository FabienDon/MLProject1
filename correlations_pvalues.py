import numpy as np
from math import sqrt, erf

# --- Load data ---
X_data = np.genfromtxt("./data/dataset/x_train.csv", delimiter=",", skip_header=1, dtype=np.float32)
with open("./data/dataset/x_train.csv", "r") as f:
    feature_names = f.readline().strip().split(",")

y_raw = np.genfromtxt("./data/dataset/y_train.csv", delimiter=",", skip_header=1, dtype=np.float32)
y_data = y_raw[:, 1].flatten()  # second column = labels

# --- Helper ---
def normal_cdf(x):
    """Approximate standard normal CDF using the error function."""
    return 0.5 * (1 + erf(x / sqrt(2)))

def corr_pvalues(x, y):
    """Compute Pearson correlation and approximate p-value, ignoring NaNs in x."""
    mask = ~np.isnan(x)
    x_valid = x[mask]
    y_valid = y[mask]

    n = len(x_valid)
    if n < 3:
        return np.nan, np.nan

    x_mean = np.mean(x_valid)
    y_mean = np.mean(y_valid)
    numerator = np.sum((x_valid - x_mean) * (y_valid - y_mean))
    denominator = sqrt(np.sum((x_valid - x_mean)**2) * np.sum((y_valid - y_mean)**2))
    if denominator == 0:
        return 0.0, np.nan

    r = numerator / denominator

    # --- p-value using t-distribution approximation ---
    if abs(r) == 1:
        p = 0.0
    else:
        t = abs(r) * sqrt((n - 2) / (1 - r**2))
        # For df > 30, t ~ normal
        p = max(2 * (1 - normal_cdf(t)), 1e-300)

    return r, p

# --- Compute correlations ---
results = {}
for i, name in enumerate(feature_names):
    col = X_data[:, i]
    r, p = corr_pvalues(col, y_data)
    results[name] = (r, p)

# --- Save results ---
with open("./correlations.txt", "w") as f:
    f.write("Feature\tCorrelation\tP-value\n")
    for name, (r, p) in results.items():
        f.write(f"{name}\t{r:.6f}\t{p:.6e}\n")

print("Correlations and p-values saved to correlations.txt")

# --- Helper: rank data (average ranks for ties) ---
def rankdata(a):
    """Return ranks (1-based), averaging ties."""
    a = np.asarray(a)
    sorter = np.argsort(a)
    inv = np.empty_like(sorter)
    inv[sorter] = np.arange(len(a))
    arr = a[sorter]
    ranks = np.zeros_like(a, dtype=float)
    start = 0
    while start < len(arr):
        end = start + 1
        while end < len(arr) and arr[end] == arr[start]:
            end += 1
        ranks[start:end] = (start + end - 1) / 2.0 + 1  # average rank
        start = end
    return ranks[inv]

# --- Compute Spearman correlations ---
spearman_results = {}
for i, name in enumerate(feature_names):
    col = X_data[:, i]
    col_ranks = rankdata(col)
    y_ranks = rankdata(y_data)
    r, p = corr_pvalues(col_ranks, y_ranks)
    spearman_results[name] = (r, p)

# --- Save Spearman results ---
with open("./spearman_correlations.txt", "w") as f:
    f.write("Feature\tSpearman_r\tP-value\n")
    for name, (r, p) in spearman_results.items():
        f.write(f"{name}\t{r:.6f}\t{p:.6e}\n")

print("Spearman correlations and p-values saved to spearman_correlations.txt")
