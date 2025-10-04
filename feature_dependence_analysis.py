import numpy as np

# Load data
X_data = np.genfromtxt("./data/dataset/x_train.csv", delimiter=",", skip_header=1, dtype=np.float32)
with open("./data/dataset/x_train.csv", "r") as f:
    feature_names = f.readline().strip().split(",")

# Pearson correlation
X = np.nan_to_num(X_data, nan=np.nanmean(X_data, axis=0))  # replace NaN with column mean
X_centered = X - np.mean(X, axis=0)
X_std = np.std(X_centered, axis=0, ddof=0)
X_scaled = X_centered / X_std
pearson_matrix = np.corrcoef(X_scaled, rowvar=False)

# Spearman correlation
def rankdata_vec(a):
    ranks = np.empty_like(a, dtype=float)
    for i in range(a.shape[1]):
        col = a[:, i]
        sorter = np.argsort(col)
        inv = np.empty_like(sorter)
        inv[sorter] = np.arange(len(col))
        arr = col[sorter]
        r = np.zeros_like(col, dtype=float)
        start = 0
        while start < len(arr):
            end = start + 1
            while end < len(arr) and arr[end] == arr[start]:
                end += 1
            r[start:end] = (start + end - 1) / 2.0 + 1
            start = end
        ranks[:, i] = r[inv]
    return ranks

X_ranked = rankdata_vec(X)
spearman_matrix = np.corrcoef(X_ranked, rowvar=False)

# Which correlations do we keep ?
threshold = 0.9

def extract_strong_pairs(matrix, names, threshold):
    strong_pairs = []
    n = matrix.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            if abs(matrix[i,j]) >= threshold:
                strong_pairs.append((names[i], names[j], matrix[i,j]))
    return strong_pairs

strong_pearson = extract_strong_pairs(pearson_matrix, feature_names, threshold)
strong_spearman = extract_strong_pairs(spearman_matrix, feature_names, threshold)

# Save results
with open("strong_pearson_features.csv", "w") as f:
    f.write("Feature1,Feature2,Pearson_r\n")
    for f1, f2, r in strong_pearson:
        f.write(f"{f1},{f2},{r:.6f}\n")

with open("strong_spearman_features.csv", "w") as f:
    f.write("Feature1,Feature2,Spearman_r\n")
    for f1, f2, r in strong_spearman:
        f.write(f"{f1},{f2},{r:.6f}\n")

print("Strongly correlated features saved to 'strong_pearson_features.csv' and 'strong_spearman_features.csv'")