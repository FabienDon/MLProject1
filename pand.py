import os
import numpy as np

class FastFrame:
    def __init__(self, data, columns=None, index=None):
        self.data = np.array(data)
        if columns is None:
            columns = [f"col{i}" for i in range(self.data.shape[1])]
        self.columns = np.array(columns)
        if index is None:
            index = np.arange(self.data.shape[0])
        self.index = np.array(index)

    def __repr__(self):
        rows = min(5, len(self.data))
        preview = [dict(zip(self.columns, row)) for row in self.data[:rows]]
        return f"FastFrame({preview} ... {len(self.data)} rows, {len(self.columns)} cols)"

    def __getitem__(self, key):
        if key in self.columns:
            col_idx = np.where(self.columns == key)[0][0]
            return self.data[:, col_idx]
        else:
            raise KeyError(f"Column {key} not found")

    def filter(self, mask):
        return FastFrame(self.data[mask], self.columns, self.index[mask])

    def row(self, i):
        return dict(zip(self.columns, self.data[i]))

    def head(self, n=5):
        return FastFrame(self.data[:n], self.columns, self.index[:n])


