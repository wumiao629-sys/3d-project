import numpy as np
from sklearn.isotonic import IsotonicRegression

class MulticlassIsotonic:
    def __init__(self):
        self.models = [IsotonicRegression(out_of_bounds="clip") for _ in range(10)]

    def fit(self, y, p):
        y = np.asarray(y, dtype=int)
        p = np.asarray(p, dtype=float)
        for c in range(10):
            target = (y == c).astype(float)
            self.models[c].fit(p[:, c], target)
        return self

    def transform(self, p):
        p = np.asarray(p, dtype=float)
        q = np.column_stack([m.predict(p[:, c]) for c, m in enumerate(self.models)])
        q = np.clip(q, 1e-8, None)
        q /= q.sum(axis=1, keepdims=True)
        return q
