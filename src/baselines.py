import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

def uniform_probs(n):
    return np.full((n, 10), 0.1)

def rolling_frequency_probs(history, col, window):
    vals = history[col].astype(int).to_numpy()
    out = np.zeros((len(vals), 10), dtype=float)
    for i in range(len(vals)):
        start = max(0, i - window)
        hist = vals[start:i]
        counts = np.bincount(hist, minlength=10) if len(hist) else np.ones(10)
        out[i] = (counts + 1.0) / (counts.sum() + 10.0)
    return out

def evaluate_baseline(y_true, probs):
    return log_loss(y_true, probs, labels=list(range(10)))
