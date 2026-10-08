import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss

def multiclass_brier(y, p):
    y = np.asarray(y, dtype=int)
    onehot = np.eye(10)[y]
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))

def topk_accuracy(y, p, k):
    top = np.argsort(p, axis=1)[:, -k:]
    return float(np.mean([yy in row for yy, row in zip(y, top)]))

def position_metrics(y, p):
    return {
        "logloss": float(log_loss(y, p, labels=list(range(10)))),
        "brier": multiclass_brier(y, p),
        "top1": topk_accuracy(y, p, 1),
        "top3": topk_accuracy(y, p, 3),
        "top5": topk_accuracy(y, p, 5),
    }

def number_candidates(ph, pt, pu, top_n=20):
    rows = []
    for h in range(10):
        for t in range(10):
            for u in range(10):
                prob = float(ph[h] * pt[t] * pu[u])
                rows.append({"number": f"{h}{t}{u}", "probability": prob})
    return pd.DataFrame(rows).sort_values("probability", ascending=False).head(top_n).reset_index(drop=True)

def full_number_hit(actual, ph, pt, pu, top_n):
    cand = number_candidates(ph, pt, pu, top_n)
    return int(actual in set(cand["number"]))
