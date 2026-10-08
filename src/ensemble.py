import numpy as np
from sklearn.metrics import log_loss

def normalize_weights(weights):
    s = sum(weights.values())
    if s <= 0:
        return {k: 1/len(weights) for k in weights}
    return {k: v/s for k, v in weights.items()}

def learn_weights(y, model_probs):
    # 简单、稳定、可解释的验证集权重：
    # 先计算每个模型的验证 LogLoss，再按 1/loss 归一化。
    scores = {}
    for name, p in model_probs.items():
        scores[name] = log_loss(y, p, labels=list(range(10)))
    raw = {name: 1.0/max(loss, 1e-6) for name, loss in scores.items()}
    return normalize_weights(raw), scores

def blend(probs, weights):
    names = [n for n in probs if n in weights]
    out = sum(probs[n] * weights[n] for n in names)
    out = np.clip(out, 1e-12, None)
    return out / out.sum(axis=1, keepdims=True)
