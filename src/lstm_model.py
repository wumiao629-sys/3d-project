import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.15):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0, batch_first=True
        )
        self.heads = nn.ModuleList([nn.Linear(hidden_size, 10) for _ in range(3)])

    def forward(self, x):
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        return [head(h) for head in self.heads]

def train_lstm(X, y, cfg):
    p = cfg["models"]["lstm"]
    seq_len = p["sequence_length"]
    if len(X) < p["min_rows"] or len(X) <= seq_len + 20:
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg["models"]["random_seed"])

    xa = np.asarray(X, dtype=np.float32)
    ya = np.asarray(y[["target_h","target_t","target_u"]], dtype=np.int64)
    xs, ys = [], []
    for i in range(seq_len, len(xa)):
        xs.append(xa[i-seq_len:i])
        ys.append(ya[i])
    xs = torch.tensor(np.asarray(xs), dtype=torch.float32)
    ys = torch.tensor(np.asarray(ys), dtype=torch.long)

    ds = TensorDataset(xs, ys)
    loader = DataLoader(ds, batch_size=p["batch_size"], shuffle=False)
    model = LSTMNet(xs.shape[-1], p["hidden_size"], p["num_layers"], p["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=p["learning_rate"])
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(p["epochs"]):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            outs = model(xb)
            loss = sum(loss_fn(outs[j], yb[:, j]) for j in range(3))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    return model, device, seq_len

def lstm_predict(model_pack, X):
    if model_pack is None:
        return None
    model, device, seq_len = model_pack
    xa = np.asarray(X, dtype=np.float32)
    if len(xa) < seq_len:
        return None
    inp = torch.tensor(xa[-seq_len:][None, ...], dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        outs = model(inp)
        return [torch.softmax(o, dim=1).cpu().numpy()[0] for o in outs]
