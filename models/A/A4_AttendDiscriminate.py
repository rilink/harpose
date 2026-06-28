"""
A4: Attend-and-Discriminate for IMU-based HAR.

Architecture:
  Input: (B, T, C)
  4× Conv2D(filter_num, (filter_size,1)) + ReLU   [(B,1,T,C) internally]
  SelfAttention per timestep across (filter_num × C) features
  GRU(filter_num*C → hidden_dim, enc_num_layers=2)
  TemporalAttention (softmax-weighted sum over time) → (B, hidden_dim)
  Dropout → Linear(hidden_dim, num_classes)

  Default: filter_num=64, hidden_dim=128, filter_size=5
  Embedding: 128-dim TemporalAttention output (pre-dropout)
"""

import ast
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset, random_split


class InertialDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        feature_cols = [
            c for c in df.columns
            if any(s in c for s in ["acc", "gyr", "ori"]) and "smpl" not in c
        ]
        self.meta = df.reset_index(drop=True)
        samples = []
        for _, row in self.meta.iterrows():
            channels = []
            for c in feature_cols:
                cell = row[c]
                if isinstance(cell, str):
                    try:
                        cell = ast.literal_eval(cell)
                    except Exception:
                        cell = np.fromstring(cell.strip("[]"), sep=",")
                channels.append(np.array(cell, dtype=np.float32))
            samples.append(np.stack(channels, axis=1))
        self.X = np.stack(samples, axis=0)
        self.labels = (df["activity_encoded"].values.astype(int) - 1).astype(int)
        self.window_size = self.X.shape[1]
        self.channels = self.X.shape[2]
        self.num_classes = int(np.unique(self.labels).shape[0])
        enc_to_name = df.groupby("activity_encoded")["activity"].first()
        self.class_names = [enc_to_name[enc] for enc in sorted(enc_to_name.index)]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
            int(idx),
        )


def instance_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True)
    return (x - mean) / (std + eps)


# ── Attend-and-Discriminate model ─────────────────────────────────────────────

def _conv1d(ni: int, no: int, ks: int = 1, padding: int = 0) -> nn.Conv1d:
    conv = nn.Conv1d(ni, no, ks, padding=padding, bias=False)
    nn.init.kaiming_normal_(conv.weight)
    return conv


class SelfAttention(nn.Module):
    def __init__(self, n_channels: int, div: int = 1):
        super().__init__()
        q_ch = n_channels // div if n_channels > 1 else n_channels
        self.query = _conv1d(n_channels, q_ch)
        self.key = _conv1d(n_channels, q_ch)
        self.value = _conv1d(n_channels, n_channels)
        self.gamma = nn.Parameter(torch.tensor([0.]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.size()
        x = x.view(*size[:2], -1)
        f, g, h = self.query(x), self.key(x), self.value(x)
        beta = F.softmax(torch.bmm(f.permute(0, 2, 1).contiguous(), g), dim=1)
        o = self.gamma * torch.bmm(h, beta) + x
        return o.view(*size).contiguous()


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 1)
        self.sm = nn.Softmax(dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, B, hidden_dim)
        out = self.fc(x).squeeze(2)                   # (T, B)
        weights_att = self.sm(out).unsqueeze(2)       # (T, B, 1)
        return torch.sum(weights_att * x, dim=0)      # (B, hidden_dim)


class FeatureExtractor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, filter_num: int,
                 filter_size: int, enc_num_layers: int, enc_is_bidirectional: bool,
                 dropout: float, dropout_rnn: float, sa_div: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(1, filter_num, (filter_size, 1))
        self.conv2 = nn.Conv2d(filter_num, filter_num, (filter_size, 1))
        self.conv3 = nn.Conv2d(filter_num, filter_num, (filter_size, 1))
        self.conv4 = nn.Conv2d(filter_num, filter_num, (filter_size, 1))
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.sa = SelfAttention(filter_num, sa_div)
        self.rnn = nn.GRU(
            filter_num * input_dim, hidden_dim,
            enc_num_layers,
            bidirectional=enc_is_bidirectional,
            dropout=dropout_rnn,
        )
        self.ta = TemporalAttention(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) → (B, 1, T, C)
        x = x.unsqueeze(1)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        # (B, filter_num, T', C)

        # self-attention per timestep
        refined = torch.cat(
            [self.sa(x[:, :, t, :].unsqueeze(3)) for t in range(x.shape[2])],
            dim=-1,
        )  # (B, filter_num, C, T')

        x = refined.permute(3, 0, 1, 2)               # (T', B, filter_num, C)
        x = x.reshape(x.shape[0], x.shape[1], -1)    # (T', B, filter_num*C)
        x = self.dropout(x)

        outputs, _ = self.rnn(x)                       # (T', B, hidden_dim)
        return self.ta(outputs)                        # (B, hidden_dim)


class AttendDiscriminateModel(nn.Module):
    def __init__(self, input_dim: int, num_classes: int,
                 hidden_dim: int = 128, filter_num: int = 64, filter_size: int = 5,
                 enc_num_layers: int = 2, enc_is_bidirectional: bool = False,
                 dropout: float = 0.5, dropout_rnn: float = 0.25,
                 dropout_cls: float = 0.5, sa_div: int = 1):
        super().__init__()
        self.fe = FeatureExtractor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            filter_num=filter_num,
            filter_size=filter_size,
            enc_num_layers=enc_num_layers,
            enc_is_bidirectional=enc_is_bidirectional,
            dropout=dropout,
            dropout_rnn=dropout_rnn,
            sa_div=sa_div,
        )
        self.dropout_cls = nn.Dropout(dropout_cls)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return self.fe(x)   # (B, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.get_embeddings(x)
        return self.classifier(self.dropout_cls(feat))


# ── Training infrastructure ────────────────────────────────────────────────────

def print_cm(yt, yp, class_names, title):
    cm = confusion_matrix(yt, yp, labels=list(range(len(class_names))))
    w = max(len(n) for n in class_names) + 1
    print(f"\n  {title} (rows=true, cols=pred):")
    header = " " * (w + 2) + "  ".join(f"{n:>{w}}" for n in class_names)
    print("  " + header)
    print("  " + "-" * len(header))
    for row, name in zip(cm, class_names):
        print("  " + f"{name:>{w}} |" + "  ".join(f"{v:>{w}}" for v in row))


def loso_splits(data_path: str):
    df = pd.read_csv(data_path)
    for subj in sorted(df["subject"].unique()):
        yield (
            subj,
            df[df["subject"] != subj].reset_index(drop=True),
            df[df["subject"] == subj].reset_index(drop=True),
        )


def _set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(train_ds, device, num_classes, make_model,
          epochs=25, batch_size=32, lr=3e-4, val_fraction=0.1, seed=42):
    _set_seed(seed)
    n_val = int(len(train_ds) * val_fraction)
    n_train = len(train_ds) - n_val
    g = torch.Generator().manual_seed(seed)
    tr_set, val_set = random_split(train_ds, [n_train, n_val], generator=g)
    tr_loader = DataLoader(tr_set, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = make_model(train_ds.channels, train_ds.window_size, num_classes)

    counts = np.bincount(train_ds.labels, minlength=num_classes).astype(float)
    w = torch.tensor(1.0 / (counts + 1e-6), dtype=torch.float32).to(device)
    w = w / w.sum() * num_classes
    criterion = nn.CrossEntropyLoss(weight=w)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5)

    best_f1, best_state, best_epoch = 0.0, None, 0
    best_val_yt, best_val_yp = [], []

    for epoch in range(epochs):
        model.train()
        tr_yt, tr_yp = [], []
        for x, y, _ in tr_loader:
            x = instance_norm(x.to(device))
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            criterion(logits, y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_yp.extend(logits.argmax(1).detach().cpu().numpy())
            tr_yt.extend(y.cpu().numpy())
        scheduler.step()
        tr_acc = accuracy_score(tr_yt, tr_yp)
        tr_f1 = f1_score(tr_yt, tr_yp, average="macro", zero_division=0)

        model.eval()
        val_yt, val_yp = [], []
        with torch.no_grad():
            for x, y, _ in val_loader:
                x = instance_norm(x.to(device))
                val_yp.extend(model(x).argmax(1).cpu().numpy())
                val_yt.extend(y.numpy())
        val_f1 = f1_score(val_yt, val_yp, average="macro", zero_division=0)
        val_acc = accuracy_score(val_yt, val_yp)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            best_val_yt = list(val_yt)
            best_val_yp = list(val_yp)

        print(f"  ep {epoch+1:02d}/{epochs}  "
              f"tr_acc={tr_acc:.3f}  tr_f1={tr_f1:.3f}  |  "
              f"val_acc={val_acc:.3f}  val_f1={val_f1:.3f}")

    if best_state:
        model.load_state_dict(best_state)
    print(f"  best epoch={best_epoch}  best_val_f1={best_f1:.4f}")
    return model, best_val_yt, best_val_yp


@torch.no_grad()
def extract(model, loader, device):
    model.eval()
    embs, yt, yp, idxs, logits_all = [], [], [], [], []
    for x, y, idx in loader:
        xn = instance_norm(x.to(device))
        z = model.get_embeddings(xn)
        logits = model(xn)
        embs.append(z.cpu().numpy())
        yt.append(y.numpy())
        yp.append(logits.argmax(1).cpu().numpy())
        idxs.append(idx.numpy())
        logits_all.append(logits.cpu().numpy())
    return (
        np.concatenate(embs),
        np.concatenate(yt),
        np.concatenate(yp),
        np.concatenate(idxs),
        np.concatenate(logits_all),
    )


def run_loso(data_path, output_folder, device, make_model, model_tag,
             epochs=25, seed=42):
    os.makedirs(output_folder, exist_ok=True)
    fold_metrics = {}
    all_yt, all_yp = [], []
    global_class_names = None

    for test_subj, train_df, test_df in loso_splits(data_path):
        print(f"\n=== LOSO fold: subject {test_subj} held out ===")

        train_ds = InertialDataset(train_df)
        num_classes = train_ds.num_classes
        class_names = train_ds.class_names
        if global_class_names is None:
            global_class_names = class_names

        model, best_val_yt, best_val_yp = train(
            train_ds, device, num_classes, make_model, epochs=epochs, seed=seed)

        test_ds = InertialDataset(test_df)
        tr_loader = DataLoader(train_ds, batch_size=64, shuffle=False)
        te_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

        emb_tr, yt_tr, yp_tr, idx_tr, log_tr = extract(model, tr_loader, device)
        emb_te, yt_te, yp_te, idx_te, log_te = extract(model, te_loader, device)

        te_acc = accuracy_score(yt_te, yp_te)
        te_f1 = f1_score(yt_te, yp_te, average="macro", zero_division=0)
        print(f"  test  acc={te_acc:.4f}  macro-f1={te_f1:.4f}")
        print_cm(best_val_yt, best_val_yp, class_names, "Val confusion matrix (best epoch)")
        print_cm(yt_te, yp_te, class_names, "Test confusion matrix")

        all_yt.extend(yt_te.tolist())
        all_yp.extend(yp_te.tolist())

        def _make_df(emb, yt, yp, idx, logits, split, meta_df):
            df_out = pd.DataFrame(emb, columns=[f"emb_{i+1}" for i in range(emb.shape[1])])
            for i in range(logits.shape[1]):
                df_out[f"logits_{model_tag}_{i}"] = logits[:, i]
            df_out["y_true"] = yt
            df_out[f"y_pred_{model_tag}"] = yp
            df_out["idx"] = idx
            df_out["split"] = split
            meta = meta_df.reset_index(drop=True).iloc[idx][
                ["subject", "activity", "file_path", "window_idx"]
            ].reset_index(drop=True)
            return pd.concat([meta, df_out.reset_index(drop=True)], axis=1)

        combined = pd.concat([
            _make_df(emb_tr, yt_tr, yp_tr, idx_tr, log_tr, "train", train_df),
            _make_df(emb_te, yt_te, yp_te, idx_te, log_te, "test", test_df),
        ], ignore_index=True)
        combined.to_csv(
            os.path.join(output_folder, f"fold_{test_subj}.csv"), index=False)
        fold_metrics[test_subj] = {"acc": te_acc, "f1": te_f1}

    accs = np.array([v["acc"] for v in fold_metrics.values()])
    f1s = np.array([v["f1"] for v in fold_metrics.values()])
    print("\n====== LOSO SUMMARY (per-fold) ======")
    print(f"Accuracy:  {accs.mean():.3f} ± {accs.std():.3f}")
    print(f"Macro-F1:  {f1s.mean():.3f} ± {f1s.std():.3f}")
    global_acc = accuracy_score(all_yt, all_yp)
    global_f1 = f1_score(all_yt, all_yp, average="macro", zero_division=0)
    print("\n====== GLOBAL AGGREGATED TEST (all folds pooled) ======")
    print(f"Accuracy:  {global_acc:.3f}")
    print(f"Macro-F1:  {global_f1:.3f}")
    print_cm(all_yt, all_yp, global_class_names, "Global test confusion matrix")
    return fold_metrics


if __name__ == "__main__":
    _set_seed(42)
    DATA_PATH = "../../data_pipeline/results_30hz_fused_acc_ori_gyr_pose/windowed_30hz_fused_acc_ori_gyr_pose_lw_rw_lp_rp_h.csv"
    OUTPUT_FOLDER = "A4_AttendDiscriminate_embeddings"
    MODEL_TAG = "attend"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    def make_model(channels, window_size, num_classes):
        # filter_num=32 (vs paper default 64) because our 75-channel data makes
        # GRU input = filter_num*channels; 64*75=4800 is impractically large on CPU.
        return AttendDiscriminateModel(
            input_dim=channels,
            num_classes=num_classes,
            hidden_dim=128,
            filter_num=32,
            filter_size=5,
            enc_num_layers=2,
            enc_is_bidirectional=False,
            dropout=0.5,
            dropout_rnn=0.25,
            dropout_cls=0.5,
            sa_div=1,
        ).to(device)

    run_loso(DATA_PATH, OUTPUT_FOLDER, device, make_model, MODEL_TAG, epochs=25, seed=42)
