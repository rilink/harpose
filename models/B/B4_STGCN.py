"""
B4: ST-GCN on FK joint positions + velocity.

Forward kinematics -> root-relative 3D joint positions, then frame-to-frame
velocity, giving a 6-channel (position + velocity) feature per joint. Graph
convolution over the SMPL kinematic tree (SMPL_EDGES adjacency), 6->24->32->32
channels, dropout=0.6, AdamW lr=3e-4 weight_decay=1e-3, CosineAnnealingLR,
25 epochs.

Reads only the poseT_*/poseP_* columns of the fused acc/ori/gyr+pose CSV
(acc/ori/gyr columns are ignored). Runs LOSO TWICE: once on ground-truth
pose (poseT_*) and once on MobilePoser-predicted pose (poseP_*), reporting
both sets of results separately and side by side.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from B3_TCN_FK import load_body_model, rotmat_rows_to_joint_positions, NUM_JOINTS

IN_CHANNELS = 6  # 3D position + 3D velocity

SMPL_EDGES = [
    (1, 0), (2, 0), (3, 0),
    (4, 1), (5, 2), (6, 3),
    (7, 4), (8, 5), (9, 6),
    (10, 7), (11, 8), (12, 9),
    (13, 9), (14, 9),
    (15, 12), (16, 13), (17, 14),
    (18, 16), (19, 17),
    (20, 18), (21, 19),
    (22, 20), (23, 21),
]


def build_adjacency(num_nodes=NUM_JOINTS, edges=SMPL_EDGES):
    A = np.eye(num_nodes, dtype=np.float32)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    d = A.sum(axis=1) ** -0.5
    return (np.diag(d) @ A @ np.diag(d)).astype(np.float32)


def joint_positions_to_node_features(pos):
    """
    pos: (N, T, 24, 3) root-relative joint positions.
    Returns (N, T, 24, 6): position + frame-to-frame velocity per joint.
    """
    vel = np.zeros_like(pos)
    vel[:, 1:] = pos[:, 1:] - pos[:, :-1]
    return np.concatenate([pos, vel], axis=-1).astype(np.float32)


# ── Dataset ───────────────────────────────────────────────────────────────────

class FKGraphDataset(Dataset):
    def __init__(self, df: pd.DataFrame, features: np.ndarray):
        """features: (n, T, 24, 6), row-aligned with df (same order)."""
        self.df = df.reset_index(drop=True)
        self.X = features.astype(np.float32)
        self.labels = (self.df["activity_encoded"].values.astype(int) - 1).astype(int)
        self.num_classes = int(np.unique(self.labels).shape[0])
        enc_to_name = self.df.groupby("activity_encoded")["activity"].first()
        self.class_names = [enc_to_name[enc] for enc in sorted(enc_to_name.index)]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx].transpose(2, 0, 1), dtype=torch.float32)  # (6, T, 24)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y, int(idx)


# ── Model ─────────────────────────────────────────────────────────────────────

class SpatialGraphConv(nn.Module):
    def __init__(self, in_ch, out_ch, A):
        super().__init__()
        self.register_buffer("A", torch.tensor(A))
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x = torch.einsum("vj,bctj->bctv", self.A, x)
        return self.conv(x)


class STGCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, A, t_kernel=9, stride=1, dropout=0.6):
        super().__init__()
        self.gcn    = SpatialGraphConv(in_ch, out_ch, A)
        self.bn_gcn = nn.BatchNorm2d(out_ch)
        pad = (t_kernel - 1) // 2
        self.tcn = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, (t_kernel, 1), (stride, 1), (pad, 0)),
            nn.BatchNorm2d(out_ch),
            nn.Dropout(dropout),
        )
        self.residual = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, (stride, 1)),
                nn.BatchNorm2d(out_ch),
            )
            if in_ch != out_ch or stride != 1 else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.residual(x)
        x   = self.relu(self.bn_gcn(self.gcn(x)))
        x   = self.tcn(x)
        return self.relu(x + res)


class STGCN(nn.Module):
    """6->24->32->32, emb=32, dropout=0.6 (~27K params)."""
    def __init__(self, num_classes, A, in_ch=IN_CHANNELS, dropout=0.6):
        super().__init__()
        self.bn_input = nn.BatchNorm2d(in_ch)
        self.blocks   = nn.Sequential(
            STGCNBlock(in_ch, 24, A, stride=1, dropout=dropout),
            STGCNBlock(24,    32, A, stride=2, dropout=dropout),
            STGCNBlock(32,    32, A, stride=1, dropout=dropout),
        )
        self.pool       = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(32, num_classes)

    def get_embeddings(self, x):
        x = self.bn_input(x)
        x = self.blocks(x)
        return self.pool(x).flatten(1)

    def forward(self, x):
        return self.classifier(self.dropout(self.get_embeddings(x)))


# ── Confusion matrix display ───────────────────────────────────────────────────

def print_cm(yt, yp, class_names, title):
    cm = confusion_matrix(yt, yp, labels=list(range(len(class_names))))
    w  = max(len(n) for n in class_names) + 1
    print(f"\n  {title} (rows=true, cols=pred):")
    header = " " * (w + 2) + "  ".join(f"{n:>{w}}" for n in class_names)
    print("  " + header)
    print("  " + "-" * len(header))
    for row, name in zip(cm, class_names):
        print("  " + f"{name:>{w}} |" + "  ".join(f"{v:>{w}}" for v in row))


# ── LOSO splits ───────────────────────────────────────────────────────────────

def loso_splits(df):
    for subj in sorted(df["subject"].unique()):
        yield subj, df[df["subject"] != subj], df[df["subject"] == subj]


def _set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Training ──────────────────────────────────────────────────────────────────

def train(train_ds, device, num_classes, A, epochs=25, batch_size=32,
          lr=3e-4, val_fraction=0.1, seed=42):
    _set_seed(seed)

    n_val   = int(len(train_ds) * val_fraction)
    n_train = len(train_ds) - n_val
    g = torch.Generator().manual_seed(seed)
    tr_set, val_set = random_split(train_ds, [n_train, n_val], generator=g)

    tr_loader  = DataLoader(tr_set,  batch_size=batch_size, shuffle=True,  generator=g)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = STGCN(num_classes=num_classes, A=A).to(device)

    counts = np.bincount(train_ds.labels, minlength=num_classes).astype(float)
    w = torch.tensor(1.0 / (counts + 1e-6), dtype=torch.float32).to(device)
    w = w / w.sum() * num_classes

    criterion = nn.CrossEntropyLoss(weight=w)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_f1, best_state, best_epoch = 0.0, None, 0
    best_val_yt, best_val_yp = [], []

    for epoch in range(epochs):
        model.train()
        tr_yt, tr_yp = [], []
        for x, y, _ in tr_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            tr_yp.extend(logits.argmax(1).detach().cpu().numpy())
            tr_yt.extend(y.cpu().numpy())
        scheduler.step()

        tr_acc = accuracy_score(tr_yt, tr_yp)
        tr_f1  = f1_score(tr_yt, tr_yp, average="macro", zero_division=0)

        model.eval()
        val_yt, val_yp = [], []
        with torch.no_grad():
            for x, y, _ in val_loader:
                val_yp.extend(model(x.to(device)).argmax(1).cpu().numpy())
                val_yt.extend(y.numpy())
        val_f1  = f1_score(val_yt, val_yp, average="macro", zero_division=0) if val_yt else 0.0
        val_acc = accuracy_score(val_yt, val_yp) if val_yt else 0.0

        if val_f1 > best_f1:
            best_f1, best_state, best_epoch = val_f1, {k: v.clone() for k, v in model.state_dict().items()}, epoch + 1
            best_val_yt, best_val_yp = list(val_yt), list(val_yp)

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
    embs, yt, yp, idxs = [], [], [], []
    for x, y, idx in loader:
        x = x.to(device)
        z = model.get_embeddings(x)
        embs.append(z.cpu().numpy())
        yt.append(y.numpy())
        yp.append(model.classifier(model.dropout(z)).argmax(1).cpu().numpy())
        idxs.append(idx.numpy())
    return (np.concatenate(embs), np.concatenate(yt),
            np.concatenate(yp),   np.concatenate(idxs))


# ── Main LOSO loop ────────────────────────────────────────────────────────────

def run_loso(df, features_all, output_folder, device, A, epochs=25, seed=42, label=""):
    os.makedirs(output_folder, exist_ok=True)
    fold_metrics = {}
    all_yt, all_yp = [], []
    global_class_names = None

    for test_subj, train_df, test_df in loso_splits(df):
        print(f"\n=== [{label}] LOSO fold: subject {test_subj} held out ===")

        train_ds    = FKGraphDataset(train_df, features_all[train_df.index.values])
        num_classes = train_ds.num_classes
        class_names = train_ds.class_names
        if global_class_names is None:
            global_class_names = class_names

        model, best_val_yt, best_val_yp = train(train_ds, device, num_classes, A, epochs=epochs, seed=seed)

        test_ds   = FKGraphDataset(test_df, features_all[test_df.index.values])
        tr_loader = DataLoader(train_ds, batch_size=64, shuffle=False)
        te_loader = DataLoader(test_ds,  batch_size=64, shuffle=False)

        emb_tr, yt_tr, yp_tr, idx_tr = extract(model, tr_loader, device)
        emb_te, yt_te, yp_te, idx_te = extract(model, te_loader, device)

        te_acc = accuracy_score(yt_te, yp_te)
        te_f1  = f1_score(yt_te, yp_te, average="macro", zero_division=0)
        print(f"  [{label}] test  acc={te_acc:.4f}  macro-f1={te_f1:.4f}")
        print_cm(best_val_yt, best_val_yp, class_names, f"[{label}] Val confusion matrix (best epoch)")
        print_cm(yt_te,       yp_te,       class_names, f"[{label}] Test confusion matrix")

        all_yt.extend(yt_te.tolist())
        all_yp.extend(yp_te.tolist())

        def _make_df(emb, yt, yp, idx, split, meta_df):
            df_out = pd.DataFrame(emb, columns=[f"emb_{i+1}" for i in range(emb.shape[1])])
            df_out["y_true"]       = yt
            df_out["y_pred_stgcn"] = yp
            df_out["idx"]          = idx
            df_out["split"]        = split
            meta = meta_df.reset_index(drop=True).loc[
                idx, ["subject", "activity", "file_path", "window_idx"]
            ].reset_index(drop=True)
            return pd.concat([meta, df_out.reset_index(drop=True)], axis=1)

        combined = pd.concat([
            _make_df(emb_tr, yt_tr, yp_tr, idx_tr, "train", train_df),
            _make_df(emb_te, yt_te, yp_te, idx_te, "test",  test_df),
        ], ignore_index=True)
        combined.to_csv(os.path.join(output_folder, f"fold_{test_subj}.csv"), index=False)
        fold_metrics[test_subj] = {"acc": te_acc, "f1": te_f1}

    accs = np.array([v["acc"] for v in fold_metrics.values()])
    f1s  = np.array([v["f1"]  for v in fold_metrics.values()])
    print(f"\n====== [{label}] LOSO SUMMARY (per-fold) ======")
    print(f"Accuracy:  {accs.mean():.3f} +/- {accs.std():.3f}")
    print(f"Macro-F1:  {f1s.mean():.3f} +/- {f1s.std():.3f}")

    global_acc = accuracy_score(all_yt, all_yp)
    global_f1  = f1_score(all_yt, all_yp, average="macro", zero_division=0)
    print(f"\n====== [{label}] GLOBAL AGGREGATED TEST (all folds pooled) ======")
    print(f"Accuracy:  {global_acc:.3f}")
    print(f"Macro-F1:  {global_f1:.3f}")
    print_cm(all_yt, all_yp, global_class_names, f"[{label}] Global test confusion matrix")

    return fold_metrics, global_acc, global_f1


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _set_seed(42)
    DATA_PATH = "../../data_pipeline/results_30hz_fused_acc_ori_gyr_pose/windowed_30hz_fused_acc_ori_gyr_pose_lw_rw_lp_rp_h.csv"
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    A = build_adjacency()

    print("Loading SMPL body model for FK ...")
    body_model = load_body_model()

    print(f"Loading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    df = df.reset_index(drop=True)
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    poseT_cols = [f"poseT_{i}{j}_{joint}" for joint in range(NUM_JOINTS) for i in range(3) for j in range(3)]
    poseP_cols = [f"poseP_{i}{j}_{joint}" for joint in range(NUM_JOINTS) for i in range(3) for j in range(3)]

    results = {}
    for label, cols, out_tag in [
        ("GT",   poseT_cols, "B4_STGCN_gt_embeddings"),
        ("PRED", poseP_cols, "B4_STGCN_pred_embeddings"),
    ]:
        print(f"\n{'='*30} {label} POSE {'='*30}")
        print(f"Running FK ({label}) ...")
        pos = rotmat_rows_to_joint_positions(df, cols, body_model)        # (N,T,24,3)
        feats = joint_positions_to_node_features(pos)                    # (N,T,24,6)
        print(f"  feature shape: {feats.shape}")
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), out_tag)
        fold_metrics, g_acc, g_f1 = run_loso(df, feats, out_dir, device, A, epochs=25, seed=42, label=label)
        results[label] = (g_acc, g_f1)

    print("\n========= FINAL SUMMARY =========")
    print(f"  {'':4s}  {'acc':>8s}  {'macro-f1':>8s}")
    for lbl, (acc, f1) in results.items():
        print(f"  {lbl:4s}  {acc:8.3f}  {f1:8.3f}")
    print("==================================\n")
