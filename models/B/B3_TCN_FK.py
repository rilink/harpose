"""
B3: Dilated TCN on FK joint positions + velocity (TCN-FK).
Input: (144, 50) — root-relative 3D positions (72 ch) + velocity (72 ch), 50 frames at 30 Hz.
Architecture: 4× ResBlock(128, d=1,2,4,8) → GlobalAvgPool → Linear(128, 4).
Training: 30 epochs, AdamW lr=3e-4, CosineAnnealingLR, class-weighted CE.
Runs LOSO twice: GT (poseT_*) and PRED (poseP_*).
"""

import os
import sys
import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score

NUM_JOINTS = 24

ROOT_CORRECTION = Rotation.from_euler("x", 90, degrees=True).as_matrix().astype(np.float32)

SMPL_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "SMPL_tools", "smpl", "basicmodel_m.pkl")
SMPL_TOOLS_DIR  = os.path.join(os.path.dirname(__file__), "..", "..", "SMPL_tools")

DATA_PATH = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../data_pipeline/results_30hz_fused_acc_ori_gyr_pose/"
    "windowed_30hz_fused_acc_ori_gyr_pose_lw_rw_lp_rp_h.csv"))


# ── SMPL body model (forward kinematics) ─────────────────────────────────────

def _patch_numpy_for_chumpy():
    for name, val in [("bool", bool), ("int", int), ("float", float),
                      ("complex", complex), ("object", object), ("str", str), ("unicode", str)]:
        if not hasattr(np, name):
            setattr(np, name, val)


def load_body_model():
    _patch_numpy_for_chumpy()
    if SMPL_TOOLS_DIR not in sys.path:
        sys.path.insert(0, SMPL_TOOLS_DIR)
    from articulate.model import ParametricModel
    return ParametricModel(SMPL_MODEL_PATH)


# ── Pose parsing ──────────────────────────────────────────────────────────────

def _parse_cell(cell):
    if isinstance(cell, list):
        return np.array(cell, dtype=np.float32)
    try:
        return np.array(ast.literal_eval(cell), dtype=np.float32)
    except Exception:
        return np.fromstring(cell.strip("[]"), sep=",", dtype=np.float32)


def rotmat_rows_to_joint_positions(df, pose_cols, body_model, batch_size=20000):
    """Returns (N, T, 24, 3) root-relative FK joint positions."""
    raw = []
    for _, row in df.iterrows():
        channels = np.stack([_parse_cell(row[c]) for c in pose_cols], axis=0)  # (216, T)
        T = channels.shape[1]
        rotmats = channels.reshape(NUM_JOINTS, 3, 3, T).transpose(3, 0, 1, 2)  # (T,24,3,3)
        raw.append(rotmats)
    rotmats_all = np.stack(raw, axis=0)  # (N,T,24,3,3)
    rotmats_all[:, :, 0] = np.einsum("ab,ntbc->ntac", ROOT_CORRECTION, rotmats_all[:, :, 0])
    N, T, V, _, _ = rotmats_all.shape
    pose = torch.from_numpy(rotmats_all.reshape(-1, V, 3, 3)).float()
    joints = []
    with torch.no_grad():
        for i in range(0, pose.shape[0], batch_size):
            _, j = body_model.forward_kinematics(pose[i:i + batch_size], shape=None, tran=None, calc_mesh=False)
            joints.append(j.numpy())
    return np.concatenate(joints, axis=0).reshape(N, T, V, 3).astype(np.float32)


def joint_positions_to_tcn_input(pos):
    """
    pos: (N, T, 24, 3) root-relative joint positions.
    Returns (N, 144, T): positions + velocities, channels-first for Conv1d.
    """
    vel = np.zeros_like(pos)
    vel[:, 1:] = pos[:, 1:] - pos[:, :-1]
    feat = np.concatenate([pos, vel], axis=-1)   # (N, T, 24, 6)
    N, T = feat.shape[:2]
    return feat.reshape(N, T, -1).transpose(0, 2, 1).astype(np.float32)  # (N, 144, T)


# ── Model ─────────────────────────────────────────────────────────────────────

class TCNResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, dilation=1, dropout=0.2):
        super().__init__()
        pad = (kernel - 1) * dilation // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel, dilation=dilation, padding=pad),
            nn.BatchNorm1d(out_ch),
        )
        self.shortcut = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.net(x) + self.shortcut(x))


class TemporalCNN_FK(nn.Module):
    """
    Lightweight dilated TCN over concatenated FK joint position + velocity.
    Input: (B, 144, T=50).  Embedding: 128-dim global-avg-pooled vector.
    """
    def __init__(self, in_ch=144, emb_dim=128, num_classes=4, dropout=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            TCNResBlock(in_ch,   emb_dim, dilation=1, dropout=dropout),
            TCNResBlock(emb_dim, emb_dim, dilation=2, dropout=dropout),
            TCNResBlock(emb_dim, emb_dim, dilation=4, dropout=dropout),
            TCNResBlock(emb_dim, emb_dim, dilation=8, dropout=dropout),
        )
        self.pool       = nn.AdaptiveAvgPool1d(1)
        self.dropout    = nn.Dropout(0.5)
        self.classifier = nn.Linear(emb_dim, num_classes)

    def get_embeddings(self, x):
        x = self.encoder(x)
        x = self.pool(x).squeeze(-1)
        return x

    def forward(self, x):
        return self.classifier(self.dropout(self.get_embeddings(x)))


# ── Dataset ───────────────────────────────────────────────────────────────────

class FKDataset(Dataset):
    def __init__(self, df, features):
        """features: (N, 144, T) — already channels-first."""
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(df["activity_encoded"].values.astype(int) - 1, dtype=torch.long)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx], int(idx)


# ── Training ──────────────────────────────────────────────────────────────────

def _class_weights(labels, num_classes, device):
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    w = torch.tensor(1.0 / (counts + 1e-6), dtype=torch.float32, device=device)
    return w / w.sum() * num_classes


def train_model(train_ds, device, num_classes=4,
                epochs=30, batch_size=32, lr=3e-4, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    n_val = int(len(train_ds) * 0.1)
    g = torch.Generator().manual_seed(seed)
    tr_set, val_set = random_split(train_ds, [len(train_ds) - n_val, n_val], generator=g)

    tr_loader  = DataLoader(tr_set,  batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model  = TemporalCNN_FK(num_classes=num_classes).to(device)
    crit   = nn.CrossEntropyLoss(weight=_class_weights(train_ds.y.numpy(), num_classes, device))
    opt    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    best_f1, best_state = 0.0, None
    for ep in range(epochs):
        model.train()
        for x, y, _ in tr_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()
        sched.step()

        model.eval()
        yt, yp = [], []
        with torch.no_grad():
            for x, y, _ in val_loader:
                yp.extend(model(x.to(device)).argmax(1).cpu().numpy())
                yt.extend(y.numpy())
        vf1 = f1_score(yt, yp, average="macro", zero_division=0)
        if vf1 > best_f1:
            best_f1 = vf1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"  ep {ep+1:02d}/{epochs}  val_f1={vf1:.4f}  best={best_f1:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def extract(model, ds, device, batch_size=128):
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    embs, yt, yp, idxs, logits_all = [], [], [], [], []
    for x, y, idx in loader:
        x = x.to(device)
        z  = model.get_embeddings(x)
        lg = model.classifier(model.dropout(z))
        embs.append(z.cpu().numpy())
        logits_all.append(lg.cpu().numpy())
        yt.append(y.numpy())
        yp.append(lg.argmax(1).cpu().numpy())
        idxs.append(idx.numpy())
    return (np.concatenate(embs), np.concatenate(yt), np.concatenate(yp),
            np.concatenate(idxs), np.concatenate(logits_all))


# ── Main LOSO loop ────────────────────────────────────────────────────────────

def run_loso(df, features_all, output_folder, device, label):
    os.makedirs(output_folder, exist_ok=True)
    fold_metrics, all_yt, all_yp = {}, [], []

    for test_subj in sorted(df["subject"].unique()):
        print(f"\n=== [{label}] LOSO fold: subject {test_subj} held out ===")
        tr_df = df[df["subject"] != test_subj]
        te_df = df[df["subject"] == test_subj]
        tr_ds = FKDataset(tr_df, features_all[tr_df.index.values])
        te_ds = FKDataset(te_df, features_all[te_df.index.values])

        model = train_model(tr_ds, device)

        emb_tr, yt_tr, yp_tr, idx_tr, log_tr = extract(model, tr_ds, device)
        emb_te, yt_te, yp_te, idx_te, log_te = extract(model, te_ds, device)

        acc = accuracy_score(yt_te, yp_te)
        f1  = f1_score(yt_te, yp_te, average="macro", zero_division=0)
        print(f"  [{label}] test: acc={acc:.4f}  macro-f1={f1:.4f}")
        fold_metrics[test_subj] = {"acc": acc, "f1": f1}
        all_yt.extend(yt_te.tolist()); all_yp.extend(yp_te.tolist())

        num_classes = log_tr.shape[1]

        def _build(emb, yt, yp, idx, logits, split, src_df):
            out = pd.DataFrame(emb, columns=[f"emb_{i+1}" for i in range(emb.shape[1])])
            out["y_true"] = yt; out["y_pred_tcn"] = yp; out["idx"] = idx; out["split"] = split
            for c in range(num_classes):
                out[f"logits_tcn_{c}"] = logits[:, c]
            meta = src_df.reset_index(drop=True).iloc[idx][
                ["subject", "activity", "file_path", "window_idx"]].reset_index(drop=True)
            return pd.concat([meta, out.reset_index(drop=True)], axis=1)

        pd.concat([
            _build(emb_tr, yt_tr, yp_tr, idx_tr, log_tr, "train", tr_df),
            _build(emb_te, yt_te, yp_te, idx_te, log_te, "test",  te_df),
        ]).to_csv(os.path.join(output_folder, f"fold_{test_subj}.csv"), index=False)

    g_acc = accuracy_score(all_yt, all_yp)
    g_f1  = f1_score(all_yt, all_yp, average="macro", zero_division=0)
    accs  = np.array([v["acc"] for v in fold_metrics.values()])
    f1s   = np.array([v["f1"]  for v in fold_metrics.values()])
    print(f"\n====== [{label}] LOSO SUMMARY ======")
    print(f"  per-fold:  acc={accs.mean():.3f}+/-{accs.std():.3f}  macro-f1={f1s.mean():.3f}+/-{f1s.std():.3f}")
    print(f"  pooled:    acc={g_acc:.3f}  macro-f1={g_f1:.3f}")
    return g_acc, g_f1


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random; random.seed(42); np.random.seed(42); torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading SMPL body model for FK ...")
    body_model = load_body_model()

    df = pd.read_csv(DATA_PATH)
    df = df.reset_index(drop=True)
    print(f"  {len(df)} windows loaded")

    poseT_cols = [f"poseT_{i}{j}_{joint}" for joint in range(NUM_JOINTS) for i in range(3) for j in range(3)]
    poseP_cols = [f"poseP_{i}{j}_{joint}" for joint in range(NUM_JOINTS) for i in range(3) for j in range(3)]

    results = {}
    for label, cols, out_tag in [
        ("GT",   poseT_cols, "B3_TCN_FK_gt_embeddings"),
        ("PRED", poseP_cols, "B3_TCN_FK_pred_embeddings"),
    ]:
        print(f"\n{'='*30} {label} POSE {'='*30}")
        print(f"Running FK ({label}) ...")
        pos = rotmat_rows_to_joint_positions(df, cols, body_model)   # (N,T,24,3)
        X   = joint_positions_to_tcn_input(pos)                      # (N,144,T)
        print(f"  TCN input shape: {X.shape}")
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), out_tag)
        g_acc, g_f1 = run_loso(df, X, out_dir, device, label)
        results[label] = (g_acc, g_f1)

    print("\n========= FINAL SUMMARY =========")
    print(f"  {'':4s}  {'acc':>8s}  {'macro-f1':>8s}")
    for lbl, (acc, f1) in results.items():
        print(f"  {lbl:4s}  {acc:8.3f}  {f1:8.3f}")
    print("==================================\n")
