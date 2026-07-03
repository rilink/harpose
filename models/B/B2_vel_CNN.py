"""
B2: CNN on SMPL axis-angle pose + velocity (PoseCNNVel).
Input: (6, 24, 50) — axis-angle pose (3 ch) + frame-to-frame velocity (3 ch), 24 joints, 50 frames at 30 Hz.
Architecture: Conv2D(6→32→64→128, k=3) + BN + ReLU + MaxPool → AdaptiveAvgPool → Linear(128, 4).
Training: 10 epochs, Adam lr=1e-3, ReduceLROnPlateau, class-weighted CE.
Runs LOSO twice: GT (poseT_*) and PRED (poseP_*).
"""

import os
import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score
from typing import Tuple, Dict, Any

NUM_JOINTS   = 24

ROOT_CORRECTION = Rotation.from_euler("x", 90, degrees=True).as_matrix().astype(np.float32)

DATA_PATH = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../data_pipeline/results_30hz_fused_acc_ori_gyr_pose/windowed_30hz_fused_acc_ori_gyr_pose_lw_rw_lp_rp_h.csv"))


# ── Pose + velocity features ──────────────────────────────────────────────────

def _parse_cell(cell):
    if isinstance(cell, list):
        return np.array(cell, dtype=np.float32)
    try:
        return np.array(ast.literal_eval(cell), dtype=np.float32)
    except Exception:
        return np.fromstring(cell.strip("[]"), sep=",", dtype=np.float32)


def rotmat_rows_to_axis_angle(df, pose_cols):
    """Returns (N, 3, 24, T) axis-angle, root-joint frame-corrected."""
    raw = []
    for _, row in df.iterrows():
        channels = np.stack([_parse_cell(row[c]) for c in pose_cols], axis=0)  # (216, T)
        T = channels.shape[1]
        rotmats = channels.reshape(NUM_JOINTS, 3, 3, T).transpose(3, 0, 1, 2)  # (T,24,3,3)
        raw.append(rotmats)
    rotmats_all = np.stack(raw, axis=0)  # (N,T,24,3,3)
    rotmats_all[:, :, 0] = np.einsum("ab,ntbc->ntac", ROOT_CORRECTION, rotmats_all[:, :, 0])
    N, T, V, _, _ = rotmats_all.shape
    aa = Rotation.from_matrix(rotmats_all.reshape(-1, 3, 3)).as_rotvec().reshape(N, T, V, 3)
    return aa.transpose(0, 3, 2, 1).astype(np.float32)  # (N, 3, 24, T)


def add_velocity_stream(aa):
    """
    aa: (N, 3, 24, T) — axis-angle pose.
    Returns (N, 6, 24, T) — pose + frame-to-frame angular velocity.
    Velocity at t=0 is set to zero (zero-padding at the left).
    """
    vel = np.diff(aa, axis=-1)                          # (N, 3, 24, T-1)
    vel = np.concatenate([np.zeros_like(vel[..., :1]), vel], axis=-1)  # (N, 3, 24, T)
    return np.concatenate([aa, vel], axis=1)            # (N, 6, 24, T)


# ── Model ─────────────────────────────────────────────────────────────────────

class PoseCNNVel(nn.Module):
    """
    CNN for SMPL pose + velocity classification.
    Input: (batch, 6, 24, T)  — 3 pose channels + 3 velocity channels.
    """
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout(0.5),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ── Dataset ───────────────────────────────────────────────────────────────────

class SMPLVelDataset(Dataset):
    def __init__(self, df: pd.DataFrame, features: np.ndarray):
        """features: (n, 6, 24, T)."""
        self.df = df.reset_index(drop=True)
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(self.df["activity_encoded"].values.astype(int) - 1, dtype=torch.long)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx], int(idx)


# ── LOSO splits ───────────────────────────────────────────────────────────────

def loso_splits(df):
    for subj in sorted(df["subject"].unique()):
        yield subj, df[df["subject"] != subj], df[df["subject"] == subj]


# ── Training ──────────────────────────────────────────────────────────────────

def train_cnn_model(
    train_dataset: SMPLVelDataset,
    device: torch.device,
    classes: int = 4,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[PoseCNNVel, Dict[str, Any]]:
    model = PoseCNNVel(num_classes=classes).to(device)

    n_total  = len(train_dataset)
    n_val    = int(n_total * val_fraction)
    n_train  = n_total - n_val
    g        = torch.Generator().manual_seed(seed)
    tr_set, val_set = random_split(train_dataset, [n_train, n_val], generator=g)

    tr_loader  = DataLoader(tr_set,  batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    labels        = train_dataset.y.numpy()
    class_counts  = np.bincount(labels, minlength=classes)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * classes
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_f1, best_state, best_ep = 0.0, None, 0
    for ep in range(epochs):
        model.train()
        tr_loss = 0.0
        for X, y, _ in tr_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()

        model.eval()
        yt, yp = [], []
        with torch.no_grad():
            for X, y, _ in val_loader:
                yp.extend(model(X.to(device)).argmax(1).cpu().numpy())
                yt.extend(y.numpy())

        vf1 = f1_score(yt, yp, average="macro", zero_division=0)
        scheduler.step(vf1)
        if vf1 > best_f1:
            best_f1 = vf1; best_state = model.state_dict(); best_ep = ep + 1
        print(f"  ep {ep+1:02d}/{epochs}  loss={tr_loss/max(1,len(tr_loader)):.4f}  val_f1={vf1:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_macro_f1": float(best_f1), "best_epoch": best_ep}


# ── Embedding extraction ───────────────────────────────────────────────────────

def extract_cnn_embeddings(model, data_loader, device):
    model.eval()
    embeddings, y_true, y_pred_cnn, indices = [], [], [], []
    logits_cnn_all = []

    with torch.no_grad():
        for X, y, idx in data_loader:
            X = X.to(device).float()
            logits = model(X)
            preds  = logits.argmax(1)
            feats  = model.features(X).view(X.size(0), -1)

            embeddings.append(feats.cpu().numpy())
            logits_cnn_all.append(logits.cpu().numpy())
            y_true.append(y.numpy())
            y_pred_cnn.append(preds.cpu().numpy())
            indices.append(idx.numpy())

    return (
        np.concatenate(embeddings),
        np.concatenate(y_true),
        np.concatenate(y_pred_cnn),
        np.concatenate(indices),
        np.concatenate(logits_cnn_all),
    )


# ── Main LOSO loop ─────────────────────────────────────────────────────────────

def run_loso_and_save(df, features_all, output_folder, device, label=""):
    os.makedirs(output_folder, exist_ok=True)
    fold_metrics = {}
    all_yt, all_yp = [], []

    for test_subj, train_df, test_df in loso_splits(df):
        print(f"\n=== [{label}] LOSO fold: subject {test_subj} held out ===")

        tr_ds = SMPLVelDataset(train_df, features_all[train_df.index.values])
        te_ds = SMPLVelDataset(test_df,  features_all[test_df.index.values])

        model, metrics = train_cnn_model(tr_ds, device)

        tr_loader = DataLoader(tr_ds, batch_size=64, shuffle=False)
        te_loader = DataLoader(te_ds, batch_size=64, shuffle=False)

        emb_tr, yt_tr, yp_tr, idx_tr, log_tr = extract_cnn_embeddings(model, tr_loader, device)
        emb_te, yt_te, yp_te, idx_te, log_te = extract_cnn_embeddings(model, te_loader, device)

        acc = accuracy_score(yt_te, yp_te)
        f1  = f1_score(yt_te, yp_te, average="macro", zero_division=0)
        print(f"  [{label}] test  acc={acc:.4f}  macro-f1={f1:.4f}")
        fold_metrics[test_subj] = {"acc": acc, "f1": f1}
        all_yt.extend(yt_te.tolist()); all_yp.extend(yp_te.tolist())

        num_classes = log_tr.shape[1]

        def _build_df(emb, yt, yp, idx, logits, split, src_df):
            out = pd.DataFrame(emb, columns=[f"emb_{i+1}" for i in range(emb.shape[1])])
            out["y_true"] = yt; out["y_pred_cnn"] = yp; out["idx"] = idx; out["split"] = split
            for c in range(num_classes):
                out[f"logits_cnn_{c}"] = logits[:, c]
            meta = src_df.reset_index(drop=True).iloc[idx][
                ["subject", "activity", "file_path", "window_idx"]].reset_index(drop=True)
            return pd.concat([meta, out.reset_index(drop=True)], axis=1)

        pd.concat([
            _build_df(emb_tr, yt_tr, yp_tr, idx_tr, log_tr, "train", train_df),
            _build_df(emb_te, yt_te, yp_te, idx_te, log_te, "test",  test_df),
        ]).to_csv(os.path.join(output_folder, f"fold_{test_subj}.csv"), index=False)

    g_acc = accuracy_score(all_yt, all_yp)
    g_f1  = f1_score(all_yt, all_yp, average="macro", zero_division=0)
    print(f"\n====== [{label}] GLOBAL AGGREGATED TEST ======")
    print(f"  acc={g_acc:.3f}  macro-f1={g_f1:.3f}")
    return fold_metrics, g_acc, g_f1


# ── Entry point ───────────────────────────────────────────────────────────────

def set_seed(seed=42):
    import random; random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


if __name__ == "__main__":
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = pd.read_csv(DATA_PATH)
    print(f"  {len(df)} rows loaded")

    poseT_cols = [f"poseT_{i}{j}_{joint}" for joint in range(NUM_JOINTS) for i in range(3) for j in range(3)]
    poseP_cols = [f"poseP_{i}{j}_{joint}" for joint in range(NUM_JOINTS) for i in range(3) for j in range(3)]

    results = {}
    for label, cols, out_tag in [
        ("GT",   poseT_cols, "B2_vel_CNN_gt_embeddings"),
        ("PRED", poseP_cols, "B2_vel_CNN_pred_embeddings"),
    ]:
        print(f"\n{'='*30} {label} POSE {'='*30}")
        print(f"Computing axis-angle + velocity ({label}) ...")
        aa = rotmat_rows_to_axis_angle(df, cols)          # (N, 3, 24, T)
        X  = add_velocity_stream(aa)                      # (N, 6, 24, T)
        print(f"  features shape: {X.shape}")
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), out_tag)
        fm, g_acc, g_f1 = run_loso_and_save(df, X, out_dir, device, label)
        results[label] = (g_acc, g_f1)

    print("\n========= FINAL SUMMARY (GT vs PRED pose) =========")
    print(f"  {'':4s}  {'acc':>8s}  {'macro-f1':>8s}")
    for lbl, (acc, f1) in results.items():
        print(f"  {lbl:4s}  {acc:8.3f}  {f1:8.3f}")
    print("====================================================\n")
