"""
B1: CNN on SMPL pose parameters (PoseCNN).

Architecture/training: Conv2d 3->32->64->128, BatchNorm+ReLU+MaxPool+Dropout2d
blocks, AdaptiveAvgPool2d, Adam lr=1e-3, ReduceLROnPlateau on val macro-F1,
10 epochs, class-weighted CrossEntropyLoss, 10% random val split.

Reads only the poseT_*/poseP_* columns of the fused acc/ori/gyr+pose CSV
(acc/ori/gyr columns are ignored); 30Hz, 50-frame windows, 24x3x3
rotation-matrix poseT_*/poseP_* taken directly from the MobilePoser .pt
files. poseT_*/poseP_* rotation matrices are converted to
axis-angle (after root-joint frame correction) for the CNN input
representation (3, 24, T). Runs LOSO TWICE: once on ground-truth pose
(poseT_*) and once on MobilePoser-predicted pose (poseP_*), reporting both
sets of results separately and side by side.

Root-joint frame correction
----------------------------
poseT_*/poseP_* joint 0 (pelvis/root) is stored in MobilePoser's native Y-up
(Xsens/DIP) frame, while joints 1-23 use AMASS's Z-up frame (verified
empirically: joints 1-23 match smpl_* to float32 precision; joint 0 differs
by R_x(-90 deg)). Before converting to axis-angle, R_x(+90 deg) is applied
to joint 0's rotation matrix to bring the whole skeleton back into AMASS
frame - the same correction applied to the root joint/translation in
`models/B/B4_STGCN.py`.
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

# Brings MobilePoser's root-joint Y-up (Xsens/DIP) frame back to AMASS's Z-up frame.
ROOT_CORRECTION = Rotation.from_euler("x", 90, degrees=True).as_matrix().astype(np.float32)


# ── rotmat rows -> axis-angle (3, 24, T) features ──────────────────────────────

def _parse_cell(cell):
    if isinstance(cell, list):
        return np.array(cell, dtype=np.float32)
    try:
        return np.array(ast.literal_eval(cell), dtype=np.float32)
    except Exception:
        return np.fromstring(cell.strip("[]"), sep=",", dtype=np.float32)


def rotmat_rows_to_axis_angle(df, pose_cols):
    """
    df rows of 216 rotation-matrix columns (joint-major order: pose*_{i}{j}_{joint}
    for joint in 0..23, i,j in 0..2; each cell = list of T values) ->
    (N, 3, 24, T) axis-angle, root-joint frame-corrected from MobilePoser's
    Y-up to AMASS's Z-up, in the same (channels, joints, time) layout
    PoseCNN expects.
    """
    raw = []
    for _, row in df.iterrows():
        channels = np.stack([_parse_cell(row[c]) for c in pose_cols], axis=0)  # (216, T)
        T = channels.shape[1]
        rotmats = channels.reshape(NUM_JOINTS, 3, 3, T).transpose(3, 0, 1, 2)  # (T, 24, 3, 3)
        raw.append(rotmats)

    rotmats_all = np.stack(raw, axis=0)  # (N, T, 24, 3, 3)
    rotmats_all[:, :, 0, :, :] = np.einsum("ab,ntbc->ntac", ROOT_CORRECTION, rotmats_all[:, :, 0, :, :])

    N, T, V, _, _ = rotmats_all.shape
    aa = Rotation.from_matrix(rotmats_all.reshape(-1, 3, 3)).as_rotvec().reshape(N, T, V, 3)
    return aa.transpose(0, 3, 2, 1).astype(np.float32)  # (N, 3, 24, T)


# ── Model ─────────────────────────────────────────────────────────────────────

class PoseCNN(nn.Module):
    """
    CNN for SMPL pose classification.
    Input: (batch, 3, 24, window_length)
    """
    def __init__(self, num_classes):
        super(PoseCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
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

class SMPLDataset(Dataset):
    def __init__(self, df: pd.DataFrame, features: np.ndarray):
        """features: (n, 3, 24, T), row-aligned with df (same order)."""
        self.df = df.reset_index(drop=True)
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(self.df["activity_encoded"].values.astype(int) - 1, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], int(idx)


# ── LOSO splits ───────────────────────────────────────────────────────────────

def loso_splits(df):
    for subj in sorted(df["subject"].unique()):
        yield subj, df[df["subject"] != subj], df[df["subject"] == subj]


# ── Training ──────────────────────────────────────────────────────────────────

def train_cnn_model(
    train_dataset: SMPLDataset,
    device: torch.device,
    classes: int = 4,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_fraction: float = 0.1,
    seed: int = 42
) -> Tuple[PoseCNN, Dict[str, Any]]:
    model = PoseCNN(num_classes=classes).to(device)

    n_total = len(train_dataset)
    n_val = int(n_total * val_fraction)
    n_train = n_total - n_val
    generator = torch.Generator().manual_seed(seed)

    train_set, val_set = random_split(train_dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    labels = train_dataset.y.numpy()
    class_counts = np.bincount(labels, minlength=classes)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_macro_f1 = 0.0
    best_state = None
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for X, y, _ in train_loader:
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_loss = train_loss / max(1, len(train_loader))

        model.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for X, y, _ in val_loader:
                X = X.to(device)
                y = y.to(device)

                logits = model(X)
                preds = torch.argmax(logits, dim=1)

                y_true.extend(y.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        macro_f1 = f1_score(y_true, y_pred, average='macro') if len(y_true) > 0 else 0.0
        scheduler.step(macro_f1)

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_state = model.state_dict()
            best_epoch = epoch + 1

        print(f"  ep {epoch+1:02d}/{epochs}  loss={avg_loss:.4f}  val_f1={macro_f1:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {"best_macro_f1": float(best_macro_f1), "best_epoch": best_epoch}


# ── Embedding/logit extraction ─────────────────────────────────────────────────

def extract_cnn_embeddings(model, data_loader, device):
    model.eval()

    embeddings, y_true, y_pred_emb, y_pred_cnn, indices = [], [], [], [], []
    logits_cnn_all, logits_emb_all = [], []

    with torch.no_grad():
        for X, y, idx in data_loader:
            X = X.to(device).float()

            logits_cnn = model(X)
            preds_cnn = torch.argmax(logits_cnn, dim=1)
            logits_cnn_all.append(logits_cnn.cpu().numpy())

            feats = model.features(X)
            feats = feats.view(feats.size(0), -1)
            embeddings.append(feats.cpu().numpy())

            logits_emb = model.classifier(feats)
            preds_emb = torch.argmax(logits_emb, dim=1)
            logits_emb_all.append(logits_emb.cpu().numpy())

            y_true.append(y.numpy())
            y_pred_emb.append(preds_emb.cpu().numpy())
            y_pred_cnn.append(preds_cnn.cpu().numpy())
            indices.append(idx.numpy())

    return (
        np.concatenate(embeddings),
        np.concatenate(y_true),
        np.concatenate(y_pred_emb),
        np.concatenate(y_pred_cnn),
        np.concatenate(indices),
        np.concatenate(logits_cnn_all),
        np.concatenate(logits_emb_all),
    )


# ── Main LOSO loop ────────────────────────────────────────────────────────────

def run_loso_and_save_cnn_embeddings(df, features_all, output_folder, device, label=""):
    os.makedirs(output_folder, exist_ok=True)
    fold_metrics = {}
    all_yt, all_yp = [], []

    for test_subj, train_df, test_df in loso_splits(df):
        print(f"\n=== [{label}] LOSO fold: subject {test_subj} held out ===")

        train_dataset_full = SMPLDataset(train_df, features_all[train_df.index.values])
        test_dataset       = SMPLDataset(test_df,  features_all[test_df.index.values])

        model, metrics = train_cnn_model(
            train_dataset=train_dataset_full,
            device=device,
            epochs=10,
            batch_size=32,
            lr=1e-3,
            val_fraction=0.1,
            seed=42,
        )

        train_loader_emb = DataLoader(train_dataset_full, batch_size=64, shuffle=False)
        test_loader_emb  = DataLoader(test_dataset, batch_size=64, shuffle=False)

        emb_train, y_true_train, y_pred_emb_train, y_pred_cnn_train, idx_train, logits_cnn_train, logits_emb_train = \
            extract_cnn_embeddings(model, train_loader_emb, device)

        emb_test, y_true_test, y_pred_emb_test, y_pred_cnn_test, idx_test, logits_cnn_test, logits_emb_test = \
            extract_cnn_embeddings(model, test_loader_emb, device)

        emb_test_acc      = accuracy_score(y_true_test, y_pred_emb_test)
        emb_test_macro_f1 = f1_score(y_true_test, y_pred_emb_test, average='macro')

        cnn_test_acc      = accuracy_score(y_true_test, y_pred_cnn_test)
        cnn_test_macro_f1 = f1_score(y_true_test, y_pred_cnn_test, average='macro')

        print(f"  [{label}] test  acc={cnn_test_acc:.4f}  macro-f1={cnn_test_macro_f1:.4f}")

        metrics['emb_test_accuracy']   = float(emb_test_acc)
        metrics['emb_test_macro_f1']   = float(emb_test_macro_f1)
        metrics['cnn_test_accuracy']   = float(cnn_test_acc)
        metrics['cnn_test_macro_f1']   = float(cnn_test_macro_f1)

        all_yt.extend(y_true_test.tolist())
        all_yp.extend(y_pred_cnn_test.tolist())

        emb_train_df = pd.DataFrame(emb_train)
        emb_test_df  = pd.DataFrame(emb_test)

        emb_train_df.columns = [f"emb_{i+1}" for i in range(emb_train_df.shape[1])]
        emb_test_df.columns  = [f"emb_{i+1}" for i in range(emb_test_df.shape[1])]

        num_classes = logits_cnn_train.shape[1]

        emb_train_df['y_true'] = y_true_train
        emb_train_df['y_pred_emb'] = y_pred_emb_train
        emb_train_df['y_pred_cnn'] = y_pred_cnn_train
        emb_train_df['idx'] = idx_train
        emb_train_df['split'] = 'train'
        for c in range(num_classes):
            emb_train_df[f"logits_cnn_{c}"] = logits_cnn_train[:, c]
            emb_train_df[f"logits_emb_{c}"] = logits_emb_train[:, c]

        emb_test_df['y_true'] = y_true_test
        emb_test_df['y_pred_emb'] = y_pred_emb_test
        emb_test_df['y_pred_cnn'] = y_pred_cnn_test
        emb_test_df['idx'] = idx_test
        emb_test_df['split'] = 'test'
        for c in range(num_classes):
            emb_test_df[f"logits_cnn_{c}"] = logits_cnn_test[:, c]
            emb_test_df[f"logits_emb_{c}"] = logits_emb_test[:, c]

        meta_train = train_df.reset_index(drop=True).loc[idx_train, ['subject', 'activity', 'file_path', 'window_idx']]
        meta_test  = test_df.reset_index(drop=True).loc[idx_test,  ['subject', 'activity', 'file_path', 'window_idx']]

        combined_train = pd.concat([meta_train.reset_index(drop=True),
                                     emb_train_df.reset_index(drop=True)], axis=1)
        combined_test = pd.concat([meta_test.reset_index(drop=True),
                                    emb_test_df.reset_index(drop=True)], axis=1)

        combined = pd.concat([combined_train, combined_test], ignore_index=True)
        combined.to_csv(os.path.join(output_folder, f"fold_{test_subj}.csv"), index=False)

        fold_metrics[test_subj] = {
            'best_val_macro_f1': float(metrics['best_macro_f1']),
            'best_epoch': int(metrics['best_epoch']),
            'emb_test_accuracy': float(metrics['emb_test_accuracy']),
            'emb_test_macro_f1': float(metrics['emb_test_macro_f1']),
            'cnn_test_accuracy': float(metrics['cnn_test_accuracy']),
            'cnn_test_macro_f1': float(metrics['cnn_test_macro_f1']),
        }

    global_acc = accuracy_score(all_yt, all_yp)
    global_f1  = f1_score(all_yt, all_yp, average="macro", zero_division=0)
    print(f"\n====== [{label}] GLOBAL AGGREGATED TEST (all folds pooled) ======")
    print(f"Accuracy:  {global_acc:.3f}")
    print(f"Macro-F1:  {global_f1:.3f}")

    return fold_metrics, global_acc, global_f1


# ── Utilities ─────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == "__main__":
    set_seed(42)
    DATA_PATH = "../../data_pipeline/results_30hz_fused_acc_ori_gyr_pose/windowed_30hz_fused_acc_ori_gyr_pose_lw_rw_lp_rp_h.csv"
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    poseT_cols = [f"poseT_{i}{j}_{joint}" for joint in range(NUM_JOINTS) for i in range(3) for j in range(3)]
    poseP_cols = [f"poseP_{i}{j}_{joint}" for joint in range(NUM_JOINTS) for i in range(3) for j in range(3)]
    assert all(c in df.columns for c in poseT_cols), "poseT_* columns missing"
    assert all(c in df.columns for c in poseP_cols), "poseP_* columns missing"

    print("\n========================= GROUND TRUTH (poseT_*) =========================")
    print("Converting rotation matrices to axis-angle (ground truth, one-time precompute) ...")
    feats_gt = rotmat_rows_to_axis_angle(df, poseT_cols)
    print(f"  features_gt shape: {feats_gt.shape}")
    fold_metrics_gt, global_acc_gt, global_f1_gt = run_loso_and_save_cnn_embeddings(
        df, feats_gt, "B1_SMPL_CNN_gt_embeddings", device, label="GT"
    )

    print("\n===================== MOBILEPOSER PREDICTED (poseP_*) =====================")
    print("Converting rotation matrices to axis-angle (predicted, one-time precompute) ...")
    feats_pred = rotmat_rows_to_axis_angle(df, poseP_cols)
    print(f"  features_pred shape: {feats_pred.shape}")
    fold_metrics_pred, global_acc_pred, global_f1_pred = run_loso_and_save_cnn_embeddings(
        df, feats_pred, "B1_SMPL_CNN_pred_embeddings", device, label="PRED"
    )

    print("\n================ FINAL SUMMARY (GT vs PREDICTED pose) ================")
    for label, fm, g_acc, g_f1 in [
        ("GT  ", fold_metrics_gt,   global_acc_gt,   global_f1_gt),
        ("PRED", fold_metrics_pred, global_acc_pred, global_f1_pred),
    ]:
        emb_acc = np.array([v['emb_test_accuracy'] for v in fm.values()])
        emb_f1  = np.array([v['emb_test_macro_f1'] for v in fm.values()])
        cnn_acc = np.array([v['cnn_test_accuracy'] for v in fm.values()])
        cnn_f1  = np.array([v['cnn_test_macro_f1'] for v in fm.values()])
        print(f"{label}: CNN per-fold acc={cnn_acc.mean():.3f}+/-{cnn_acc.std():.3f}  "
              f"macro-f1={cnn_f1.mean():.3f}+/-{cnn_f1.std():.3f}  |  "
              f"pooled acc={g_acc:.3f}  macro-f1={g_f1:.3f}")
        print(f"      emb  per-fold acc={emb_acc.mean():.3f}+/-{emb_acc.std():.3f}  "
              f"macro-f1={emb_f1.mean():.3f}+/-{emb_f1.std():.3f}")
    print("=======================================================================\n")
