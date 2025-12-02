"""
CNN based on SMPL parameters
Differentiates between four TotalCapture activities using SMPL pose parameters.
"""

import os
import sys
import glob
import random
import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from typing import Tuple, Dict, Any


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


class SMPLDataset(Dataset):
    """
    Loads windowed SMPL pose data from windowed_smpl_imu.csv.
    Each smpl_i column contains a list of 100 values.
    Reconstructs SMPL pose (100, 72) -> CNN input (3, 24, 100).
    Returns (features, label, idx).
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

        self.smpl_cols = [c for c in df.columns if c.startswith("smpl_")]
        assert len(self.smpl_cols) == SMPL_PARAMS, \
            f"Expected {SMPL_PARAMS} smpl columns, got {len(self.smpl_cols)}"

        X_list = []
        for _, row in self.df.iterrows():
            frame_matrix = []
            for c in self.smpl_cols:
                cell = row[c]
                if isinstance(cell, str):
                    try:
                        cell = ast.literal_eval(cell)
                    except Exception:
                        cell = np.fromstring(cell.strip("[]"), sep=",", dtype=np.float32)

                arr = np.array(cell, dtype=np.float32)
                assert arr.shape[0] == WINDOW, \
                    f"Column {c} has window {arr.shape[0]}, expected {WINDOW}"

                frame_matrix.append(arr)

            # stack → (72, 100)
            frame_matrix = np.stack(frame_matrix, axis=0)
            # reshape: (72 = 24*3) → (24, 3, 100)
            frame_matrix = frame_matrix.reshape(NUM_JOINTS, NUM_CHANNELS, WINDOW)
            # permute → (3, 24, 100) for CNN
            frame_matrix = frame_matrix.transpose(1, 0, 2)

            X_list.append(frame_matrix)

        self.X = torch.tensor(np.stack(X_list, axis=0), dtype=torch.float32)

        if "activity_encoded" not in df.columns:
            raise ValueError("CSV must contain 'activity_encoded' column for labels.")
        self.y = torch.tensor(df["activity_encoded"].values.astype(int) - 1,
                              dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], int(idx)

# ---------------------------
# LOSO generator
# ---------------------------
def LOSO(data_path: str):
    """
    Yield LOSO splits: (test_subj, {'train': train_df, 'test': test_df}).
    """
    df = pd.read_csv(data_path)
    subjects = sorted(df["subject"].unique())
    for test_subj in subjects:
        train_df = df[df["subject"] != test_subj].reset_index(drop=True)
        test_df = df[df["subject"] == test_subj].reset_index(drop=True)
        yield test_subj, {'train': train_df, 'test': test_df}

# ---------------------------
# Training CNN 
# ---------------------------
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
    """
    Train PoseCNN with internal train/val split, identical structure to DeepConvLSTM training.
    """

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

    # -------------------------------------------
    # Training loop
    # -------------------------------------------
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

        # ---------------------------
        # Validation
        # ---------------------------
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

        print(f"Epoch {epoch+1}/{epochs}  |  loss {avg_loss:.4f}  val_f1 {macro_f1:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    

    metrics = {
        "best_macro_f1": float(best_macro_f1),
        "best_epoch": best_epoch,
    }

    return model, metrics


# ---------------------------
# CNN Embedding and Logit Extraction
# ---------------------------
def extract_cnn_embeddings(model, data_loader, device):
    """
    Extract embeddings AND logits from the CNN model.

    Returns:
        embeddings      : normalized feature vectors from CNN penultimate layer
        y_true          : ground-truth labels
        y_pred_emb      : predictions based on embeddings
        y_pred_cnn      : direct CNN predictions
        indices         : original dataset indices
        logits_cnn_all  : final CNN logits
        logits_emb_all  : logits computed from embeddings
    """
    model.eval()

    embeddings = []
    y_true = []
    y_pred_emb = []
    y_pred_cnn = []
    indices = []

    logits_cnn_all = []
    logits_emb_all = []

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
        np.concatenate(logits_emb_all)   
    )


# ---------------------------
# LOSO + Save CNN Embeddings
# ---------------------------
def run_loso_and_save_cnn_embeddings(data_path: str, output_folder: str, device: torch.device):
    """
    Run LOSO training + CNN embedding extraction and save combined train+test
    embeddings per fold.
    """
    os.makedirs(output_folder, exist_ok=True)
    fold_metrics = {}

    for test_subj, split in LOSO(data_path):

        train_df = split['train'].reset_index(drop=True)
        test_df  = split['test'].reset_index(drop=True)

        train_dataset_full = SMPLDataset(train_df)
        test_dataset       = SMPLDataset(test_df)

        # ---------------------------
        # Train CNN model
        # ---------------------------
        model, metrics = train_cnn_model(
            train_dataset=train_dataset_full,
            device=device,
            epochs=10,
            batch_size=32,
            lr=1e-3,
            val_fraction=0.1,
            seed=42
        )

        # ---------------------------
        # Embedding extraction
        # ---------------------------
        train_loader_emb = DataLoader(train_dataset_full, batch_size=64, shuffle=False)
        test_loader_emb  = DataLoader(test_dataset, batch_size=64, shuffle=False)

        emb_train, y_true_train, y_pred_emb_train, y_pred_cnn_train, idx_train, logits_cnn_train, logits_emb_train = \
            extract_cnn_embeddings(model, train_loader_emb, device)

        emb_test, y_true_test, y_pred_emb_test, y_pred_cnn_test, idx_test, logits_cnn_test, logits_emb_test  = \
            extract_cnn_embeddings(model, test_loader_emb, device)

        # ---------------------------
        # Compute metrics
        # ---------------------------
        emb_test_acc      = accuracy_score(y_true_test, y_pred_emb_test)
        emb_test_macro_f1 = f1_score(y_true_test, y_pred_emb_test, average='macro')

        cnn_test_acc      = accuracy_score(y_true_test, y_pred_cnn_test)
        cnn_test_macro_f1 = f1_score(y_true_test, y_pred_cnn_test, average='macro')

        metrics['emb_test_accuracy']   = float(emb_test_acc)
        metrics['emb_test_macro_f1']   = float(emb_test_macro_f1)
        metrics['cnn_test_accuracy']   = float(cnn_test_acc)
        metrics['cnn_test_macro_f1']   = float(cnn_test_macro_f1)

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

        meta_train = train_df.reset_index().loc[idx_train, ['subject', 'activity', 'file_path', 'window_idx']]
        meta_test  = test_df.reset_index().loc[idx_test,  ['subject', 'activity', 'file_path', 'window_idx']]

        combined_train = pd.concat([meta_train.reset_index(drop=True),
                                    emb_train_df.reset_index(drop=True)], axis=1)

        combined_test = pd.concat([meta_test.reset_index(drop=True),
                                   emb_test_df.reset_index(drop=True)], axis=1)

        combined = pd.concat([combined_train, combined_test], ignore_index=True)

        # Save CSV
        out_path = os.path.join(output_folder, f"fold_{test_subj}.csv")
        combined.to_csv(out_path, index=False)

        fold_metrics[test_subj] = {
            'best_val_macro_f1': float(metrics['best_macro_f1']),
            'best_epoch': int(metrics['best_epoch']),
            'emb_test_accuracy': float(metrics['emb_test_accuracy']),
            'emb_test_macro_f1': float(metrics['emb_test_macro_f1']),
            'cnn_test_accuracy': float(metrics['cnn_test_accuracy']),
            'cnn_test_macro_f1': float(metrics['cnn_test_macro_f1']),
        }

    return fold_metrics


# -------------------------------------------
# Seed
# -------------------------------------------
def set_seed(seed: int = 42):
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    set_seed(42)
    DATA_PATH = "../../1_data_pipeline/processed_data/windowed_smpl_imu.csv"
    OUTPUT_FOLDER = "CNN_embeddings"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    WINDOW = 100
    NUM_JOINTS = 24
    NUM_CHANNELS = 3
    SMPL_PARAMS = NUM_JOINTS * NUM_CHANNELS   # = 72


    fold_metrics = run_loso_and_save_cnn_embeddings(DATA_PATH, OUTPUT_FOLDER, device)
    print("Fold metrics:", fold_metrics)

    # Convert dict → arrays
    emb_acc      = np.array([v['emb_test_accuracy']  for v in fold_metrics.values()])
    emb_f1       = np.array([v['emb_test_macro_f1']  for v in fold_metrics.values()])
    cnn_acc      = np.array([v['cnn_test_accuracy']  for v in fold_metrics.values()])
    cnn_f1       = np.array([v['cnn_test_macro_f1']  for v in fold_metrics.values()])

    print("\n================ LOSO SUMMARY ================")

    print(f"Embedding Test Accuracy:   {emb_acc.mean():.3f} ± {emb_acc.std():.3f}")
    print(f"Embedding Test Macro-F1:   {emb_f1.mean():.3f} ± {emb_f1.std():.3f}")

    print(f"CNN Test Accuracy:         {cnn_acc.mean():.3f} ± {cnn_acc.std():.3f}")
    print(f"CNN Test Macro-F1:         {cnn_f1.mean():.3f} ± {cnn_f1.std():.3f}")

    print("=============================================\n")
