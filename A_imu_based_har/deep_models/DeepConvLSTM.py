"""
DeepConvLSTM LOSO training + embedding extraction (LSTM features).
- Columns containing 'smpl' are ignored throughout.
- Saves one CSV per fold under DeepConvLSTM_embeddings/ containing logits.
"""

import os
import ast
import glob
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import f1_score, accuracy_score


# ---------------------------
# Model
# ---------------------------
class DeepConvLSTM(nn.Module):
    """
    DeepConvLSTM architecture.
    The network: 4 Conv2D layers -> reshape -> single LSTM -> classifier.
    get_lstm_features returns the last hidden output of LSTM (embedding).
    """
    def __init__(
        self,
        channels: int,
        classes: int,
        window_size: int,
        conv_kernels: int = 64,
        conv_kernel_size: int = 5,
        lstm_units: int = 128,
        lstm_layers: int = 1,
        dropout: float = 0.5,
        feature_extract: str = None
    ):
        super().__init__()
        self.conv_kernels = conv_kernels
        self.conv_kernel_size = conv_kernel_size
        self.lstm_units = lstm_units
        self.feature_extract = feature_extract

        self.conv1 = nn.Conv2d(1, conv_kernels, (conv_kernel_size, 1))
        self.conv2 = nn.Conv2d(conv_kernels, conv_kernels, (conv_kernel_size, 1))
        self.conv3 = nn.Conv2d(conv_kernels, conv_kernels, (conv_kernel_size, 1))
        self.conv4 = nn.Conv2d(conv_kernels, conv_kernels, (conv_kernel_size, 1))
   
        # compute final sequence length after 4 convs
        final_seq_len = window_size - (conv_kernel_size - 1) * 4
        self.final_seq_len = final_seq_len if final_seq_len > 0 else 1

        lstm_input_size = conv_kernels * channels
        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=lstm_units, num_layers=lstm_layers, batch_first=True)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_units, classes)
        self.activation = nn.ReLU()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that returns class logits for the last timestep.
        Input shape: (batch, seq_len, channels)
        """
        x = x.unsqueeze(1)  # (batch, 1, seq_len, channels)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))

        if self.feature_extract == 'conv':
            return x.view(x.shape[0], -1)


        x = x.permute(0, 2, 1, 3)  # (batch, seq_len', conv_kernels, channels)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # (batch, seq_len', conv_kernels*channels)

        lstm_out, _ = self.lstm(x)  # (batch, seq_len', hidden)
        last = lstm_out[:, -1, :]  # (batch, hidden)
        out = self.dropout(last)
        logits = self.classifier(out)
        return logits
    

    def get_lstm_features(self, x):
        """
        Return LSTM last hidden features (embeddings) for input batch.
        Input shape: (batch, seq_len, channels)
        Output shape: (batch, lstm_units)
        """
        x = x.unsqueeze(1)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        lstm_out, _ = self.lstm(x)
        return lstm_out[:, -1, :].detach() 

# ---------------------------
# Dataset
# ---------------------------
class InertialDataset(Dataset):
    """
    Dataset that converts dataframe rows (list-like strings or lists) into tensors.
    Columns containing 'smpl' are ignored.
    """
    def __init__(self, df: pd.DataFrame):
        feature_cols = [c for c in df.columns if any(sensor in c for sensor in ['acc', 'gyr', 'mag']) and ('smpl' not in c)]
        self.meta = df.reset_index(drop=True)
        self.feature_cols = feature_cols

        features = []
        for _, row in self.meta.iterrows():
            channel_arrays = []
            for c in feature_cols:
                cell = row[c]
                if isinstance(cell, str):
                    try:
                        cell = ast.literal_eval(cell)
                    except Exception:
                        cell = np.fromstring(cell.strip("[]"), sep=",")
                channel_arrays.append(np.array(cell, dtype=np.float32))
            sensor_data = np.stack(channel_arrays, axis=1)
            features.append(sensor_data)

        self.features = np.stack(features, axis=0)
        self.labels = (self.meta['activity_encoded'].values.astype(int) - 1).astype(int)
        self.channels = self.features.shape[2]
        self.window_size = self.features.shape[1]
        self.classes = int(np.unique(self.labels).shape[0])

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.features)

    def __getitem__(self, idx: int):
        """Return (features, label, idx)."""
        x = self.features[idx]
        y = self.labels[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long), int(idx)

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
# Utilities
# ---------------------------
def compute_mean_std_from_dataset(dataset: InertialDataset) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-channel mean and std across the dataset features (over samples and time).
    """
    data = dataset.features  # shape (N, T, C)
    mean = np.mean(data, axis=(0, 1)).astype(np.float32)
    std = np.std(data, axis=(0, 1)).astype(np.float32)
    std[std == 0.0] = 1.0
    return torch.tensor(mean).view(1, 1, -1), torch.tensor(std).view(1, 1, -1)

# ---------------------------
# Training (separate function)
# ---------------------------
def train_model(
    train_dataset: InertialDataset,
    device: torch.device,
    window_size: int,
    channels: int,
    classes: int,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_fraction: float = 0.1,
    seed: int = 42
) -> Tuple[DeepConvLSTM, Dict[str, Any]]:
    """
    Train DeepConvLSTM with internal train/val split. Returns trained model and metrics.
    Training uses mean/std from the full training dataset to normalize train/val/test.
    """
    set_seed(seed)
    model = DeepConvLSTM(channels=channels, classes=classes, window_size=window_size).to(device)

    # Prepare train/val split
    n_total = len(train_dataset)
    n_val = int(n_total * val_fraction)
    n_train = n_total - n_val
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(train_dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # Compute normalization
    mean, std = compute_mean_std_from_dataset(train_dataset)
    mean = mean.to(device)
    std = std.to(device)

    # Class weights
    all_labels = train_dataset.labels
    class_counts = np.bincount(all_labels, minlength=classes)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_macro_f1 = 0.0
    best_state = None
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X, y, _ in train_loader:
            X = X.to(device).float()
            X = (X - mean) / (std + 1e-8)  
            X = torch.tensor(X, dtype=torch.float32).to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_loss = train_loss / max(1, len(train_loader))

        # Validation
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for X, y, _ in val_loader:
                X = X.to(device).float()
                X = (X - mean) / (std + 1e-8) 
                X = torch.tensor(X, dtype=torch.float32).to(device)
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
        'best_macro_f1': best_macro_f1,
        'best_epoch': best_epoch,
        'mean': mean,
        'std': std
    }
    return model, metrics

# ---------------------------
# Embedding extraction (separate function)
# ---------------------------
def extract_embeddings(model, data_loader, device, mean, std, level='lstm',):
    model.eval()
    model.feature_extract = level

    embeddings, y_true, y_pred_emb, y_pred_dcl, indices = [], [], [], [], []

    with torch.no_grad():
        for X, y, idx in data_loader:
            X = X.to(device).float()
            X = (X - mean) / (std)
            print('X')
            print(X.shape)
            logits_dcl = model(X)
            preds_dcl = torch.argmax(logits_dcl, dim=1)
            z = model.get_lstm_features(X)
            embeddings.append(z.cpu().numpy())
            
            logits_emb = model.classifier(model.dropout(z.to(device)))
            preds_emb = torch.argmax(logits_emb, dim=1)

            y_true.append(y.numpy())
            y_pred_emb.append(preds_emb.cpu().numpy())
            y_pred_dcl.append(preds_dcl.cpu().numpy())
            indices.append(idx.numpy())

    print('embeddings')
    print(embeddings)
            
    model.feature_extract = None
    return (
        np.concatenate(embeddings),
        np.concatenate(y_true),
        np.concatenate(y_pred_emb),
        np.concatenate(y_pred_dcl),
        np.concatenate(indices)
    )

# ---------------------------
# Orchestration per fold
# ---------------------------
def run_loso_and_save_embeddings(data_path: str, output_folder: str, device: torch.device):
    """
    Run LOSO training + embedding extraction, and save combined embeddings per fold under output_folder.
    """
    os.makedirs(output_folder, exist_ok=True)
    fold_metrics = {}
    for test_subj, split in LOSO(data_path):
        print('LOSO Test subject:', test_subj)
        train_df = split['train'].reset_index(drop=True)
        test_df = split['test'].reset_index(drop=True)

        train_dataset_full = InertialDataset(train_df)
        test_dataset = InertialDataset(test_df)

        model, metrics = train_model(
            train_dataset=train_dataset_full,
            device=device,
            window_size=train_dataset_full.window_size,
            channels=train_dataset_full.channels,
            classes=train_dataset_full.classes,
            epochs=10,
            batch_size=32,
            lr=1e-3,
            val_fraction=0.1,
            seed=42
        )

        mean = metrics['mean']
        std = metrics['std']

        train_loader_for_embeddings = DataLoader(train_dataset_full, batch_size=64, shuffle=False)
        test_loader_for_embeddings = DataLoader(test_dataset, batch_size=64, shuffle=False)

        emb_train, y_true_train, y_pred_emb_train, y_pred_dcl_train, idx_train = extract_embeddings(model, train_loader_for_embeddings, device, mean=mean, std=std, level='lstm') 
        emb_test, y_true_test, y_pred_emb_test, y_pred_dcl_test, idx_test = extract_embeddings(model, test_loader_for_embeddings, device, mean=mean, std=std, level='lstm') 

        emb_test_accuracy = accuracy_score(y_true_test, y_pred_emb_test)
        emb_test_macro_f1 = f1_score(y_true_test, y_pred_emb_test, average='macro')

        dcl_test_accuracy = accuracy_score(y_true_test, y_pred_dcl_test)
        dcl_test_macro_f1 = f1_score(y_true_test, y_pred_dcl_test, average='macro')

        metrics['emb_test_accuracy'] = float(emb_test_accuracy)
        metrics['emb_test_macro_f1'] = float(emb_test_macro_f1)

        metrics['dcl_test_accuracy'] = float(dcl_test_accuracy)
        metrics['dcl_test_macro_f1'] = float(dcl_test_macro_f1)

        emb_train_df = pd.DataFrame(emb_train)
        emb_test_df = pd.DataFrame(emb_test)

        emb_train_df.columns = [f"emb_{i+1}" for i in range(emb_train_df.shape[1])]
        emb_test_df.columns = [f"emb_{i+1}" for i in range(emb_test_df.shape[1])]

        emb_train_df['y_true'] = y_true_train
        emb_train_df['y_pred_emb'] = y_pred_emb_train
        emb_train_df['y_pred_dcl'] = y_pred_dcl_train
        emb_train_df['idx'] = idx_train
        emb_train_df['split'] = 'train'

        emb_test_df['y_true'] = y_true_test
        emb_test_df['y_pred_emb'] = y_pred_emb_test
        emb_test_df['y_pred_dcl'] = y_pred_dcl_test
        emb_test_df['idx'] = idx_test
        emb_test_df['split'] = 'test'

        meta_train = train_df.reset_index().loc[idx_train, ['subject', 'activity', 'file_path', 'window_idx']].reset_index(drop=True)
        meta_test = test_df.reset_index().loc[idx_test, ['subject', 'activity', 'file_path', 'window_idx']].reset_index(drop=True)

        combined_train = pd.concat([meta_train, emb_train_df.reset_index(drop=True)], axis=1)
        combined_test = pd.concat([meta_test, emb_test_df.reset_index(drop=True)], axis=1)
        combined = pd.concat([combined_train, combined_test], ignore_index=True, sort=False)

        out_path = os.path.join(output_folder, f"fold_{test_subj}.csv")
        combined.to_csv(out_path, index=False)

        fold_metrics[test_subj] = {
            'best_val_macro_f1': float(metrics['best_macro_f1']),
            'best_epoch': int(metrics['best_epoch']),
            'emb_test_accuracy': metrics['emb_test_accuracy'],
            'emb_test_macro_f1': metrics['emb_test_macro_f1'],
            'dcl_test_accuracy': metrics['dcl_test_accuracy'],
            'dcl_test_macro_f1': metrics['dcl_test_macro_f1']
        }

    return fold_metrics

# ---------------------------
# Seed
# ---------------------------
def set_seed(seed: int = 42) -> None:
    """Set deterministic seeds for reproducible experiments."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    set_seed(42)
    DATA_PATH = "../../1_data_pipeline/processed_data/windowed_smpl_imu.csv"
    OUTPUT_FOLDER = "DeepConvLSTM_embeddings"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fold_metrics = run_loso_and_save_embeddings(DATA_PATH, OUTPUT_FOLDER, device)
    print("Fold metrics:", fold_metrics)

    # Convert dict → arrays
    emb_acc      = np.array([v['emb_test_accuracy']  for v in fold_metrics.values()])
    emb_f1       = np.array([v['emb_test_macro_f1']  for v in fold_metrics.values()])
    dcl_acc      = np.array([v['dcl_test_accuracy']  for v in fold_metrics.values()])
    dcl_f1       = np.array([v['dcl_test_macro_f1']  for v in fold_metrics.values()])

    print("\n================ LOSO SUMMARY ================")

    print(f"Embedding Test Accuracy:   {emb_acc.mean():.3f} ± {emb_acc.std():.3f}")
    print(f"Embedding Test Macro-F1:   {emb_f1.mean():.3f} ± {emb_f1.std():.3f}")

    print(f"DCL Test Accuracy:         {dcl_acc.mean():.3f} ± {dcl_acc.std():.3f}")
    print(f"DCL Test Macro-F1:         {dcl_f1.mean():.3f} ± {dcl_f1.std():.3f}")

    print("=============================================\n")

