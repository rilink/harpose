"""
A1: LightGBM on handcrafted IMU features.
Input: acc (15) + ori (45) + gyr (15) = 75 channels, 50-frame windows at 30 Hz.
Features: statistical + frequency-domain descriptors per channel.
"""

import os
import ast
import numpy as np
import pandas as pd

from scipy.stats import iqr, entropy, kurtosis, skew
from scipy.signal import welch, find_peaks
from scipy.fftpack import fft
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from lightgbm import LGBMClassifier


# ============================================================
# Feature calculation
# ============================================================

def safe_literal_eval(val):
    try:
        return ast.literal_eval(val)
    except Exception:
        return []


def signal_magnitude_vector(window):
    return np.sqrt(np.sum(np.square(window), axis=1))


def spectral_entropy(signal, sf=50):
    freqs, psd = welch(signal, sf)
    psd_norm = psd / np.sum(psd)
    return -np.sum(psd_norm * np.log2(psd_norm + 1e-12))


def hjorth_parameters(signal):
    d1 = np.diff(signal)
    d2 = np.diff(d1)

    var0 = np.var(signal)
    var1 = np.var(d1)
    var2 = np.var(d2)

    mobility = np.sqrt(var1 / var0) if var0 > 0 else 0
    complexity = (np.sqrt(var2 / var1) / mobility) if var1 > 0 and mobility > 0 else 0
    return mobility, complexity


def mean_crossing_rate(signal):
    mean_val = np.mean(signal)
    crossings = np.where(np.diff(np.signbit(signal - mean_val)))[0]
    return len(crossings) / len(signal)


def differential_entropy(signal):
    var = np.var(signal)
    return 0.5 * np.log(2 * np.pi * np.e * var)


def petrosian_fd(signal):
    diff = np.diff(signal)
    sign_changes = np.sum(diff[1:] * diff[:-1] < 0)
    n = len(signal)
    return np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * sign_changes)))


def katz_fd(signal):
    L = np.sum(np.abs(np.diff(signal)))
    d = np.max(np.abs(signal - signal[0]))
    n = len(signal)
    return np.log10(n) / (np.log10(d / (L + 1e-6) + 1e-6))


def orientation(x, y, z, dt=0.02):
    acc = np.stack([x, y, z], axis=1)
    vel = np.cumsum(acc * dt, axis=0)
    pos = np.cumsum(vel * dt, axis=0)

    pos_x, pos_y, pos_z = pos[-1]
    velo_acc = np.sum(vel[-1])
    pos_acc = np.sum(pos[-1])

    return pos_x, pos_y, pos_z, velo_acc, pos_acc


def calculate_1D_features(signal, features, axis_name):
    signal = np.asarray(signal)
    result = {}

    fft_vals = np.abs(fft(signal))[: len(signal)//2]

    # --- Time-domain ---
    if 'mean' in features:
        result[f'{axis_name}_mean'] = np.mean(signal)
    if 'var' in features:
        result[f'{axis_name}_var'] = np.var(signal)
    if 'std' in features:
        result[f'{axis_name}_std'] = np.std(signal)
    if 'rms' in features:
        result[f'{axis_name}_rms'] = np.sqrt(np.mean(signal**2))
    if 'shape_factor' in features:
        rms = np.sqrt(np.mean(signal**2))
        mv = np.mean(signal)
        result[f'{axis_name}_shape_factor'] = rms / (abs(mv) + 1e-6)

    if 'min' in features:
        result[f'{axis_name}_min'] = np.min(signal)
    if 'max' in features:
        result[f'{axis_name}_max'] = np.max(signal)
    if 'median' in features:
        result[f'{axis_name}_median'] = np.median(signal)
    if 'iqr' in features:
        result[f'{axis_name}_iqr'] = iqr(signal)
    if 'sma' in features:
        result[f'{axis_name}_sma'] = np.sum(np.abs(signal)) / len(signal)
    if 'skewness' in features:
        result[f'{axis_name}_skewness'] = skew(signal)
    if 'kurtosis' in features:
        result[f'{axis_name}_kurtosis'] = kurtosis(signal)

    if 'jerk_mean' in features or 'jerk_std' in features:
        j = np.diff(signal)
        if 'jerk_mean' in features:
            result[f'{axis_name}_jerk_mean'] = np.mean(np.abs(j))
        if 'jerk_std' in features:
            result[f'{axis_name}_jerk_std'] = np.std(j)

    if 'num_peak' in features:
        peaks, _ = find_peaks(signal, distance=3)
        result[f'{axis_name}_peaks'] = len(peaks)
    if 'zero_crossings' in features:
        result[f'{axis_name}_zero_crossings'] = np.sum(signal[:-1] * signal[1:] < 0)

    # --- Frequency-domain ---
    if 'energy' in features:
        result[f'{axis_name}_energy'] = np.sum(fft_vals**2)
    if 'dominant_freq' in features:
        result[f'{axis_name}_dominant_freq'] = np.argmax(fft_vals)
    if 'entropy' in features:
        prob = fft_vals / np.sum(fft_vals)
        result[f'{axis_name}_entropy'] = entropy(prob)
    if 'signal_entropy' in features:
        hist, _ = np.histogram(signal, bins=10, density=True)
        result[f'{axis_name}_signal_entropy'] = entropy(hist + 1e-10)

    return result


def calculate_3D_features(x, y, z, features):
    result = {}

    smv = signal_magnitude_vector(np.column_stack((x, y, z)))

    x_mean, y_mean, z_mean = np.mean(x), np.mean(y), np.mean(z)

    # Orientation / trajectory
    if any(f in features for f in ['pos_x', 'pos_y', 'pos_z', 'velo_acc', 'pos_acc']):
        pos_x, pos_y, pos_z, velo_acc, pos_acc = orientation(x, y, z)
        if 'pos_x' in features: result['pos_x'] = pos_x
        if 'pos_y' in features: result['pos_y'] = pos_y
        if 'pos_z' in features: result['pos_z'] = pos_z
        if 'velo_acc' in features: result['velo_acc'] = velo_acc
        if 'pos_acc' in features: result['pos_acc'] = pos_acc

    # Posture
    if 'pitch' in features:
        result['pitch'] = np.degrees(np.arctan2(-x_mean, np.sqrt(y_mean**2 + z_mean**2)))
    if 'roll' in features:
        result['roll'] = np.degrees(np.arctan2(y_mean, z_mean))
    if 'total_sqrt' in features:
        result['total_sqrt'] = np.sqrt(x_mean**2 + y_mean**2 + z_mean**2)
    if 'total' in features:
        result['total'] = x_mean + y_mean + z_mean
    if 'signal_magnitude' in features:
        result['signal_magnitude'] = np.sum(x + y + z) / len(x)
    if 'signal_magnitude_abs' in features:
        result['signal_magnitude_abs'] = np.sum(np.abs(x) + np.abs(y) + np.abs(z)) / len(x)

    # Correlations
    if 'axis_correlation' in features:
        result.update({
            'axis_corr_x_y': np.corrcoef(x, y)[0, 1],
            'axis_corr_x_z': np.corrcoef(x, z)[0, 1],
            'axis_corr_y_z': np.corrcoef(y, z)[0, 1],
        })

    if 'axis_covariance' in features:
        result.update({
            'axis_cov_x_y': np.cov(x, y)[0, 1],
            'axis_cov_x_z': np.cov(x, z)[0, 1],
            'axis_cov_y_z': np.cov(y, z)[0, 1],
        })

    # Other 3D features
    if 'tilt_angle' in features:
        norm = np.sqrt(x_mean**2 + y_mean**2 + z_mean**2)
        result['tilt_angle'] = np.degrees(np.arccos(z_mean / (norm + 1e-6)))
    if 'spectral_entropy' in features:
        result['spectral_entropy'] = spectral_entropy(smv)
    if 'min_smv' in features:
        result['min_smv'] = np.min(smv)
    if 'max_smv' in features:
        result['max_smv'] = np.max(smv)
    if 'ptp_smv' in features:
        result['ptp_smv'] = np.ptp(smv)
    if 'iqr_smv' in features:
        result['iqr_smv'] = iqr(smv)
    if 'std_smv' in features:
        result['std_smv'] = np.std(smv)
    if 'skew_smv' in features:
        result['skew_smv'] = skew(smv)
    if 'kurtosis_smv' in features:
        result['kurtosis_smv'] = kurtosis(smv)

    if 'mobility' in features or 'complexity' in features:
        mob, comp = hjorth_parameters(smv)
        if 'mobility' in features: result['mobility'] = mob
        if 'complexity' in features: result['complexity'] = comp

    if 'mean_crossing' in features:
        result['mean_crossing'] = mean_crossing_rate(smv)
    if 'differential_entropy' in features:
        result['differential_entropy'] = differential_entropy(smv)
    if 'petrosian_fd' in features:
        result['petrosian_fd'] = petrosian_fd(smv)
    if 'katz_fd' in features:
        result['katz_fd'] = katz_fd(smv)

    return result


def feature_calculation(df, output_path, features_1D=None, features_3D=None):
    if features_1D is None:
        features_1D = [
            'mean', 'var', 'rms', 'shape_factor',
            'min', 'max', 'median', 'iqr', 'sma',
            'energy', 'dominant_freq', 'entropy', 'signal_entropy',
            'jerk_mean', 'jerk_std', 'num_peak', 'zero_crossings'
        ]

    if features_3D is None:
        features_3D = [
            'pos_x', 'pos_y', 'pos_z', 'velo_acc', 'pos_acc',
            'pitch', 'roll',
            'total', 'total_sqrt', 'signal_magnitude', 'signal_magnitude_abs',
            'axis_correlation', 'axis_covariance', 'tilt_angle',
            'spectral_entropy', 'min_smv', 'max_smv', 'ptp_smv', 'iqr_smv', 'std_smv',
            'skew_smv', 'kurtosis_smv', 'mobility', 'complexity', 'mean_crossing',
            'differential_entropy', 'petrosian_fd', 'katz_fd'
        ]

    imu_columns = df.columns[6:]
    df[imu_columns] = df[imu_columns].map(safe_literal_eval)

    feature_rows = []

    for _, row in df.iterrows():

        if any(len(row[col]) == 0 for col in imu_columns):
            continue

        sample = row.drop(imu_columns).to_dict()
        features = {}

        # 1D features
        for axis in imu_columns:
            features.update(
                calculate_1D_features(row[axis], features_1D, axis)
            )

        # 3D features (accelerometer and gyroscope)
        for sensor_type in ['acc', 'gyr']:
            xs = [c for c in imu_columns if sensor_type in c and 'x' in c]
            ys = [c for c in imu_columns if sensor_type in c and 'y' in c]
            zs = [c for c in imu_columns if sensor_type in c and 'z' in c]

            if xs and ys and zs:
                x = np.asarray(row[xs[0]])
                y = np.asarray(row[ys[0]])
                z = np.asarray(row[zs[0]])

                dim3 = calculate_3D_features(x, y, z, features_3D)
                features.update({f"{k}_{sensor_type}": v for k, v in dim3.items()})

        sample.update(features)
        feature_rows.append(sample)

    result_df = pd.DataFrame(feature_rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df.to_csv(output_path, index=False)
    return result_df


# ============================================================
# LightGBM training / evaluation
# ============================================================

def LOSO(df_path):
    """
    Generator for Leave-One-Subject-Out cross-validation.
    Yields (test_subject_id, {'train': train_df, 'test': test_df}).
    Assumes the dataset has a 'subject' column with integer labels.
    """
    df = pd.read_csv(df_path)
    subjects = sorted(df["subject"].unique())

    for test_subj in subjects:
        train_df = df[df["subject"] != test_subj]
        test_df = df[df["subject"] == test_subj]
        yield test_subj, {"train": {"df": train_df}, "test": {"df": test_df}}


def train_and_evaluate_lgbm(train_df, test_df, drop_cols=None):
    """
    Trains a LightGBM classifier on train_df and evaluates on test_df.
    Returns model, x_train, y_train, x_test, y_test, predictions, accuracy, macro_f1.
    """
    if drop_cols is None:
        drop_cols = ["activity_num", "activity", "activity_encoded",
                     "subject", "file_path", "window_idx"]

    x_train = train_df.drop(columns=drop_cols)
    y_train = train_df["activity_encoded"]
    x_test = test_df.drop(columns=drop_cols)
    y_test = test_df["activity_encoded"]

    model = LGBMClassifier(
        subsample=0.9,
        boosting_type='gbdt',
        n_estimators=400,
        max_depth=15,
        learning_rate=0.1,
        colsample_bytree=1,
        n_jobs=-1,
        min_split_gain=0.05,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
    )

    model.fit(x_train, y_train)

    predicted = model.predict(x_test)
    accuracy = accuracy_score(y_test, predicted)
    macro_f1 = f1_score(y_test, predicted, average='macro')

    print(f"Accuracy on test data: {accuracy:.3f}")
    print(f"Macro F1 score on test data: {macro_f1:.3f}")

    return model, x_train, y_train, x_test, y_test, predicted, accuracy, macro_f1


def pooled_metrics(y_true, y_pred):
    """
    Pool predictions across all LOSO folds and compute accuracy, macro-F1,
    weighted-F1 and a confusion matrix over the combined set.

    Avoids the instability of per-fold macro-F1 when a test subject is
    missing one or more classes (small per-fold class counts mean a single
    misclassification can swing macro-F1 a lot).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro', labels=labels, zero_division=0),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted', labels=labels, zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=labels),
        'labels': labels,
    }


def export_lightgbm_csv(X_train, X_test, y_train, y_test, train_df, test_df,
                         gbm, num_classes, output_folder, test_subj):
    """
    Saves a per-fold CSV with the same structure as the DCL/CNN exported files
    (metadata + y_true/y_pred + per-class logits), for use in fusion experiments.
    """
    logits_train = np.array(gbm.predict(X_train, raw_score=True))
    logits_test = np.array(gbm.predict(X_test, raw_score=True))

    y_pred_train = logits_train.argmax(axis=1)
    y_pred_test = logits_test.argmax(axis=1)

    emb_train_df = pd.DataFrame()
    emb_test_df = pd.DataFrame()

    emb_train_df["y_true"] = y_train - 1
    emb_train_df["y_pred_lgbm"] = y_pred_train
    emb_train_df["idx"] = np.arange(len(y_train))
    emb_train_df["split"] = "train"

    emb_test_df["y_true"] = y_test - 1
    emb_test_df["y_pred_lgbm"] = y_pred_test
    emb_test_df["idx"] = np.arange(len(y_test))
    emb_test_df["split"] = "test"

    for c in range(num_classes):
        emb_train_df[f"logits_lgbm_{c}"] = logits_train[:, c]
        emb_test_df[f"logits_lgbm_{c}"] = logits_test[:, c]

    meta_train = train_df.reset_index()[["subject", "activity", "file_path", "window_idx"]]
    meta_test = test_df.reset_index()[["subject", "activity", "file_path", "window_idx"]]

    combined_train = pd.concat([meta_train.reset_index(drop=True),
                                 emb_train_df.reset_index(drop=True)], axis=1)
    combined_test = pd.concat([meta_test.reset_index(drop=True),
                                emb_test_df.reset_index(drop=True)], axis=1)
    combined = pd.concat([combined_train, combined_test], ignore_index=True)

    os.makedirs(output_folder, exist_ok=True)
    out_path = os.path.join(output_folder, f"fold_{test_subj}.csv")
    combined.to_csv(out_path, index=False)

    print(f"[LightGBM] Exported: {out_path}   ({len(combined)} rows)")
    return combined


# ============================================================
# Entry point
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_CSV = os.path.join(SCRIPT_DIR,
    "../../data_pipeline/results_30hz_fused_acc_ori_gyr_pose/windowed_30hz_fused_acc_ori_gyr_pose_lw_rw_lp_rp_h.csv")
FEATURES_CSV = os.path.join(SCRIPT_DIR, "data_features/imu_features.csv")
OUTPUT_FOLDER = os.path.join(SCRIPT_DIR, "A1_LightGBM_embeddings")
NUM_CLASSES = 4

if __name__ == "__main__":
    print(f"Loading {SRC_CSV} ...")
    raw_df = pd.read_csv(SRC_CSV)
    meta_cols = ['activity_num', 'activity', 'activity_encoded', 'subject', 'file_path', 'window_idx']
    imu_cols = [c for c in raw_df.columns if c.startswith(('acc_', 'ori_', 'gyr_'))]
    raw_df = raw_df[meta_cols + imu_cols]
    print(f"  {len(raw_df)} rows, {len(imu_cols)} IMU channels")

    if not os.path.exists(FEATURES_CSV):
        print("Computing features ...")
        feature_calculation(raw_df, output_path=FEATURES_CSV)
    else:
        print(f"Features already exist: {FEATURES_CSV}")

    macro_f1s, accs = [], []
    all_y_true, all_y_pred = [], []

    for test_subj, data_split in LOSO(FEATURES_CSV):
        train_df = data_split['train']['df']
        test_df = data_split['test']['df']

        model, X_train, y_train, X_test, y_test, y_pred, accuracy, macro_f1 = \
            train_and_evaluate_lgbm(train_df, test_df, drop_cols=None)

        macro_f1s.append(macro_f1)
        accs.append(accuracy)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        print(f"  subj {test_subj}: acc={accuracy:.3f}  macro-F1={macro_f1:.3f}")

        export_lightgbm_csv(
            X_train, X_test, y_train, y_test,
            train_df, test_df,
            model, NUM_CLASSES,
            OUTPUT_FOLDER, test_subj
        )

    print(f"\n====== LOSO Summary ======")
    print(f"macro-F1: {np.mean(macro_f1s):.3f} +/- {np.std(macro_f1s):.3f}")
    print(f"accuracy: {np.mean(accs):.3f} +/- {np.std(accs):.3f}")

    m = pooled_metrics(all_y_true, all_y_pred)
    print(f"\n====== POOLED ======")
    print(f"accuracy={m['accuracy']:.3f}  macro_f1={m['macro_f1']:.3f}")
    print(f"\nFold CSVs saved to: {OUTPUT_FOLDER}")
