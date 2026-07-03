"""
C1 (A2+B3): Confidence-threshold fusion of DeepConvLSTM (A2) and TCN-FK (B3).
Threshold τ selected per fold on pooled test data from all other folds.
"""

import os
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import f1_score, accuracy_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DCL_DIR = os.path.normpath(os.path.join(
    SCRIPT_DIR,
    "../A/A2_DeepConvLSTM_embeddings"))

POSE_DIRS = {
    "GT":   os.path.normpath(os.path.join(
        SCRIPT_DIR,
        "../B/B3_TCN_FK_gt_embeddings")),
    "PRED": os.path.normpath(os.path.join(
        SCRIPT_DIR,
        "../B/B3_TCN_FK_pred_embeddings")),
}

NUM_CLASSES = 4
TAU_VALUES  = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
SUBJECTS    = [1, 2, 3, 4, 5]

DCL_LOGIT_COLS  = [f"logits_dcl_{i}" for i in range(NUM_CLASSES)]
POSE_LOGIT_COLS = [f"logits_tcn_{i}" for i in range(NUM_CLASSES)]
MERGE_KEYS      = ["subject", "file_base", "window_idx"]


def _add_file_base(df):
    df = df.copy()
    df["file_base"] = df["file_path"].apply(lambda x: os.path.basename(str(x)))
    return df


def load_fold(subj, pose_dir):
    dcl  = _add_file_base(pd.read_csv(os.path.join(DCL_DIR,  f"fold_{subj}.csv")))
    pose = _add_file_base(pd.read_csv(os.path.join(pose_dir, f"fold_{subj}.csv")))

    dcl_sub  = dcl[MERGE_KEYS + ["split", "y_true"] + DCL_LOGIT_COLS]
    pose_sub = pose[MERGE_KEYS + POSE_LOGIT_COLS].rename(
        columns={f"logits_tcn_{i}": f"logits_pose_{i}" for i in range(NUM_CLASSES)})

    return pd.merge(dcl_sub, pose_sub, on=MERGE_KEYS, how="inner")


def threshold_fusion(p_imu, p_pose, tau):
    use_imu = (p_pose.max(axis=1) < tau) & (p_imu.max(axis=1) > p_pose.max(axis=1))
    return np.where(use_imu[:, None], p_imu, p_pose).argmax(axis=1)


def get_probs(df):
    p_imu  = softmax(df[DCL_LOGIT_COLS].values.astype(np.float32), axis=1)
    p_pose = softmax(df[[f"logits_pose_{i}" for i in range(NUM_CLASSES)]].values.astype(np.float32), axis=1)
    y      = df["y_true"].values.astype(int)
    return p_imu, p_pose, y


def best_tau_on_pool(folds_data, exclude_subj):
    rows = [d for s, d in folds_data.items() if s != exclude_subj]
    df   = pd.concat(rows, ignore_index=True)
    test = df[df["split"] == "test"]
    p_imu, p_pose, y = get_probs(test)
    best_f1, best_tau = -1, TAU_VALUES[0]
    for tau in TAU_VALUES:
        f1 = f1_score(y, threshold_fusion(p_imu, p_pose, tau),
                      average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_tau = f1, tau
    return best_tau


if __name__ == "__main__":
    print("=" * 62)
    print("  C1 (A2+B3) — Threshold fusion (DeepConvLSTM + TCN-FK)")
    print("  IMU: g-acc+ori+gyr (75ch)  |  tau: per-fold on other folds")
    print("=" * 62)

    for label, pose_dir in POSE_DIRS.items():
        folds_data = {s: load_fold(s, pose_dir) for s in SUBJECTS}

        f1s, accs, taus = [], [], []
        for test_subj in SUBJECTS:
            tau = best_tau_on_pool(folds_data, exclude_subj=test_subj)
            taus.append(tau)

            test_df = folds_data[test_subj]
            test_df = test_df[test_df["split"] == "test"]
            p_imu, p_pose, y = get_probs(test_df)

            yp  = threshold_fusion(p_imu, p_pose, tau)
            f1  = f1_score(y, yp, average="macro", zero_division=0)
            acc = accuracy_score(y, yp)
            f1s.append(f1)
            accs.append(acc)
            print(f"  [{label}] subj {test_subj}  tau={tau:.2f}  macro-f1={f1:.3f}  acc={acc:.3f}")

        print(f"\n  [{label}] per-fold taus: {taus}")
        print(f"  [{label}] macro-f1 = {np.mean(f1s):.3f} +/- {np.std(f1s):.3f}")
        print(f"  [{label}] accuracy = {np.mean(accs):.3f} +/- {np.std(accs):.3f}\n")

    print("Done.")
