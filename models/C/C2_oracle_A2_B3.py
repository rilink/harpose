"""
C2 oracle — A2 (DeepConvLSTM, g-acc+ori+gyr) + B3 (TCN-FK).
Picks whichever model is correct; if both right/wrong, prefer pose (B3).
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DCL_DIR = os.path.normpath(os.path.join(
    SCRIPT_DIR, "../A/A2_DeepConvLSTM_embeddings"))

POSE_DIRS = {
    "GT":   os.path.normpath(os.path.join(
        SCRIPT_DIR, "../B/B3_TCN_FK_gt_embeddings")),
    "PRED": os.path.normpath(os.path.join(
        SCRIPT_DIR, "../B/B3_TCN_FK_pred_embeddings")),
}

SUBJECTS = [1, 2, 3, 4, 5]
MERGE_KEYS = ["subject", "file_base", "window_idx"]


def _add_file_base(df):
    df = df.copy()
    df["file_base"] = df["file_path"].apply(lambda x: os.path.basename(str(x)))
    return df


def oracle_predict(y_true, y_imu, y_pose):
    """Prefer the IMU prediction if correct, otherwise fall back to pose
    (covers both "pose is right" and "both are wrong" cases)."""
    y_true, y_imu, y_pose = map(np.asarray, (y_true, y_imu, y_pose))
    return np.where(y_imu == y_true, y_imu, y_pose)


if __name__ == "__main__":
    print("=" * 58)
    print("  C2 oracle — A2 (DCL g-acc+ori+gyr) + B3 (TCN-FK)")
    print("=" * 58)

    for label, pose_dir in POSE_DIRS.items():
        f1s, accs = [], []
        for subj in SUBJECTS:
            dcl  = _add_file_base(pd.read_csv(os.path.join(DCL_DIR,   f"fold_{subj}.csv")))
            pose = _add_file_base(pd.read_csv(os.path.join(pose_dir,  f"fold_{subj}.csv")))

            dcl_sub  = dcl[MERGE_KEYS + ["split", "y_true", "y_pred_dcl"]]
            pose_sub = pose[MERGE_KEYS + ["y_pred_tcn"]].rename(columns={"y_pred_tcn": "y_pred_pose"})
            merged   = pd.merge(dcl_sub, pose_sub, on=MERGE_KEYS, how="inner")
            test     = merged[merged["split"] == "test"]

            y        = test["y_true"].values.astype(int)
            y_imu    = test["y_pred_dcl"].values.astype(int)
            y_pose   = test["y_pred_pose"].values.astype(int)
            y_oracle = oracle_predict(y, y_imu, y_pose)

            f1  = f1_score(y, y_oracle, average="macro", zero_division=0)
            acc = accuracy_score(y, y_oracle)
            f1s.append(f1); accs.append(acc)
            print(f"  [{label}] subj {subj}  macro-f1={f1:.3f}  acc={acc:.3f}")

        print(f"\n  [{label}] macro-f1 = {np.mean(f1s):.3f} +/- {np.std(f1s):.3f}")
        print(f"  [{label}] accuracy = {np.mean(accs):.3f} +/- {np.std(accs):.3f}\n")

    print("Done.")
