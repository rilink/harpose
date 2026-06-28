"""
Pooled (all-LOSO-folds) confusion matrices for every model, with both
row-normalized (recall) and column-normalized (precision) views side by
side. Reads the per-fold embedding/logit CSVs each model script already
writes (run the model scripts first); for C1/C2 it reuses the fusion logic
from models/C directly rather than recomputing it.

Output: figures/confusion_matrices/<name>.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../figures/confusion_matrices"))
CLASS_NAMES = ["acting", "freestyle", "rom", "walking"]
SUBJECTS = [1, 2, 3, 4, 5]


def plot_confusion_pair(y_true, y_pred, title, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    row_norm = cm / cm.sum(axis=1, keepdims=True)  # recall
    col_norm = cm / cm.sum(axis=0, keepdims=True)  # precision

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, mat, subtitle in [
        (axes[0], row_norm, "row-normalized (recall)"),
        (axes[1], col_norm, "column-normalized (precision)"),
    ]:
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right")
        ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(subtitle)
        for i in range(len(CLASS_NAMES)):
            for j in range(len(CLASS_NAMES)):
                val = mat[i, j]
                color = "white" if val > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}\n(n={cm[i, j]})", ha="center", va="center",
                        color=color, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def pooled_from_folds(embeddings_dir, pred_col):
    parts = []
    for subj in SUBJECTS:
        df = pd.read_csv(os.path.join(embeddings_dir, f"fold_{subj}.csv"))
        parts.append(df[df["split"] == "test"][["y_true", pred_col]])
    pooled = pd.concat(parts, ignore_index=True)
    return pooled["y_true"].values.astype(int), pooled[pred_col].values.astype(int)


def run_a_models():
    a_dir = os.path.join(SCRIPT_DIR, "A")
    specs = [
        ("A1_LightGBM", "A1_LightGBM_embeddings", "y_pred_lgbm"),
        ("A2_DeepConvLSTM", "A2_DeepConvLSTM_embeddings", "y_pred_dcl"),
        ("A3_CNNHAR", "A3_CNNHAR_embeddings", "y_pred_cnnhar"),
        ("A4_AttendDiscriminate", "A4_AttendDiscriminate_embeddings", "y_pred_attend"),
        ("A5_TinyHAR", "A5_TinyHAR_embeddings", "y_pred_tinyhar"),
    ]
    for name, subdir, pred_col in specs:
        yt, yp = pooled_from_folds(os.path.join(a_dir, subdir), pred_col)
        plot_confusion_pair(yt, yp, name, os.path.join(OUT_DIR, f"{name}.png"))


def run_b_models():
    b_dir = os.path.join(SCRIPT_DIR, "B")
    specs = [
        ("B1_SMPL_CNN", "B1_SMPL_CNN_{cond}_embeddings", "y_pred_cnn"),
        ("B2_vel_CNN", "B2_vel_CNN_{cond}_embeddings", "y_pred_cnn"),
        ("B3_TCN_FK", "B3_TCN_FK_{cond}_embeddings", "y_pred_tcn"),
        ("B4_STGCN", "B4_STGCN_{cond}_embeddings", "y_pred_stgcn"),
    ]
    for name, subdir_tpl, pred_col in specs:
        for cond in ["gt", "pred"]:
            subdir = os.path.join(b_dir, subdir_tpl.format(cond=cond))
            yt, yp = pooled_from_folds(subdir, pred_col)
            plot_confusion_pair(yt, yp, f"{name} ({cond.upper()})",
                                 os.path.join(OUT_DIR, f"{name}_{cond}.png"))


def run_c_models():
    c_dir = os.path.join(SCRIPT_DIR, "C")
    sys.path.insert(0, c_dir)

    import C1_A1_B3_fusion as c1a1
    import C1_A2_B3_fusion as c1a2
    import C2_oracle_A1_B3 as c2a1
    import C2_oracle_A2_B3 as c2a2

    # C1 (threshold fusion): rerun the per-fold tau selection, pool y_true/y_pred.
    for name, mod in [("C1_A1_B3", c1a1), ("C1_A2_B3", c1a2)]:
        for label, pose_dir in mod.POSE_DIRS.items():
            folds_data = {s: mod.load_fold(s, pose_dir) for s in SUBJECTS}
            all_yt, all_yp = [], []
            for test_subj in SUBJECTS:
                tau = mod.best_tau_on_pool(folds_data, exclude_subj=test_subj)
                test_df = folds_data[test_subj]
                test_df = test_df[test_df["split"] == "test"]
                p_imu, p_pose, y = mod.get_probs(test_df)
                yp = mod.threshold_fusion(p_imu, p_pose, tau)
                all_yt.extend(y.tolist())
                all_yp.extend(yp.tolist())
            cond = "gt" if label == "GT" else "pred"
            plot_confusion_pair(all_yt, all_yp, f"{name} ({label})",
                                 os.path.join(OUT_DIR, f"{name}_{cond}.png"))

    # C2 (oracle): pool y_true/y_pred across folds.
    for name, mod, dir_attr, y_pred_col in [
        ("C2_oracle_A1_B3", c2a1, "A1_DIR", "y_pred_lgbm"),
        ("C2_oracle_A2_B3", c2a2, "DCL_DIR", "y_pred_dcl"),
    ]:
        imu_dir = getattr(mod, dir_attr)
        for label, pose_dir in mod.POSE_DIRS.items():
            all_yt, all_yp = [], []
            for subj in SUBJECTS:
                imu = mod._add_file_base(pd.read_csv(os.path.join(imu_dir, f"fold_{subj}.csv")))
                pose = mod._add_file_base(pd.read_csv(os.path.join(pose_dir, f"fold_{subj}.csv")))
                imu_sub = imu[mod.MERGE_KEYS + ["split", "y_true", y_pred_col]]
                pose_sub = pose[mod.MERGE_KEYS + ["y_pred_tcn"]].rename(columns={"y_pred_tcn": "y_pred_pose"})
                merged = pd.merge(imu_sub, pose_sub, on=mod.MERGE_KEYS, how="inner")
                test = merged[merged["split"] == "test"]
                y = test["y_true"].values.astype(int)
                y_imu = test[y_pred_col].values.astype(int)
                y_pose = test["y_pred_pose"].values.astype(int)
                y_oracle = mod.oracle_predict(y, y_imu, y_pose)
                all_yt.extend(y.tolist())
                all_yp.extend(y_oracle.tolist())
            cond = "gt" if label == "GT" else "pred"
            plot_confusion_pair(all_yt, all_yp, f"{name} ({label})",
                                 os.path.join(OUT_DIR, f"{name}_{cond}.png"))


if __name__ == "__main__":
    print("Category A models ...")
    run_a_models()
    print("Category B models ...")
    run_b_models()
    print("Category C models ...")
    run_c_models()
    print("\nDone.")
