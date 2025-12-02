"""
===========================================================
    Fusion Experiments: CNN + DCL and CNN + LightGBM
===========================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import random
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ---------------------------------------------------------
# Utility: merge DCL/CNN or LGBM/CNN embeddings
# ---------------------------------------------------------

def merge_embeddings(df_a: pd.DataFrame,
                     df_b: pd.DataFrame,
                     merge_on=('subject', 'activity', 'file_path', 'window_idx')):
    """
    Generic merge for any pair of models sharing the same window alignment.
    """
    merged = df_a.merge(df_b, how='inner', on=list(merge_on))

    drop_cols = [c for c in merged.columns if "split" in c.lower() or c.lower() == "idx"]
    merged = merged.drop(columns=drop_cols, errors="ignore")

    return merged


def compute_confusion_matrix(y_true, y_pred):
    labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=labels, columns=labels)


# ---------------------------------------------------------
# Fusion methods
# ---------------------------------------------------------

def weighted_logit_fusion(y_true, logits_1, logits_2, w1=0.6, w2=0.4):
    fused_logits = logits_1 * w1 + logits_2 * w2
    y1 = logits_1.argmax(axis=1)
    y2 = logits_2.argmax(axis=1)
    yf = fused_logits.argmax(axis=1)

    return {
        "acc1": accuracy_score(y_true, y1),
        "acc2": accuracy_score(y_true, y2),
        "acc_fused": accuracy_score(y_true, yf),
        "f1_1": f1_score(y_true, y1, average="macro"),
        "f1_2": f1_score(y_true, y2, average="macro"),
        "f1_fused": f1_score(y_true, yf, average="macro"),
        "y_pred_fused": yf
    }


def oracle_fusion(y_true, y_pred_1, y_pred_2):
    mask = (y_pred_1 == y_true) | (y_pred_2 == y_true)
    y_oracle = np.where(mask, y_true, y_pred_1)
    return {
        "acc": accuracy_score(y_true, y_oracle),
        "f1": f1_score(y_true, y_oracle, average="macro")
    }


# ---------------------------------------------------------
# EXPERIMENT 1: LGBM (IMU) + CNN (Pose) fusion
# ---------------------------------------------------------

def run_experiment_lgbm_cnn():
    acc_list = []
    f1_list = []
    acc_w_list = []
    f1_w_list = []
    acc_lgbm_list = []
    f1_lgbm_list = []
    acc_cnn_list = []
    f1_cnn_list = []
    

    for subj in range(1, 6):
        lgbm_path = f"../../A_imu_based_har/feature_models/lgbm_logits/fold_{subj}.csv"
        cnn_path = f"../../B_pose_based_har/smpl_models/CNN_embeddings/fold_{subj}.csv"

        df_cnn = pd.read_csv(cnn_path)
        df_lgbm = pd.read_csv(lgbm_path)

        merged = merge_embeddings(df_cnn, df_lgbm)
        merged = merged[merged["subject"] == subj]

        y_true = merged["y_true_x"].to_numpy()
        y_pred_cnn = merged["y_pred_cnn"].to_numpy()
        y_pred_lgbm = merged["y_pred_lgbm"].to_numpy()

        # LGBM
        acc_lgbm_list.append(accuracy_score(y_true, y_pred_lgbm))
        f1_lgbm_list.append(f1_score(y_true, y_pred_lgbm, average="macro"))

        # CNN
        acc_cnn_list.append(accuracy_score(y_true, y_pred_cnn))
        f1_cnn_list.append(f1_score(y_true, y_pred_cnn, average="macro"))

        # Oracle
        oracle = oracle_fusion(y_true, y_pred_cnn, y_pred_lgbm)
        acc_list.append(oracle["acc"])
        f1_list.append(oracle["f1"])

        # Weighted logits
        num_classes = 4
        log_cnn = merged[[f"logits_cnn_{c}" for c in range(num_classes)]].to_numpy()
        log_lgbm = merged[[f"logits_lgbm_{c}" for c in range(num_classes)]].to_numpy()

        fused = weighted_logit_fusion(y_true, log_cnn, log_lgbm, w1=0.75, w2=0.25)
        acc_w_list.append(fused["acc_fused"])
        f1_w_list.append(fused["f1_fused"])

    return acc_list, f1_list, acc_w_list, f1_w_list, acc_lgbm_list, f1_lgbm_list, acc_cnn_list, f1_cnn_list 


# ---------------------------------------------------------
# EXPERIMENT 2: DCL (IMU) + CNN (Pose) fusion
# ---------------------------------------------------------

def run_experiment_dcl_cnn():
    acc_dcl_list = []
    f1_dcl_list = []
    acc_cnn_list = []
    f1_cnn_list = []
    acc_oracle_list = []
    f1_oracle_list = []
    acc_w_list = []
    f1_w_list = []

    for subj in range(1, 6):
        dcl_path = f"../../A_imu_based_har/deep_models/DeepConvLSTM_embeddings/fold_{subj}.csv"
        cnn_path = f"../../B_pose_based_har/smpl_models/CNN_embeddings/fold_{subj}.csv"

        df_dcl = pd.read_csv(dcl_path)
        df_cnn = pd.read_csv(cnn_path)

        merged = merge_embeddings(df_dcl, df_cnn)
        merged = merged[merged["subject"] == subj]

        # Extract predictions
        y_true = merged["y_true_x"].to_numpy()
        dcl_pred = merged["y_pred_dcl"].to_numpy()
        cnn_pred = merged["y_pred_cnn"].to_numpy()

        # Individual model scores
        acc_dcl_list.append(accuracy_score(y_true, dcl_pred))
        f1_dcl_list.append(f1_score(y_true, dcl_pred, average="macro"))
        acc_cnn_list.append(accuracy_score(y_true, cnn_pred))
        f1_cnn_list.append(f1_score(y_true, cnn_pred, average="macro"))

        # Oracle
        oracle = oracle_fusion(y_true, dcl_pred, cnn_pred)
        acc_oracle_list.append(oracle["acc"])
        f1_oracle_list.append(oracle["f1"])

        # Weighted logits
        num_classes = 4
        log_dcl = merged[[f"logits_dcl_{c}" for c in range(num_classes)]].to_numpy()
        log_cnn = merged[[f"logits_cnn_{c}" for c in range(num_classes)]].to_numpy()

        fused = weighted_logit_fusion(y_true, log_dcl, log_cnn, w1=0.25, w2=0.75)
        acc_w_list.append(fused["acc_fused"])
        f1_w_list.append(fused["f1_fused"])

    return (
        acc_dcl_list, f1_dcl_list,
        acc_cnn_list, f1_cnn_list,
        acc_oracle_list, f1_oracle_list,
        acc_w_list, f1_w_list
    )



# -------------------------------------------
# Seed
# -------------------------------------------

def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":
    set_global_seed(42)
    acc_lgbm_oracle, f1_lgbm_oracle, acc_w_lgbm, f1_w_lgbm, acc_lgbm_list, f1_lgbm_list, acc_cnn_list, f1_cnn_list  = run_experiment_lgbm_cnn()

    results_dcl = run_experiment_dcl_cnn()

    # Pretty printing
    def summarize(name, accs, f1s):
        print(f"{name}: acc={np.mean(accs):.3f}±{np.std(accs):.3f}, "
              f"f1={np.mean(f1s):.3f}±{np.std(f1s):.3f}")

    print("\n===== SUMMARY =====")
    print("\n===== Experiments: LGBM (IMU) + CNN (POSE) =====")

    summarize("LGBM (A1)", acc_lgbm_list, f1_lgbm_list)
    summarize("CNN", acc_cnn_list, f1_cnn_list)
    summarize("LGBM-CNN Weighted Ensemble Fusion", acc_w_lgbm, f1_w_lgbm)
    summarize("LGBM-CNN Oracle Fusion", acc_lgbm_oracle, f1_lgbm_oracle)

    (
        acc_dcl, f1_dcl,
        acc_cnn, f1_cnn,
        acc_oracle, f1_oracle,
        acc_w, f1_w
    ) = results_dcl

    print("\n===== Experiments: DCL (IMU) + CNN (POSE) =====")

    summarize("DCL (A2)", acc_dcl, f1_dcl)
    summarize("CNN (B1)", acc_cnn, f1_cnn)
    summarize("DCL-CNN Weighted Ensemble Fusion (C1)", acc_w, f1_w)
    summarize("DCL-CNN Oracle Fusion (C2)", acc_oracle, f1_oracle)

    
    print("\n==================================")
    print("Disclaimer: Oracle Fusion assumes knowing which labels are true! = Data leak on purpose")
