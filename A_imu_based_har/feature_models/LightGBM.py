import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from lightgbm import LGBMClassifier
import lightgbm as lgb


activities = {
    1: 'acting',
    2: 'freestyle',
    3: 'rom',
    4: 'walking'
}

activities_num_dict = {
    10: 'acting1', 11: 'acting2', 12: 'acting3',
    20: 'freestyle1', 21: 'freestyle2', 22: 'freestyle3',
    30: 'rom1', 31: 'rom2', 32: 'rom3',
    40: 'walking1', 41: 'walking2', 42: 'walking3'
}


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

        print(f"\n=== LOSO Fold: Subject {test_subj} held out ===")
        print(f"Train subjects: {train_df['subject'].unique()}")
        print(f"Test subject: {test_df['subject'].unique()}")

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
    print(x_train.columns)
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
    )

    model.fit(x_train, y_train)

    predicted = model.predict(x_test)
    accuracy = accuracy_score(y_test, predicted)
    macro_f1 = f1_score(y_test, predicted, average='macro')

    print(f"Accuracy on test data: {accuracy:.3f}")
    print(f"Macro F1 score on test data: {macro_f1:.3f}")

    return model, x_train, y_train, x_test, y_test, predicted, accuracy, macro_f1


def plot_confusion_matrix(y_test, predicted):
    """
    Plots confusion, precision, and recall heatmaps.
    """
    labels = list(activities.keys())
    labels_words = list(activities.values())

    C = confusion_matrix(y_test, predicted, labels=labels)
    A = (((C.T) / (C.sum(axis=1))).T)   # Recall
    B = (C / C.sum(axis=0))            # Precision

    # Confusion matrix
    plt.figure(figsize=(17, 7))
    sns.heatmap(C, annot=True, cmap="Greens", fmt=".3f",
                xticklabels=labels_words, yticklabels=labels_words)
    plt.xlabel('Predicted Class')
    plt.ylabel('Ground Truth Class')
    plt.show()

    # Precision matrix
    plt.figure(figsize=(17, 7))
    sns.heatmap(B, annot=True, cmap="Greens", fmt=".3f",
                xticklabels=labels_words, yticklabels=labels_words)
    plt.xlabel('Predicted Class')
    plt.ylabel('Ground Truth Class')
    plt.tight_layout()
    plt.show()

    # Recall matrix
    plt.figure(figsize=(17, 7))
    sns.heatmap(A, annot=True, cmap="Greens", fmt=".3f",
                xticklabels=labels_words, yticklabels=labels_words)
    plt.xlabel('Predicted Class')
    plt.ylabel('Ground Truth Class')
    plt.tight_layout()
    plt.show()


def countplot_activities(train_df, test_df):
    """
    Plots a count distribution of activity labels in the training set.
    """
    plt.figure(figsize=(10, 4))
    sns.countplot(x='activity', data=train_df)
    plt.xlabel("Activity")
    plt.ylabel("Count")
    plt.show()


def plot_ft_importances(model):
    """
    Plots LightGBM feature importances (split importance).
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    lgb.plot_importance(model, ax=ax, importance_type='split')
    plt.title("LightGBM Feature Importance")
    plt.tight_layout()
    plt.show()

def export_lightgbm_csv(
        X_train, X_test,
        y_train, y_test,
        train_df, test_df,
        gbm,
        num_classes,
        output_folder,
        test_subj
    ):
    """
    Creates a CSV for LightGBM in the same structure as your DCL/CNN exported files.
    """

    # -----------------------------
    # Predict logits and labels
    # -----------------------------
    logits_train = gbm.predict(X_train, raw_score=True)
    logits_test  = gbm.predict(X_test, raw_score=True)

    # Convert raw scores to numpy if needed
    logits_train = np.array(logits_train)
    logits_test  = np.array(logits_test)

    # Predicted class indices
    y_pred_train = logits_train.argmax(axis=1)
    y_pred_test  = logits_test.argmax(axis=1)

    # -----------------------------
    # Build empty dataframes
    # -----------------------------
    emb_train_df = pd.DataFrame()
    emb_test_df = pd.DataFrame()

    # Add truth + predictions
    emb_train_df["y_true"] = y_train -1 
    emb_train_df["y_pred_lgbm"] = y_pred_train
    emb_train_df["idx"] = np.arange(len(y_train))
    emb_train_df["split"] = "train"

    emb_test_df["y_true"] = y_test -1
    emb_test_df["y_pred_lgbm"] = y_pred_test
    emb_test_df["idx"] = np.arange(len(y_test))
    emb_test_df["split"] = "test"

    # -----------------------------
    # Add logits (one column per class)
    # -----------------------------
    for c in range(num_classes):
        emb_train_df[f"logits_lgbm_{c}"] = logits_train[:, c]
        emb_test_df[f"logits_lgbm_{c}"] = logits_test[:, c]

    # -----------------------------
    # Add embeddings (X)
    # -----------------------------
    # for d in range(X_train.shape[1]):
    #     emb_train_df[f"emb_{d}"] = X_train[:, d]
    # for d in range(X_test.shape[1]):
    #     emb_test_df[f"emb_{d}"] = X_test[:, d]

    # -----------------------------
    # Attach metadata (subject, activity, file_path, window_idx)
    # -----------------------------
    meta_train = train_df.reset_index()[["subject", "activity", "file_path", "window_idx"]]
    meta_test  = test_df.reset_index()[["subject", "activity", "file_path", "window_idx"]]

    combined_train = pd.concat([meta_train.reset_index(drop=True),
                                emb_train_df.reset_index(drop=True)], axis=1)

    combined_test = pd.concat([meta_test.reset_index(drop=True),
                               emb_test_df.reset_index(drop=True)], axis=1)

    combined = pd.concat([combined_train, combined_test], ignore_index=True)

    # -----------------------------
    # Save CSV
    # -----------------------------
    os.makedirs(output_folder, exist_ok=True)
    out_path = os.path.join(output_folder, f"fold_{test_subj}.csv")
    combined.to_csv(out_path, index=False)

    print(f"[LightGBM] Exported: {out_path}   ({len(combined)} rows)")
    return combined


if __name__ == "__main__":
    macro_f1s = []
    accs = []

    df_path = "data_features/imu_features.csv"
    output_folder = "lgbm_embeddings"
    os.makedirs(output_folder, exist_ok=True)

    for test_subj, data_split in LOSO(df_path):
        print(f"\n===== Test Subject {test_subj} =====")

        train_df = data_split['train']['df']
        test_df = data_split['test']['df']

        # -----------------------------------------------------
        # Train LightGBM & get predictions + metrics
        # -----------------------------------------------------
        model, X_train, y_train, X_test, y_test, y_pred, accuracy, macro_f1 = \
            train_and_evaluate_lgbm(train_df, test_df, drop_cols=None)

        macro_f1s.append(macro_f1)
        accs.append(accuracy)

        print(f"Acc = {accuracy:.3f}, Macro-F1 = {macro_f1:.3f}")

        # -----------------------------------------------------
        # Export CSV with logits + embeddings (NEW FUNCTION)
        # -----------------------------------------------------
        export_lightgbm_csv(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            train_df=train_df,
            test_df=test_df,
            gbm=model,
            num_classes=len(np.unique(y_train)),
            output_folder=output_folder,
            test_subj=test_subj
        )

    # ---------------------------------------------------------
    # Final LOSO summary
    # ---------------------------------------------------------
    print("\n====== LOSO Summary ======")
    print(f"Average macro F1: {np.mean(macro_f1s):.3f} ± {np.std(macro_f1s):.3f}")
    print(f"Average accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f}")


# if __name__ == "__main__":
#     macro_f1s = []
#     accs = []

#     df_path = 'data_features/imu_features.csv'

#     for test_subj, data_split in LOSO(df_path):
#         train_df = data_split['train']['df']
#         test_df = data_split['test']['df']

#         model, x_train, y_train, x_test, y_test, predicted, accuracy, macro_f1 = \
#             train_and_evaluate_lgbm(train_df, test_df, drop_cols=None)

#         macro_f1s.append(macro_f1)
#         accs.append(accuracy)

#     print("\n====== LOSO Summary ======")
#     print(f"Average macro F1: {np.mean(macro_f1s):.3f} ± {np.std(macro_f1s):.3f}")
#     print(f"Average accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    

