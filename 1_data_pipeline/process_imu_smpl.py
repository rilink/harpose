"""
Extracts SMPL and IMU data, windows them, and exports them.
"""

import os
import glob
import pickle
import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Configuration / Constants
# ------------------------------------------------------------

PKL_FILES = glob.glob(
    os.path.join('..', '1_data_pipeline', 'data', '**', '*.pkl'),
    recursive=True
)

subjects = [1, 2, 3, 4, 5]

activities_dict = {
    1: 'acting',
    2: 'freestyle',
    3: 'rom',
    4: 'walking'
}

activities_num_dict = {
    10: 'acting1',   11: 'acting2',   12: 'acting3',
    20: 'freestyle1', 21: 'freestyle2', 22: 'freestyle3',
    30: 'rom1',        31: 'rom2',       32: 'rom3',
    40: 'walking1',    41: 'walking2',   42: 'walking3'
}


# ------------------------------------------------------------
# Windowing
# ------------------------------------------------------------

def window_data(data, window_size=100, stride=50):
    """Create overlapping windows from time-series data."""
    T = len(data)
    if T < window_size:
        return np.zeros((0, window_size) + data.shape[1:])
    windows = [
        data[start:start + window_size]
        for start in range(0, T - window_size + 1, stride)
    ]
    return np.array(windows)


# ------------------------------------------------------------
# Loading and preprocessing
# ------------------------------------------------------------

def load_pkl(path):
    """Load a pkl file with latin1 encoding."""
    with open(path, 'rb') as f:
        return pickle.load(f, encoding='latin1')


def extract_modalities(data):
    """Extract raw IMU and SMPL modalities in array form."""
    acc = np.array(data['acc'][1:])
    gyr = np.array(data['gyr'][1:])
    mag = np.array(data['mag'][1:])
    smpl = np.array(data['smpl_poses'])
    print(acc.shape)
    print(gyr.shape)
    print(mag.shape)
    print(smpl.shape)
    return acc, gyr, mag, smpl


def truncate_modalities(acc, gyr, mag, smpl, window_size):
    """Truncate modalities to aligned lengths and validate frame count."""
    min_frames = min(len(acc), len(gyr), len(mag), len(smpl))
    if min_frames < window_size:
        return None
    return acc[:min_frames], gyr[:min_frames], mag[:min_frames], smpl[:min_frames]


# ------------------------------------------------------------
# Window extraction and row building
# ------------------------------------------------------------

def create_signal_rows(acc_w, gyr_w, mag_w):
    """Expand IMU windows into flattened column dict."""
    row = {}
    signals = {"acc": acc_w, "gyr": gyr_w, "mag": mag_w}

    for signal_name, window in signals.items():
        for sensor_idx in range(window.shape[1]):
            for axis_idx, axis_name in enumerate(["x", "y", "z"]):
                col_name = f"{signal_name}_{axis_name}_{sensor_idx+1}"
                row[col_name] = window[:, sensor_idx, axis_idx].tolist()
    return row


def create_smpl_rows(smpl_window):
    """Expand SMPL window (window_size × 72) into column dict."""
    return {
        f"smpl_{i}": smpl_window[:, i].tolist()
        for i in range(smpl_window.shape[1])
    }


# ------------------------------------------------------------
# File processing
# ------------------------------------------------------------

def process_single_file(path, subject, activity, window_size, stride):
    """Process one PKL file if matching subject + activity."""
    tag = activities_num_dict[activity].lower()
    if f"s{subject}" not in path or tag not in path.lower():
        return []

    data = load_pkl(path)
    acc, gyr, mag, smpl = extract_modalities(data)

    truncated = truncate_modalities(acc, gyr, mag, smpl, window_size)
    if truncated is None:
        return []

    acc, gyr, mag, smpl = truncated

    acc_w = window_data(acc, window_size, stride)
    gyr_w = window_data(gyr, window_size, stride)
    mag_w = window_data(mag, window_size, stride)
    smpl_w = window_data(smpl, window_size, stride)

    n = min(len(acc_w), len(gyr_w), len(mag_w), len(smpl_w))
    rows = []

    for i in range(n):
        meta = {
            "activity_num": activity,
            "activity": activities_dict[int(str(activity)[0])],
            "activity_encoded": str(activity)[0],
            "subject": subject,
            "file_path": path,
            "window_idx": i
        }
        row = {}
        row.update(meta)
        row.update(create_signal_rows(acc_w[i], gyr_w[i], mag_w[i]))
        row.update(create_smpl_rows(smpl_w[i]))
        rows.append(row)

    return rows


def extract_data_from_pkl(subjects, activities, window_size, stride, save=False):
    """Iterate through PKL files and extract windowed samples."""
    all_rows = []

    for path in PKL_FILES:
        for subject in subjects:
            for activity in activities:
                rows = process_single_file(path, subject, activity, window_size, stride)
                all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    if save:
        subj_str = "_".join(map(str, subjects))
        act_str = "_".join(map(str, activities))
        out_path = (
            f"results_smpl_and_imu_fusion/"
            f"subj{subj_str}_act{act_str}_ws{window_size}_str{stride}.csv"
        )
        df.to_csv(out_path, index=False)
        print(f"Saved CSV to {out_path}")

    return df


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

samples_per_run = []

for subject in subjects:
    for activity in activities_num_dict:
        samples = extract_data_from_pkl(
            subjects=[subject],
            activities=[activity],
            window_size=100,
            stride=50,
            save=False
        )
        samples_per_run.append(samples)

all_samples = pd.concat(samples_per_run, ignore_index=True)
all_samples.to_csv('processed_data/windowed_smpl_imu.csv', index=False)
