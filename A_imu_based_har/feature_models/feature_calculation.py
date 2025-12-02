import os
import math
import ast
import numpy as np
import pandas as pd

from scipy.stats import iqr, entropy, kurtosis, skew
from scipy.signal import welch, find_peaks
from scipy.fftpack import fft


# ============================================================
# Utility
# ============================================================

def safe_literal_eval(val):
    try:
        return ast.literal_eval(val)
    except Exception:
        return []


def signal_magnitude_vector(window):
    return np.sqrt(np.sum(np.square(window), axis=1))


# ============================================================
# Feature helpers
# ============================================================

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


# ============================================================
# 1D FEATURE CALC
# ============================================================

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


# ============================================================
# 3D FEATURE CALC
# ============================================================

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


# ============================================================
# MAIN FEATURE PIPELINE
# ============================================================

def feature_calculation(
    df,
    output_path="processed_data/one_sec_data_features.csv",
    features_1D=None,
    features_3D=None,
):

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
    df[imu_columns] = df[imu_columns].applymap(safe_literal_eval)

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
    result_df.to_csv(output_path, index=False)
    return result_df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    df = pd.read_csv('../../1_data_pipeline/processed_data/windowed_smpl_imu.csv')
    df = df.loc[:, ~df.columns.str.contains('smpl')]
    out = feature_calculation(df, output_path="data_features/imu_features.csv")
    print(out)




# TODO: currently it also uses SMPL for feature calculation results of Light GBM with SMPL: 
# TODO this would be early fusion, right?? # # ====== LOSO Summary ======
# Average macro F1: 0.564 ± 0.048
# Average accuracy: 0.738 ± 0.041
# TODO remove smpl to restore imu only lightgbm + feature calc
# TODO put smpl model into paper ?! 

# import os
# import numpy as np
# import pandas as pd
# from scipy.stats import iqr, entropy
# from scipy.fftpack import fft
# from scipy.stats import entropy
# from scipy.signal import find_peaks
# import ast
# from scipy.stats import iqr, kurtosis, skew
# from scipy.signal import welch
# #from scipy.integrate import simps
# import math
# from collections import Counter


# def safe_literal_eval(val):
#     try:
#         return ast.literal_eval(val)
#     except (ValueError, SyntaxError, TypeError):
#         return [] 

# def signal_magnitude_vector(window):
#     magnitude = np.sqrt(np.sum(np.square(window), axis=1)) 
#     return magnitude

# def spectral_entropy(signal, sf=50):
#     freqs, psd = welch(signal, sf)
#     psd_norm = psd / np.sum(psd)
#     entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))  
#     return entropy

# def hjorth_parameters(signal):
#     first_deriv = np.diff(signal)
#     second_deriv = np.diff(first_deriv)
#     var_zero = np.var(signal)
#     var_d1 = np.var(first_deriv)
#     var_d2 = np.var(second_deriv)
#     mobility = np.sqrt(var_d1 / var_zero) if var_zero != 0 else 0
#     complexity = np.sqrt(var_d2 / var_d1) / mobility if var_d1 != 0 else 0
#     return mobility, complexity

# def mean_crossing_rate(signal):
#     mean_val = np.mean(signal)
#     crossings = np.where(np.diff(np.signbit(signal - mean_val)))[0]
#     return len(crossings) / len(signal)

# def differential_entropy(signal):
#     var = np.var(signal)
#     return 0.5 * np.log(2 * np.pi * np.e * var)


# #https://github.com/raphaelvallat/entropy/blob/master/entropy/fractal.py
# def petrosian_fd(signal):
#     diff = np.diff(signal)
#     N_delta = np.sum(diff[1:] * diff[:-1] < 0)
#     n = len(signal)
#     return np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * N_delta)))

# def katz_fd(signal):
#     L = np.sum(np.sqrt(np.square(np.diff(signal))))
#     d = np.max(np.abs(signal - signal[0]))
#     n = len(signal)
#     return np.log10(n) / (np.log10(d / L + 1e-6))  

# def orientation(x, y, z, dt=0.02):     
#     acc_data = np.stack([x, y, z], axis=1)
#     velocity = np.cumsum(acc_data * dt, axis=0)
#     position = np.cumsum(velocity * dt, axis=0)
#     pos_x = position[-1][0]
#     pos_y = position[-1][1]
#     pos_z = position[-1][2]
#     velo_acc = np.sum(velocity[-1])
#     pos_acc = np.sum(position[-1])
#     return pos_x, pos_y, pos_z, velo_acc, pos_acc

# def calculate_1D_features(signal, features, axis_name):
#     result = {}    
#     signal = np.array(signal)

#     # time-domain features
#     if 'mean' in features:
#         result[f'{axis_name}_mean'] = np.mean(signal)
#     if 'var' in features:
#         result[f'{axis_name}_var'] = np.var(signal)
#     if 'std' in features:
#         result[f'{axis_name}_std'] = np.std(signal)
#     if 'rms' in features:
#         result[f'{axis_name}_rms'] = np.sqrt(np.mean(signal**2))
#     if 'shape_factor' in features: 
#         mv = np.mean(signal)
#         rms = np.sqrt(np.mean(signal**2))
#         shape_factor = rms / np.abs(mv)
#         result[f'{axis_name}_shape_factor'] = shape_factor

#     if 'min' in features:
#         result[f'{axis_name}_min'] = np.min(signal)
#     if 'max' in features:
#         result[f'{axis_name}_max'] = np.max(signal)
#     if 'median' in features:
#         result[f'{axis_name}_median'] = np.median(signal)
#     if 'iqr' in features:
#         result[f'{axis_name}_iqr'] = iqr(signal)
#     if 'sma' in features:
#         result[f'{axis_name}_sma'] = np.sum(np.abs(signal)) / len(signal)
#     if 'skewness' in features:
#         result[f'{axis_name}_skewness'] = skew(signal)
#     if 'kurtosis' in features:
#         result[f'{axis_name}_kurtosis'] = kurtosis(signal)
#     if 'jerk_mean' in features: 
#         j = np.diff(signal)
#         result[f'{axis_name}_jerk_mean'] = np.mean(np.abs(j))
#     if 'jerk_std' in features:
#         j = np.diff(signal)
#         result[f'{axis_name}_jerk_std'] = np.std(j)
#     if 'num_peak' in features: 
#         peaks, _ = find_peaks(signal, distance=3)
#         result[f'{axis_name}_peaks'] = len(peaks)
#     if 'zero_crossings' in features: 
#         result[f'{axis_name}_zero_crossings'] = np.sum((signal[:-1] * signal[1:]) < 0)

#     # frequency-domain features
#     fft_vals = np.abs(fft(signal))
#     fft_vals = fft_vals[:len(fft_vals)//2]  
#     if 'energy' in features:
#         result[f'{axis_name}_energy'] = np.sum(fft_vals ** 2)
#     if 'dominant_freq' in features:
#         result[f'{axis_name}_dominant_freq'] = np.argmax(fft_vals)
#     if 'entropy' in features:
#         prob_dist = fft_vals / np.sum(fft_vals)
#         result[f'{axis_name}_entropy'] = entropy(prob_dist)
#     if 'signal_entropy' in features: 
#         hist, _ = np.histogram(signal, bins=10, density=True)
#         result[f'{axis_name}_signal_entropy'] = entropy(hist + 1e-10)  
#     return result

# def calculate_3D_features(x_y_z_list, features):
#     """
#     3D features are calculated here, i.e. all three axis of the accelerometer measurements are used to calculate the features
#     """
#     result = {}
#     x = x_y_z_list[0]
#     y = x_y_z_list[1]
#     z = x_y_z_list[2]
#     # x = np.asarray(row['x_axis'])
#     # y = np.asarray(row['y_axis'])
#     # z = np.asarray(row['z_axis'])
#     smv = signal_magnitude_vector(np.column_stack((x, y, z)))
#     x_mean, y_mean, z_mean = np.mean(x), np.mean(y), np.mean(z)

#     if 'pos_x' or 'pos_y' or 'pos_z' or 'velo_acc' or 'pos_acc' in features:
#         pos_x, pos_y, pos_z, velo_acc, pos_acc = orientation(x,y,z, dt=0.02)
#         if 'pos_x' in features: 
#             result['pos_x'] = pos_x
#         if 'pos_y' in features: 
#             result['pos_y'] = pos_y
#         if 'pos_z' in features: 
#             result['pos_z'] = pos_z
#         if 'velo_acc' in features: 
#             result['velo_acc'] = velo_acc
#         if 'pos_acc' in features: 
#             result['pos_acc'] = pos_acc

#     if 'pitch' in features:
#         pitch = np.degrees(np.arctan2(-x_mean, np.sqrt(y_mean**2 + z_mean**2)))
#         result['pitch'] = pitch
#     if 'roll' in features:
#         roll = np.degrees(np.arctan2(y_mean, z_mean))
#         result['roll'] = roll
#     if 'total_sqrt' in features:
#         result['total_sqrt'] = np.sqrt(x_mean**2 + y_mean**2 + z_mean**2) 
#     if 'total' in features:
#         result['total'] = x_mean + y_mean + z_mean
#     if 'signal_magnitude' in features:
#         result['signal_magnitude'] = np.sum((x) + (y) + (z)) / len(x)
#     if 'signal_magnitude_abs' in features:
#         result['signal_magnitude_abs'] = np.sum(np.abs(x) + np.abs(y) + np.abs(z)) / len(x)

#     if 'axis_correlation' in features:
#         result.update({
#             'axis_correlation_x_y': np.corrcoef(x, y)[0, 1],
#             'axis_correlation_x_z': np.corrcoef(x, z)[0, 1],
#             'axis_correlation_y_z': np.corrcoef(y, z)[0, 1],
#         })

#     if 'axis_covariance' in features:
#         result.update({
#             'axis_covariance_x_y': np.cov(x, y)[0, 1],
#             'axis_covariance_x_z': np.cov(x, z)[0, 1],
#             'axis_covariance_y_z': np.cov(y, z)[0, 1],
#         })

#     if 'tilt_angle' in features:
#         norm = np.sqrt(x_mean**2 + y_mean**2 + z_mean**2) + 1e-6
#         result['tilt_angle'] = np.degrees(np.arccos(z_mean / norm))

#     if 'spectral_entropy' in features:
#         result['spectral_entropy'] = spectral_entropy(smv)
#     if 'min_smv' in features:
#         result['min_smv'] = np.min(smv)
#     if 'max_smv' in features:
#         result['max_smv'] = np.max(smv)
#     if 'ptp_smv' in features:
#         result['ptp_smv'] = np.ptp(smv)
#     if 'iqr_smv' in features:
#         result['iqr_smv'] = iqr(smv)
#     if 'std_smv' in features:
#         result['std_smv'] = np.std(smv)
#     if 'skew_smv' in features:
#         result['skew_smv'] = skew(smv)
#     if 'kurtosis_smv' in features:
#         result['kurtosis_smv'] = kurtosis(smv)
#     if 'mobility' in features or 'complexity' in features:
#         mobility, complexity = hjorth_parameters(smv)
#         if 'mobility' in features:
#             result['mobility'] = mobility
#         if 'complexity' in features:
#             result['complexity'] = complexity
#     if 'mean_crossing' in features:
#         result['mean_crossing'] = mean_crossing_rate(smv)
#     if 'differential_entropy' in features:
#         result['differential_entropy'] = differential_entropy(smv)
#     if 'petrosian_fd' in features:
#         result['petrosian_fd'] = petrosian_fd(smv)
#     if 'katz_fd' in features:
#         result['katz_fd'] = katz_fd(smv)
#     return result

# ################################################################################################################################################

# def feature_calculation(
#     df, #raw sensor data
#     output_path='processed_data\\one_sec_data_features.csv',
#     features_1D=['mean', 'var', 
#                  'rms', 'shape_factor', # new features
#                  'min', 'max', 'median', 'iqr', 'sma', 'energy', 'dominant_freq', 
#                  'entropy', 'signal_entropy', 'jerk_mean', 'jerk_std', 'num_peak', 'zero_crossings'
#                  ],
#     # # add again later!! #TODO
#     features_3D=[
#         'pos_x', 'pos_y', 'pos_z', 'velo_acc', 'pos_acc', # trajectory features
#         'pitch', 'roll', #orienation features
#         'total', 'total_sqrt', 'signal_magnitude', 'signal_magnitude_abs', 
#         'axis_correlation', 'axis_covariance', 'tilt_angle',
#         'spectral_entropy', 'min_smv', 'max_smv', 'ptp_smv', 'iqr_smv', 'std_smv', 'skew_smv',
#         'kurtosis_smv', 'mobility', 'complexity', 'mean_crossing', 'differential_entropy', 'petrosian_fd', 'katz_fd'
#          ],
#     add_features=None, # if provided the new features will be calculated (not the old ones recalculated)
#     remove_features=None, # if provided the features to be removed will be removed (not the old ones recalculated)
#     update_existing=False
# ):

#     df_columns = df.columns 
#     df_imu_columns = df_columns[6:]
#     print(df_imu_columns)
#     for axis in df_imu_columns:
#         df[axis] = df[axis].apply(safe_literal_eval)
#     print('Done literal.')

#     feature_rows = []
#     for idx, row in df.iterrows():
#         #print(idx,row)
#         if all(row[axis]!= [] for axis in df_imu_columns):
#             base_info = row.drop(df_imu_columns).to_dict()
#             #print(base_info)
#             new_features = {}
#             for axis in df_imu_columns:
#                 stats = calculate_1D_features(row[axis], features_1D, axis_name=axis)
#                 for col_name, col_val in stats.items():
#                     new_features[col_name] = col_val

#             if features_3D:
#                 for df_imu_column in df_imu_columns: 
#                     # 3D features for acclerometer only. Add gyr or mag?? TODO
#                     if 'acc' in df_imu_column:
#                         for i in range(1,7): # all 6 sensors
#                             if str(i) in df_imu_column: 
#                                 if 'x' in df_imu_column: 
#                                     x = np.asarray(row[df_imu_column])
#                                 if 'y' in df_imu_column:
#                                     y = np.asarray(row[df_imu_column])
#                                 if 'z' in df_imu_column:
#                                     z = np.asarray(row[df_imu_column])   
#                                     x_y_z_list = [x, y, z]
#                                     dim3_stats = calculate_3D_features(x_y_z_list, features_3D)
#                                     for col_name, col_val in dim3_stats.items():
#                                         new_features[col_name + '_acc'] = col_val
#                     if 'gyr' in df_imu_column:
#                         for i in range(1,7): # all 6 sensors
#                             if str(i) in df_imu_column: 
#                                 if 'x' in df_imu_column: 
#                                     x = np.asarray(row[df_imu_column])
#                                 if 'y' in df_imu_column:
#                                     y = np.asarray(row[df_imu_column])
#                                 if 'z' in df_imu_column:
#                                     z = np.asarray(row[df_imu_column])   
#                                     x_y_z_list = [x, y, z]
#                                     dim3_stats = calculate_3D_features(x_y_z_list, features_3D)
#                                     for col_name, col_val in dim3_stats.items():
#                                         new_features[col_name + '_gyr'] = col_val

#             base_info.update(new_features)
#             feature_rows.append(base_info)

#     final_df = pd.DataFrame(feature_rows)
#     final_df.to_csv(output_path, index=False)
#     return final_df


# if __name__ == "__main__":
#     one_sec_data = pd.read_csv('../../1_data_pipeline/processed_data/windowed_smpl_imu.csv', index_col=False)
#     one_sec_data_with_features = feature_calculation(one_sec_data, output_path='processed_data\\imu_features.csv')
#     print(one_sec_data_with_features)
#     # Logbook: 31.10.: without last three rows of 3D features and only for acc
#     # TODO add: 'file_path','window_idx fix 3:, drop out path etc. !! 

