# harpose/1_data_pipeline

harpose/1_data_pipeline: The data is to be stored in .pkl files. For each subject a separate folder named ```si``` for $i \in \{1,\dots,5\}$ where each activity recording of each subject is stored as a separate pkl file stored as, e.g., ```acting1.pkl``` and ```walking2.pkl```. 
The .pkl files are to be constructed in a way that matches these contents of ```process_imu_smpl.py```

```
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


data = load_pkl(path)
acc, gyr, mag, smpl = extract_modalities(data)

```
Returning e.g. for s1/acting1.pkl
```
(4113, 6, 3)
(4113, 6, 3)
(4113, 6, 3)
(4113, 72)
```
where there are 4113 frames, 6 selected sensor locations of 3D sensor data for acc, gyr, mag and 72 for SMPL for the 24 joints à 3 parameters per axis-angle.   
The six chosen IMU locations are 
```
TC_six_imus = {9: 'L_LowLeg',
                10: 'R_LowLeg',
                5: 'L_LowArm',
                6: 'R_LowArm',
                0: 'Head', 
                2: 'Pelvis'}
```