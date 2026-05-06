import ast
import wfdb
import pandas as pd
import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt
from pathlib import Path
from skfda.representation import FDataGrid

#### ---- Dataset Source ----  ####
# https://physionet.org/content/ptb-xl/1.0.3/

#### ---- Global Parameters ---- ####
path = "../../ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"
# path = '~/projects/def-chenh/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/'
sampling_rate=100
diagnostics = np.array(['NORM', 'MI', 'STTC', 'CD', 'HYP'])

def get_diagnostics():
    return diagnostics

def get_sr():
    return sampling_rate


#### ---- Load PTB-XL Dataset ---- ####
def aggregate_diagnostic(y_dic):
    agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

def get_data():
    """
    Load PTB-XL dataset into .npy files
    """
    save_path = "../data"
    Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

    noise_cols = ['baseline_drift', 'static_noise', 'burst_noise', 'electrodes_problems', 'extra_beats', 'pacemaker']
    Y = Y[Y[noise_cols].isna().all(axis=1)]

    # Work with normal ECG now
    for sampling_rate in [100, 500]:
        for diagnostic in diagnostics:
            if sampling_rate == 100:
                data = [wfdb.rdsamp(path+f) for f in Y.filename_lr if Y.loc[Y.filename_lr == f].diagnostic_superclass.values[0] == [diagnostic]]
            else:
                data = [wfdb.rdsamp(path+f) for f in Y.filename_hr if Y.loc[Y.filename_hr == f].diagnostic_superclass.values[0] == [diagnostic]]
            data = np.array([signal for signal, meta in data])
            data = data.transpose(0, 2, 1) # (n_records, n_leads, n_samples)

            np.save(save_path + f"/{diagnostic}_{sampling_rate}.npy", data)

#### ---- Load Dataset ---- ####
def load_dataset(diagnostic, sampling_rate, lead=None):
    data = np.load(f"../data/{diagnostic}_{sampling_rate}.npy")
    if lead is not None:
        data = data[:, lead, :]
    return data

def load_synthetic_dataset(diagnostic, lead):
    data = np.load("../data/synthetic_final.npy")
    label = np.load("../data/synthetic_final_labels.npy")

    class_index = np.where(diagnostics == diagnostic)[0][0]
    mask = (label[:, class_index] == 1)

    filtered = data[mask][:, lead, :]

    return filtered

#### --- ECG Preprocess ---- ####
# Landmark registration by clinical features
def extract_ecg_clinical_landmarks(signals, n_beats, sr):
    """
    Use R peaks as landmarks for landmark registration
    """
    landmarks = []
    data = []
    for signal in signals:
        df, _ = nk.ecg_process(signal, sampling_rate=sampling_rate)
        p_onsets = list(filter(lambda i: df['ECG_P_Peaks'].tolist()[i] == 1, range(len(df['ECG_P_Onsets'].tolist()))))
        # p_peaks = list(filter(lambda i: df['ECG_P_Peaks'].tolist()[i] == 1, range(len(df['ECG_P_Peaks'].tolist()))))
        # q_peaks = list(filter(lambda i: df['ECG_Q_Peaks'].tolist()[i] == 1, range(len(df['ECG_Q_Peaks'].tolist()))))
        r_peaks = list(filter(lambda i: df['ECG_R_Peaks'].tolist()[i] == 1, range(len(df['ECG_R_Peaks'].tolist()))))
        # s_peaks = list(filter(lambda i: df['ECG_S_Peaks'].tolist()[i] == 1, range(len(df['ECG_S_Peaks'].tolist()))))
        # t_peaks = list(filter(lambda i: df['ECG_T_Peaks'].tolist()[i] == 1, range(len(df['ECG_T_Peaks'].tolist()))))
        t_offsets = list(filter(lambda i: df['ECG_T_Offsets'].tolist()[i] == 1, range(len(df['ECG_T_Offsets'].tolist()))))
        
        if len(r_peaks) < n_beats: continue
        if len(r_peaks) == len(p_onsets) == len(t_offsets):
            landmark = r_peaks[:n_beats]
            landmark.sort()
            if 0 in np.diff(landmark): continue

            first_p = p_onsets[0]
            end_t = t_offsets[n_beats-1]
            landmark = [peak - first_p for peak in landmark]

            trimmed_signal = signal[first_p:end_t]
            landmarks.append(landmark)
            data.append(trimmed_signal)
            
    # Commmon Grid Alignment
    common_grid = np.linspace(0, 1, n_beats * sr)
    resampled_data = []
    for i in range(len(data)):
        n = len(data[i])
        t = np.linspace(0, 1, n)
        fd = FDataGrid(data_matrix=data[i], grid_points=t)
        resampled = fd(common_grid).squeeze()
        resampled_data.append(resampled)
        landmarks[i] = [landmark / n for landmark in landmarks[i]]

    fd = FDataGrid(data_matrix=resampled_data, grid_points=common_grid)
    landmarks = np.array(landmarks)

    return fd, landmarks

if __name__ == "__main__":
    # get_data()
    synthetic = load_synthetic_dataset("CD", 1)

    

    