import ast
import wfdb
import pandas as pd
import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

#### ---- Dataset Source ----  ####
# https://physionet.org/content/ptb-xl/1.0.3/

#### ---- Global Parameters ---- ####
path = "../ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"
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

def get_data(diagnostic, lead=None, holdout=False):
    """
    Docstring for get_data

    :param diagnostic: The diagnostic class to filter by [NORM, MI, STTC, CD, HYP]
    :param lead: The target lead to use. Default returns the entire dataset
    :param holdout: Split into train and holdout sets if True
    """
    Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

    noise_cols = ['baseline_drift', 'static_noise', 'burst_noise', 'electrodes_problems', 'extra_beats', 'pacemaker']
    Y = Y[Y[noise_cols].isna().all(axis=1)]

    # Work with normal ECG now
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in Y.filename_lr if Y.loc[Y.filename_lr == f].diagnostic_superclass.values[0] == diagnostic]
    else:
        data = [wfdb.rdsamp(path+f) for f in Y.filename_hr if Y.loc[Y.filename_hr == f].diagnostic_superclass.values[0] == diagnostic]

    data = np.array([signal for signal, meta in data])
    data = data.transpose(0, 2, 1)
    n_records, _, _ = data.shape

    if lead:
        data = np.squeeze(data[:, lead, :])

    if holdout:
        return data[:n_records//2], data[n_records//2:]
    return data

#### ---- Load Synthetic Dataset ---- ####
def load_synthetic_dataset(diagnostic, lead):
    data = np.load("synthetic_final.npy")
    label = np.load("synthetic_final_labels.npy")

    class_index = np.where(diagnostics == diagnostic)[0][0]
    mask = (label[:, class_index] == 1)

    return data[mask][:, lead, :]

#### ---- ECG Data Processing ---- ####
def get_first_n_beats(ecg_signal, n_beats):
    cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate, method="neurokit")
    _, info = nk.ecg_peaks(cleaned, sampling_rate=sampling_rate, method="elgendi2010")  
    peaks = info['ECG_R_Peaks']

    start = max(0, peaks[0] - 150)
    end = min(peaks[n_beats-1] + 300, len(cleaned))
    return cleaned[start:end]

def trim_ecg(data, n_beats):
    trimmed = []
    target_len = n_beats * sampling_rate # assume each beat has length of 1 second
    x_new = np.linspace(0, 1, target_len)
    for record in data:
        trimmed_record = get_first_n_beats(record, n_beats)
        
        x_old = np.linspace(0, 1, len(trimmed_record))
        f = interp1d(x_old, trimmed_record, kind='linear')
        new_record = np.array(f(x_new))
        new_record = 2 * (new_record - np.min(new_record)) / (np.max(new_record) - np.min(new_record)) - 1
        trimmed.append(f(x_new))

    return np.array(trimmed)


if __name__ == "__main__":
    diagnostic = ["NORM"]
    lead = 1
    synth_all = load_synthetic_dataset(diagnostic, lead)
    

    