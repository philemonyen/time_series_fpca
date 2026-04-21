import ast
import wfdb
import pandas as pd
import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from pathlib import Path

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
    data = data.transpose(0, 2, 1) # (n_records, n_leads, n_samples)
    # Perform min-max scaling for each lead
    # min_per_lead = data.min(axis=(0, 2), keepdims=True)  # shape: (1, n_leads, 1)
    # max_per_lead = data.max(axis=(0, 2), keepdims=True)  # shape: (1, n_leads, 1)
    # denom = (max_per_lead - min_per_lead)
    # denom[denom == 0] = 1  # avoid division by zero
    # data = (data - min_per_lead) / denom
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

    filtered = data[mask][:, lead, :]
    # min_per_lead = filtered.min(axis=0, keepdims=True) 
    # max_per_lead = filtered.max(axis=0, keepdims=True)
    # denom = (max_per_lead - min_per_lead)
    # denom[denom == 0] = 1  # avoid division by zero
    # filtered = (filtered - min_per_lead) / denom
    # Perform z-normalization (mean=0, std=1) for each lead
    # mean_per_lead = filtered.mean(axis=0, keepdims=True)
    # std_per_lead = filtered.std(axis=0, keepdims=True)
    # std_per_lead[std_per_lead == 0] = 1  # avoid division by zero
    # filtered = (filtered - mean_per_lead) / std_per_lead

    return filtered

#### ---- ECG Data Processing ---- ####
def get_first_n_beats(ecg_signal, n_beats):
    cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate, method="neurokit")
    _, info = nk.ecg_peaks(cleaned, sampling_rate=sampling_rate, method="elgendi2010")  
    peaks = info['ECG_R_Peaks']
    if len(peaks) < n_beats:
        return None

    start = max(0, peaks[0] - 150)
    end = min(peaks[n_beats-1] + 300, len(cleaned))
    return cleaned[start:end]

def trim_ecg(data, n_beats):
    trimmed = []
    target_len = n_beats * sampling_rate # assume each beat has length of 1 second
    x_new = np.linspace(0, 1, target_len)
    for record in data:
        trimmed_record = get_first_n_beats(record, n_beats)
        if trimmed_record is None:
            continue
        x_old = np.linspace(0, 1, len(trimmed_record))
        f = interp1d(x_old, trimmed_record, kind='linear')
        new_record = np.array(f(x_new))
        trimmed.append(new_record)

    return np.array(trimmed)

def plot(name, directory, fd, fd_smooth, fd_aligned, mean, components, n_beats):
    save_path = f"images/{directory}"
    path=Path(save_path)
    path.mkdir(parents=True, exist_ok=True)

    fd.plot()
    plt.title(f"{name}: Raw ({n_beats} beats)")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (mV)")
    plt.savefig(save_path + '/raw.png')
    plt.close()
    fd_smooth.plot()
    plt.title(f"{name}: Smoothed ({n_beats} beats)")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (mV)")
    plt.savefig(save_path + "/smoothed.png")
    plt.close()
    fd_aligned.plot()
    plt.title(f"{name}: Aligned ({n_beats} beats)")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (mV)")
    plt.savefig(save_path + "/aligned.png")
    plt.close()
    mean.plot()
    plt.title(f"{name}: FPCA Mean Curve ({n_beats} beats)")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (mV)")
    plt.savefig(save_path + "/mean.png")
    plt.close()
    component_matrix = components.data_matrix
    n_components = component_matrix.shape[0]
    fig, axes = plt.subplots(n_components, 1, figsize=(8, 12))
    xvals = np.linspace(0, n_beats, n_beats*get_sr())
    for i in range(n_components):
        axes[i].plot(xvals, component_matrix[i])
        axes[i].set_title(f"{name}: Eigenfunction {i+1} ({n_beats} beats)")
        axes[i].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(save_path + "/components.png")
    plt.close()

if __name__ == "__main__":
    diagnostic = ["NORM"]
    lead = 1
    synth_all = load_synthetic_dataset(diagnostic, lead)

    

    