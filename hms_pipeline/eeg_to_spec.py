import librosa
import numpy as np
import pandas as pd
import os
import joblib

from tqdm import tqdm
from scipy.signal import butter, lfilter

from .project_configs import  SIGNAL_NAME, EEG_FEATS

def create_dataset_eeg_spec_imgs(eeg_ids, eeg_input_dir,  output_dir, num_workers=4, lowcut = 0):
    # group by eeg_id

    os.makedirs(output_dir, exist_ok=True)
    task_list = []
    for eeg_id in eeg_ids:
        eeg_path = os.path.join(eeg_input_dir, f"{eeg_id}.parquet")
        output_path = os.path.join(output_dir, f"{eeg_id}.npy")
        task_list.append((eeg_path, output_path, lowcut))
    assert len(task_list) > 0, "No EEG files found in the input directory"

    result = joblib.Parallel(n_jobs=num_workers)(joblib.delayed(save_eeg_spec_imgs)(*task) for task in tqdm(task_list))

def save_eeg_spec_imgs(eeg_path, eeg_spec_path, lowcut = 0, height = 128, hop_length=20):
    eeg_df = pd.read_parquet(eeg_path)
    signal_df = eeg_to_signal_df(eeg_df)
    eeg_spec_data = signal_df_to_specs(signal_df, height, hop_length, lowcut=lowcut)
    output_path = os.path.join(eeg_spec_path)
    np.save(output_path, eeg_spec_data)

def eeg_to_signal_df(eeg_df):
    signal_df = pd.DataFrame()
    for k in range(len(SIGNAL_NAME)):
        COLS = EEG_FEATS[k]
        for kk in range(len(COLS)-1):
            signal_col_name = f"{SIGNAL_NAME[k]}_{COLS[kk]}-{COLS[kk+1]}"
            # COMPUTE PAIR DIFFERENCES
            x = eeg_df[COLS[kk]].values - eeg_df[COLS[kk+1]].values 
            # FILL NANS
            m = np.nanmean(x)
            if np.isnan(x).mean()<1: x = np.nan_to_num(x,nan=m)
            else: x[:] = 0

            signal_df[signal_col_name] = x
    return signal_df

def butter_lowpass_filter(data, cutoff_freq=20, sampling_rate=200, order=4):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    filtered_data = lfilter(b, a, data, axis=0)
    return filtered_data


def butter_bandpass_filter(data, lowcut = 0.5, highcut = 20, fs = 200, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    y = lfilter(b, a, data)
    return y

def signal_df_to_specs(signal_df, height = 128, hop_length = 20, lowcut = 0):
    # the resulting specs will have width = len(signal_df)//20.
    # All given specs will have length = 10000 + 200*n
    # 200 mod hop_length must be 0
    assert 200%hop_length == 0

    width = len(signal_df)//20
    eeg_spec_data = dict()
    
    all_specs = []
    for k in range(len(SIGNAL_NAME)):
        COLS = EEG_FEATS[k]

        signal_name = SIGNAL_NAME[k]
        signal_eeg_spec = np.zeros((height,width),dtype='float32')
        for kk in range(len(COLS)-1):
            spec_col_name = f"{signal_name}_{COLS[kk]}-{COLS[kk+1]}"
            x = signal_df[spec_col_name].values
            
            # x = butter_lowpass_filter(x, cutoff_freq=20, sampling_rate=200, order=4)

            # bandpass of spec causing issues
            # x = butter_bandpass_filter(x, lowcut = 0.5, highcut = 20, fs = 200, order=4)
            
            mel_spec = librosa.feature.melspectrogram(y=x, sr=200, hop_length=hop_length, 
                  n_fft=1024, n_mels=height, fmin=lowcut, fmax=20, win_length=height)
            correct_width = (mel_spec.shape[1]//hop_length)*hop_length

            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)[:,:correct_width]
            # mel_spec_db = (mel_spec_db+40)/40 
            signal_eeg_spec += mel_spec_db
        signal_eeg_spec /= float(len(COLS)-1)
        eeg_spec_data[signal_name] = signal_eeg_spec
        all_specs.append(signal_eeg_spec)

    all_eeg_spec = np.concatenate(all_specs, axis=0)
    eps = 1e-6
    mean = np.mean(all_eeg_spec)
    std = np.std(all_eeg_spec)

    for signal_name in SIGNAL_NAME:
        eeg_spec_data[signal_name] =  (eeg_spec_data[signal_name] - mean)/(std+eps)

    return eeg_spec_data




