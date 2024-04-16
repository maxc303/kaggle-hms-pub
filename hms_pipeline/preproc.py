import numpy as np
import pandas as pd
import os
from .project_configs import *
import cv2
import yaml
import joblib
import librosa
from tqdm import tqdm
from scipy.signal import butter, lfilter, sosfiltfilt


############################################
# Signal Processing
############################################
def butter_lowpass_filter(data, cutoff_freq=20, sampling_rate=200, order=4):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    filtered_data = lfilter(b, a, data, axis=0)
    return filtered_data

def butter_bandpass_filter(data, lowcut = 0.4, highcut = 20, fs = 200, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    y = lfilter(b, a, data)
    return y

def sos_bandpass_filter(data, lowcut = 0.4, highcut = 20, fs = 200, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    sos = butter(order, [low, high], btype="band", analog=False, output="sos")
    filtered_data = sosfiltfilt(sos, data)
    return filtered_data

def band_remove(x, thres_amp= 300):
    # find the max amplitude of the signal
    max_amp = np.max(np.abs(x))

    min_left_idx = len(x)
    max_right_idx = 0
    while max_amp > thres_amp:

        # find the index of the max amplitude
        max_amp_idx = np.argmax(np.abs(x))
        # find the left and right cloest index that the amplitude is 0
        left_idx = max_amp_idx
        right_idx = max_amp_idx
        max_amp_sign = np.sign(x[max_amp_idx])
        # print(x[max_amp_idx])

        while left_idx > 0 and x[left_idx] * max_amp_sign > 0:
            left_idx -= 1

        while right_idx < len(x) and x[right_idx] * max_amp_sign > 0:
            right_idx += 1
        x[left_idx:right_idx] = 0
        max_amp = np.max(np.abs(x))
        min_left_idx = min(min_left_idx, left_idx)
        max_right_idx = max(max_right_idx, right_idx)
    return x, min_left_idx, max_right_idx


############################################
# Preprocessing Raw EEG
############################################
def preprocess_raw_eeg_list(eeg_dir, eeg_ids, output_dir, filter_type = "bandpass", sub_width = 500, num_workers = 8, norm_type = "global", highcut = 20):
    task_list = []
    os.makedirs(output_dir, exist_ok=True)

    assert sub_width % 50 == 0, "sub_width must be a factor of 50"
    freq = sub_width//50
    assert 200 % freq == 0
    sub_width = 50 * freq
    metadata = {"filter_type": filter_type, "freq": freq, "sub_width": sub_width, "norm_type": norm_type, "highcut": highcut}
    print(metadata)
    # save metadata as a yaml file
    metadata_path = os.path.join(output_dir, "metadata.yaml")
    with open(metadata_path, "w") as f:
        yaml.dump(metadata, f)

    for eeg_id in eeg_ids:
        eeg_path = os.path.join(eeg_dir, f"{eeg_id}.parquet")
        output_path = os.path.join(output_dir, f"{eeg_id}.npy")
        task_list.append((eeg_path, output_path, filter_type, freq, norm_type, highcut))

    _ = joblib.Parallel(n_jobs=num_workers)(joblib.delayed(preprocess_raw_eeg)(*task) for task in tqdm(task_list))



def preprocess_raw_eeg(eeg_path, output_path, filter_type = "bandpass", freq = 200, norm_type = "global", highcut = 20):
    # Frequency must be a factor of 200
    assert 200 % freq == 0
    eeg_signal_dict = load_raw_eeg(eeg_path, filter=filter_type, norm_type=norm_type, highcut=highcut)
    eeg_signal_dict = resize_eeg_dict(eeg_signal_dict, freq)
    np.save(output_path, eeg_signal_dict)

def resize_eeg_dict(eeg_dict, freq):
    # Frequency must be a factor of 200
    assert 200 % freq == 0
    new_eeg_dict = dict()
    for signal_name in SIGNAL_NAME:
        signal_df = eeg_dict[signal_name]
        new_signal_df = pd.DataFrame()
        data = signal_df.values
        # resize the signal to the new frequency
        new_data = cv2.resize(data, (data.shape[1], int(data.shape[0] * freq / 200)), interpolation=cv2.INTER_AREA)
        new_signal_df = pd.DataFrame(new_data, columns=signal_df.columns)
        # print(new_signal_df.shape)
        new_eeg_dict[signal_name] = new_signal_df
    return new_eeg_dict

def load_raw_eeg(eeg_path, filter=None, norm_type = "global", lowcut = 0.5, highcut = 20):
    eeg_df = pd.read_parquet(eeg_path)
    eeg_signal_dict = dict()

    all_eeg_signals = []
    filter_amp = 300
    for k in range(len(SIGNAL_NAME)):
        cols = EEG_FEATS[k]
        signal_name = SIGNAL_NAME[k]
        signal_df = pd.DataFrame()
        for kk in range(len(cols) - 1):
            sub_signal_name = f"{signal_name}_{cols[kk]}-{cols[kk+1]}"
            x = eeg_df[cols[kk]].values - eeg_df[cols[kk + 1]].values
            m = np.nanmean(x)
            if np.isnan(x).mean() < 1:
                x = np.nan_to_num(x, nan=m)

            if filter == "bandpass":
                x = butter_bandpass_filter(x, lowcut = lowcut , highcut = highcut, fs = 200, order=4)
            elif filter == "bandclip":
                x = butter_bandpass_filter(x, lowcut = lowcut , highcut = highcut, fs = 200, order=4)
                x = np.clip(x, -filter_amp, filter_amp)
            elif filter == "bandblock":
                x = butter_bandpass_filter(x, lowcut = lowcut , highcut = highcut, fs = 200, order=4)
                # x > 200 or x < -200 = 0
                x[x > filter_amp] = 0
                x[x < -filter_amp] = 0
            elif filter == "bandremove":
                x = butter_bandpass_filter(x, lowcut = lowcut , highcut = highcut, fs = 200, order=4)
                x, min_left_idx, max_right_idx = band_remove(x, thres_amp=filter_amp)
            elif filter == "sosbandpass":
                x = sos_bandpass_filter(x, lowcut = lowcut , highcut = highcut, fs = 200, order=6)
            elif filter == "sosbandclip":
                x = sos_bandpass_filter(x, lowcut = lowcut , highcut = highcut, fs = 200, order=6)
                x = np.clip(x, -filter_amp, filter_amp)
            elif filter == "sosbandremove":
                x = sos_bandpass_filter(x, lowcut = lowcut , highcut = highcut, fs = 200, order=6)
                x, _ , _ = band_remove(x, thres_amp=filter_amp)
            else:
                x = butter_lowpass_filter(x, cutoff_freq=20, sampling_rate=200, order=4)

            signal_df[sub_signal_name] = x
            all_eeg_signals.append(x)
        eeg_signal_dict[signal_name] = signal_df

    eps = 1e-6
    if norm_type == "constant":
        mean = 0.0
        std = 100.0
    else:
        all_eeg_signals = np.concatenate(all_eeg_signals, axis=0)
        mean = np.mean(all_eeg_signals)
        std = np.std(all_eeg_signals)
        # print(f"mean: {mean}, std: {std}")

    for signal_name in SIGNAL_NAME:
        eeg_signal_dict[signal_name] = (
            eeg_signal_dict[signal_name] - mean
        ) / (std + eps)
    return eeg_signal_dict

############################################
# Preprocessing Long Spectrogram
############################################

def preprocess_long_spec_list(long_spec_dir, long_spec_ids, output_dir, signal_height = 100, num_workers = 8):
    task_list = []
    os.makedirs(output_dir, exist_ok=True)
    metadata = {"signal_height": signal_height, "freq": 0.5, "sub_width": 300}
    print(metadata)
    # save metadata as a yaml file
    metadata_path = os.path.join(output_dir, "metadata.yaml")
    with open(metadata_path, "w") as f:
        yaml.dump(metadata, f)

    for long_spec_id in long_spec_ids:
        long_spec_path = os.path.join(long_spec_dir, f"{long_spec_id}.parquet")
        output_path = os.path.join(output_dir, f"{long_spec_id}.npy")
        task_list.append((long_spec_path, output_path, signal_height))

    _ = joblib.Parallel(n_jobs=num_workers)(joblib.delayed(preprocess_long_spec)(*task) for task in tqdm(task_list))

def resize_long_spec_dict(long_spec_dict, signal_height = 100):
    new_long_spec_dict = dict()
    for signal_name in ["LL", "RL", "LP", "RP"]:
        signal_img = long_spec_dict[signal_name]
        new_signal_img = cv2.resize(signal_img, (signal_img.shape[1], signal_height))
        new_long_spec_dict[signal_name] = new_signal_img
    return new_long_spec_dict

def preprocess_long_spec(long_spec_path, output_path, signal_height = 100):
    # default is 2second per row
    long_spec_dict = load_long_spec(long_spec_path)
    long_spec_dict = resize_long_spec_dict(long_spec_dict, signal_height)
    np.save(output_path, long_spec_dict)

def load_long_spec(long_spec_path):
    long_spec_df = pd.read_parquet(long_spec_path)
    long_spec_dict = dict()

    # note: the order is different from public notebook
    signal_names = ["LL", "RL", "LP", "RP"]
    # only keep the last 400 columns
    signal_columns = {}
    for signal in signal_names:
        signal_columns[signal] = [
            col for col in long_spec_df.columns if signal in col
        ]

    all_specs = []
    for k in range(len(signal_names)):
        columns = signal_columns[signal_names[k]]
        signal_img = long_spec_df[columns].values.T
        # flip vertically
        signal_img = np.flipud(signal_img)
        ## TODO: Check this value
        signal_img = np.nan_to_num(signal_img, nan=0.0)
        signal_img = np.clip(signal_img, 10e-3, 10e3) 
        signal_img = 10 * np.log10(signal_img) # Convert to dB (-20, 40)

        # Normalize to [-1, 1]
        signal_img = (signal_img - 10.0) / 30.0
        all_specs.append(signal_img)
        long_spec_dict[signal_names[k]] = signal_img

    all_specs = np.concatenate(all_specs, axis=0)
    # eps = 1e-6
    # mean = np.mean(all_specs)
    # std = np.std(all_specs)
    # for k in range(len(signal_names)):
    #     long_spec_dict[signal_names[k]] = (long_spec_dict[signal_names[k]] - mean) / (
    #         std + eps
    #     )
    return long_spec_dict

############################################
# Preprocessing EEG Spectrogram
############################################

def preprocess_eeg_spec_list(eeg_input_dir, eeg_ids, output_dir, signal_height = 100, sub_width = 200, gen_freq = 100, n_fft = 1024, win_length = 1024, lowcut = 0, prefilter = False, num_workers = 8):
    """
    sub_width: the width of the 50s mel spectrogram. Max width is 10000 as the original signal
    gen_freq: the frequency of the generated mel spectrogram
    """
    os.makedirs(output_dir, exist_ok=True)
    task_list = []

    # assert sub_width % 50 == 0, "sub_width must be divisible 50"
    freq = sub_width//50
    # assert 200 % freq == 0, "Frequency must be a factor of 200"
    # hop = 200//freq
    assert 200%gen_freq ==0 
    hop = 200//gen_freq

    gen_width = 10000//hop
    ## Native frequency of the mel spec is 10. Will be resized to target
    sub_width = 50 * freq
    metadata = {"signal_height": signal_height, "hop": hop, "freq": freq, "gen_freq":gen_freq, "sub_width": sub_width, "gen_width":gen_width, "lowcut": lowcut, "prefilter": prefilter, "n_fft": n_fft, "win_length": win_length}
    print(metadata)

    # save metadata as a yaml file
    metadata_path = os.path.join(output_dir, "metadata.yaml")
    with open(metadata_path, "w") as f:
        yaml.dump(metadata, f)

    for eeg_id in eeg_ids:
        eeg_path = os.path.join(eeg_input_dir, f"{eeg_id}.parquet")
        output_path = os.path.join(output_dir, f"{eeg_id}.npy")
        task_list.append((eeg_path, output_path, lowcut, signal_height, freq, gen_freq, prefilter, n_fft, win_length))
    assert len(task_list) > 0, "No EEG files found in the input directory"

    result = joblib.Parallel(n_jobs=num_workers)(joblib.delayed(preprocess_eeg_spec)(*task) for task in tqdm(task_list))



def resize_eeg_spec_dict(eeg_spec_dict, freq = 5, gen_freq = 10):
    new_eeg_spec_dict = dict()
    assert gen_freq % freq == 0
    assert gen_freq >= freq
    ratio = freq/gen_freq

    for signal_name in SIGNAL_NAME:
        signal_img = eeg_spec_dict[signal_name]
        new_width = int(signal_img.shape[1]*ratio)
        if new_width == signal_img.shape[1]:
            new_eeg_spec_dict[signal_name] = signal_img
            continue
        new_signal_img = cv2.resize(signal_img, (new_width, signal_img.shape[0]), interpolation=cv2.INTER_AREA)
        new_eeg_spec_dict[signal_name] = new_signal_img
    return new_eeg_spec_dict

def preprocess_eeg_spec(eeg_path, eeg_spec_path, lowcut = 0, signal_height = 128, freq = 5, gen_freq = 10,  prefilter = False, n_fft = 1024, win_length = 1024):
    eeg_spec_data = load_eeg_spec(eeg_path, signal_height, gen_freq, lowcut=lowcut, prefilter=prefilter, n_fft=n_fft, win_length=win_length)
    eeg_spec_data = resize_eeg_spec_dict(eeg_spec_data, freq = freq, gen_freq=gen_freq)
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

def load_eeg_spec(eeg_path, height = 128, gen_freq = 10, lowcut = 0, prefilter = False, mel_scale = True, n_fft = 2048, win_length = 1024):
    # the resulting specs will have width = len(signal_df)//20.
    # All given specs will have length = 10000 + 200*n
    # 200 mod hop_length must be 0
    hop_length = 200//gen_freq
    assert 200%hop_length == 0
    eeg_df = pd.read_parquet(eeg_path)
    width = len(eeg_df)//hop_length
    eeg_spec_data = dict()
    
    all_specs = []
    for k in range(len(SIGNAL_NAME)):
        COLS = EEG_FEATS[k]

        signal_name = SIGNAL_NAME[k]
        signal_eeg_spec = np.zeros((height,width),dtype='float32')
        for kk in range(len(COLS)-1):
            x = eeg_df[COLS[kk]].values - eeg_df[COLS[kk+1]].values 
            # FILL NANS
            m = np.nanmean(x)
            if np.isnan(x).mean()<1: x = np.nan_to_num(x,nan=m)
            else: x[:] = 0
            
            if prefilter:
                x = sos_bandpass_filter(x, lowcut = lowcut, highcut = 20, fs = 200, order=6)

            mel_spec = librosa.feature.melspectrogram(y=x, sr=200, hop_length=hop_length, 
                n_fft=n_fft, n_mels=height, fmin=lowcut, fmax=20, win_length=win_length)

            correct_width = mel_spec.shape[1]-1
            assert correct_width == width, f"Width mismatch: {correct_width} != {width}"

            # mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=80).astype(np.float32)[:,:correct_width]
            mel_spec_db = 10 * np.log10(mel_spec+1e-6)
            mel_spec_db = mel_spec_db[:,:correct_width]
            # print(mel_spec_db.min(), mel_spec_db.max())
            # mel_spec_db = (mel_spec_db+40)/40 
            # mel_spec_db = (mel_spec_db+30)/30

            signal_eeg_spec += mel_spec_db
        signal_eeg_spec /= float(len(COLS)-1)
        # signal_eeg_spec = signal_eeg_spec/100
        signal_eeg_spec = (signal_eeg_spec-50)/50
        eeg_spec_data[signal_name] = signal_eeg_spec
        all_specs.append(signal_eeg_spec)

    # all_eeg_spec = np.concatenate(all_specs, axis=0)
    # eps = 1e-6
    # mean = np.mean(all_eeg_spec)
    # std = np.std(all_eeg_spec)

    for signal_name in SIGNAL_NAME:
        # eeg_spec_data[signal_name] =  (eeg_spec_data[signal_name] - mean)/(std+eps)
        #flip vertically
        eeg_spec_data[signal_name] = np.flipud(eeg_spec_data[signal_name])

    return eeg_spec_data