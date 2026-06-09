import audiofile
import librosa
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class Wav2VOTDataset(Dataset):
    def __init__(self, csv_path, sampling_rate = 16000):
        self.csv_path = csv_path
        self.data = pd.read_csv(csv_path)
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        audio_name = row['obs_id']
        ## convert framewise labels back into array
        row_data = self.data[self.data['obs_id'] == audio_name]
        labels = self.convert_to_frames(row_data)

        audio_path = self.csv_path.parents[0].joinpath("wavs").joinpath(audio_name).with_suffix(".wav")
        audio, _ = librosa.load(audio_path, sr = self.sampling_rate)

        return {
            "input_values": torch.tensor(audio, dtype = torch.float32),
            "labels": torch.tensor(labels, dtype = torch.long)
        }

    def convert_to_frames(self, data_row, stride = 0.001, label_map = {'none': 0, 'closure': 1, 'vot': 2, 'lenition' : 3}):
        '''
        Converts the audio timestamps into framewise labels
        corresponding to stop annotations
        '''
        ## extract audio duration
        wavpath = self.csv_path.parents[0].joinpath("wavs").joinpath(data_row['obs_id'].values[0]).with_suffix(".wav")
        wav_duration = audiofile.duration(wavpath)

        ## create an n(frames) vector of 0s
        num_frames = int(np.ceil(wav_duration / stride))
        labels = np.full(num_frames, label_map['none'], dtype = np.int64)

        ## get index of first closure frame if present
        if "closure_start" in data_row.columns:
            closure_start_idx = int(data_row['closure_start'].iloc[0] / stride)

        ## get the VOT start/end indexes
        vot_start_idx = int(data_row['vot_start'].iloc[0] / stride)
        vot_end_idx = int(data_row['vot_end'].iloc[0] / stride)

        ## fill frames between closure start and VOT start
        if "closure_start" in data_row.columns:
            labels[closure_start_idx:vot_start_idx] = label_map['closure']

        ## fill VOT frames with VOT label
        labels[vot_start_idx:vot_end_idx] = label_map['vot']

        ## replace closure/VOT labels with lenition label if present
        if "lenition" in data_row.columns:
            if data_row['lenition'].iloc[0] == 1:
                for i, label in enumerate(labels):
                    if label != 0:
                        labels[i] = label_map['lenition']
        return labels

class Wav2VOTPredictionDataset(Dataset):
    def __init__(self, csv_path, sampling_rate = 16000):
        self.csv_path = csv_path
        self.data = pd.read_csv(csv_path, usecols = ['obs', 'left_window', 'start'])
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        audio_name = row['obs']
        left_window = row['left_window']
        obs_start = row['start']

        ## load the audio file
        audio_path = self.csv_path.parents[0].joinpath("wavs").joinpath(audio_name).with_suffix(".wav")
        audio, sr = librosa.load(audio_path, sr = self.sampling_rate)
        audio = librosa.resample(audio, orig_sr = sr, target_sr = 16000)

        return {
            "audio_name" : audio_name,
            "audio" : audio,
            "left_window" : left_window,
            "start" : obs_start,
            "input_values" : torch.tensor(audio, dtype = torch.float32),
            "input_lengths" : len(audio)
        }
