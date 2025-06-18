import torch
from torch.utils.data import Dataset
import os
import random
import numpy as np
from glob import glob
from wav2mel import wav_to_mel

class TripletAudioDataset(Dataset):
    def __init__(self, standard_dir, non_standard_dir, max_frames=300):
        self.standard_files = glob(os.path.join(standard_dir, "**", "*.wav"), recursive=True)
        self.non_standard_files = glob(os.path.join(non_standard_dir, "**", "*.wav"), recursive=True)
        self.max_frames = max_frames

    def __len__(self):
        return len(self.standard_files)

    def __getitem__(self, idx):
        anchor = self.standard_files[idx]
        positive = random.choice(self.standard_files)
        while positive == anchor:
            positive = random.choice(self.standard_files)
        negative = random.choice(self.non_standard_files)

        a = self._process(wav_to_mel(anchor))
        p = self._process(wav_to_mel(positive))
        n = self._process(wav_to_mel(negative))

        return a, p, n

    def _process(self, mel):
        # mel.shape: [n_mels, time]
        n_mels, time = mel.shape
        if time > self.max_frames:
            mel = mel[:, :self.max_frames]
        elif time < self.max_frames:
            pad_width = self.max_frames - time
            mel = np.pad(mel, ((0, 0), (0, pad_width)), mode='constant')
        return torch.from_numpy(mel).float()  # shape: [n_mels, max_frames]
