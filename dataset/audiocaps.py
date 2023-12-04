import pandas as pd
import os
import random
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
import torchaudio


class AudioCaps(Dataset):
    def __init__(self, data_dir, meta_dir,
                 seg_length=8, sr=24000):
        self.datadir = data_dir
        meta = pd.read_csv(meta_dir)
        self.meta = meta[meta['audio_length'] != 0]
        self.seg_len = seg_length
        self.sr = sr

    def __getitem__(self, index):
        row = self.meta.iloc[index]
        # load current audio
        audio_path = self.datadir + str(row['audiocap_id']) + '.wav'
        y, sr = torchaudio.load(audio_path)
        assert sr == self.sr

        total_length = y.shape[-1]
        if int(total_length - self.sr * self.seg_len) > 0:
            start = np.random.randint(0, int(total_length - self.sr * self.seg_len) + 1)
        else:
            start = 0
        end = min(start + self.seg_len * self.sr, total_length)
        audio_clip = torch.zeros(self.seg_len * self.sr)

        audio_clip[:end - start] = y[0, start: end]

        text = row['caption']

        return audio_clip, text

    def __len__(self):
        return len(self.meta)


if __name__ == '__main__':
    dataset = AudioCaps('../data/audiocaps_24k/test/', '../data/test.csv')
    train_loader = DataLoader(dataset, num_workers=0, batch_size=32)
    for batch in train_loader:
        break