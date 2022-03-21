# -*- coding: utf-8 -*-
import random
import torch
import torch.nn.functional as F
import librosa
from glob import glob
from torch.utils.data import Dataset


class VCTK092Dataset(Dataset):
    def __init__(self, dtype, mode, lr=16000, hr=48000):
        super(VCTK092Dataset, self).__init__()
        self.hr, self.lr = hr, lr
        self.rate = hr // lr
        self.dtype = dtype
        self.mode = mode
        self.file_names = []

        assert dtype in ['single', 'multi']
        assert mode in ['train', 'val', 'test']

        if dtype == 'single':
            if mode == 'train':
                self.file_names += glob('dataset/VCTK-Corpus-0.92/wav48_silence_trimmed/p225/*mic1.flac')[:-16]
            elif mode == 'val':
                self.file_names += glob('dataset/VCTK-Corpus-0.92/wav48_silence_trimmed/p225/*mic1.flac')[-16:-8]
            elif mode == 'test':
                self.file_names += glob('dataset/VCTK-Corpus-0.92/wav48_silence_trimmed/p225/*mic1.flac')[-8:]

        elif dtype == 'multi':
            dir_names = glob('dataset/VCTK-Corpus-0.92/wav48_silence_trimmed/*/')
            dir_names = list(filter(lambda x: 'p280' not in x and 'p315' not in x, dir_names))
            if mode == 'train':
                for directory in dir_names[:-16]:
                    self.file_names += glob(f'{directory}*mic1.flac')
            elif mode == 'val':
                for directory in dir_names[-16:-8]:
                    self.file_names += glob(f'{directory}*mic1.flac')
            elif mode == 'test':
                for directory in dir_names[-8:]:
                    self.file_names += glob(f'{directory}*mic1.flac')

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        raw_khz_hr = librosa.load(file_name, sr=self.hr)[0]
        raw_khz_lr = librosa.resample(y=raw_khz_hr, orig_sr=self.hr, target_sr=self.lr, res_type='linear')

        if self.mode == 'train':
            max_length = 15000 // self.rate
            lr_length = len(raw_khz_lr)
            start = random.randrange(max(1, lr_length - max_length))

            khz_lr = torch.Tensor(raw_khz_lr[start:start + max_length])
            khz_lr = F.pad(khz_lr, (0, max(0, max_length - len(khz_lr))), 'constant', 0)
            khz_hr = torch.Tensor(raw_khz_hr[self.rate * start:self.rate * (start + max_length)])
            khz_hr = F.pad(khz_hr, (0, max(0, self.rate * max_length - len(khz_hr))), 'constant', 0)

        else:
            khz_lr = torch.Tensor(raw_khz_lr)
            khz_hr = torch.Tensor(raw_khz_hr)

        return khz_lr, khz_hr

    def __len__(self):
        return len(self.file_names)
