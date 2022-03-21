# -*- coding: utf-8 -*-
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def SNR(pred, target):
    return (20 * torch.log10(torch.norm(target, dim=-1) / torch.norm(pred - target, dim=-1).clamp(min=1e-8))).mean()


def LSD(pred, target, nfft=2048, hop=512):
    window = torch.hann_window(nfft).to(device)
    stft_p = torch.stft(pred, nfft, hop, window=window, return_complex=False)
    stft_t = torch.stft(target, nfft, hop, window=window, return_complex=False)

    mag_p = torch.norm(stft_p, p=2, dim=-1)
    mag_t = torch.norm(stft_t, p=2, dim=-1)

    sp = torch.log10(mag_p.square().clamp(1e-8))
    st = torch.log10(mag_t.square().clamp(1e-8))
    return (sp - st).square().mean(dim=1).sqrt().mean()
