# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class DilatedLayer(nn.Module):
    def __init__(self, hidden, dilation):
        super(DilatedLayer, self).__init__()
        self.dil_conv = nn.Conv1d(hidden, hidden, kernel_size=3, padding=dilation, dilation=dilation)
        self.local_conv = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.direct_conv = nn.Conv1d(hidden, hidden, kernel_size=1)
        self.act = nn.SiLU(inplace=True)
        self.gate_conv = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.filter_conv = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.proj_conv = nn.Conv1d(hidden, hidden, kernel_size=1)

    def forward(self, x):
        y = self.dil_conv(x) + self.local_conv(x) + self.direct_conv(x)
        g = self.gate_conv(y)
        f = self.filter_conv(y)
        y = torch.sigmoid(g) * torch.tanh(f)
        y = self.proj_conv(y)
        return x + y, y


class DilatedBlock(nn.Module):
    def __init__(self, hidden_channel):
        super(DilatedBlock, self).__init__()
        self.layers = nn.ModuleList([DilatedLayer(hidden_channel, 3**i) for i in range(6)])

    def forward(self, x):
        y = 0
        for layer in self.layers:
            x, skip = layer(x)
            y = skip + y
        return y


class AudioSR(nn.Module):
    def __init__(self, rate):
        super(AudioSR, self).__init__()
        hidden_channel = 128 * 3 // rate

        self.name = 'AudioSR'
        self.rate = rate
        self.hidden_channel = hidden_channel
        self.pre_conv = nn.Conv1d(1, hidden_channel, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([DilatedBlock(hidden_channel) for _ in range(self.rate)])
        self.post_conv1 = nn.Conv1d(hidden_channel, hidden_channel, kernel_size=3, padding=1)
        self.post_conv2 = nn.Conv1d(hidden_channel, 1, kernel_size=3, padding=1)

    def forward(self, lr):
        B, L = lr.size()
        lr = lr.unsqueeze(1)

        x = self.pre_conv(lr)
        x = F.silu(x)

        y = torch.zeros(B, self.hidden_channel, self.rate * L).to(device)
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            y[:, :, i::self.rate] = x

        x = F.silu(y)
        x = self.post_conv1(x)
        x = F.silu(x)
        x = self.post_conv2(x)
        x = x.squeeze(1)
        return x
