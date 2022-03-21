# -*- coding: utf-8 -*-
import os
import json
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from glob import glob

from utils import SNR, LSD
from torch.utils.tensorboard import SummaryWriter

from dataset import VCTK092Dataset
from model.audiosr import AudioSR

device = 'cuda' if torch.cuda.is_available() else 'cpu'

config_files = glob('./config/*.json')
print("------- Configuration --------")
for i in range(len(config_files)):
    print(f'{i}. {os.path.basename(config_files[i])}')
config_file = config_files[int(input("Choose Configuration: "))]

with open(config_file, 'r') as f:
    config = json.load(f)

LR_sample_rate = config['LR_sample_rate']
HR_sample_rate = config['HR_sample_rate']
rate = HR_sample_rate // LR_sample_rate

max_epoch = config['max_epoch']
batch_size = config['batch_size']

dtype = config['dtype']
train_dataset = VCTK092Dataset(dtype=dtype, mode='train', lr=LR_sample_rate, hr=HR_sample_rate)
val_dataset = VCTK092Dataset(dtype=dtype, mode='val', lr=LR_sample_rate, hr=HR_sample_rate)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
val_loader = DataLoader(val_dataset, shuffle=False, batch_size=1)

model = AudioSR(rate).to(device)
optimizer = optim.Adam(model.parameters(), lr=config['lr'])

root = Path('./')
ckpt_root = root / 'model' / dtype / f'x{rate}'
ckpt_root.mkdir(parents=True, exist_ok=True)
last_path = ckpt_root / 'last.pth'
best_path = ckpt_root / 'best.pth'
log_dir = ckpt_root / 'log'
log_dir.mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(log_dir)
best_val_loss = np.inf
last_epoch = 0

if os.path.exists(best_path):
    ckpt = torch.load(best_path)
    best_val_loss = ckpt['val_loss']

if os.path.exists(last_path):
    ckpt = torch.load(last_path, map_location=torch.device(device))
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    last_epoch = ckpt['epoch']
    print('Last checkpoint is loaded.')
    print(f'Last Epoch: {ckpt["epoch"]} |',
          f'Last Avg Train Loss: {ckpt["train_loss"]} |',
          f'Last Avg Val Loss: {ckpt["val_loss"]}')
else:
    print('No checkpoint is found.')

for epoch in range(last_epoch+1, max_epoch+1):
    start_epoch = time.time()
    print(f'-------- EPOCH {epoch} / {max_epoch} --------')

    model.train()
    train_iter, train_loss = 0, 0
    for lr, hr in train_loader:
        lr = lr.to(device)
        hr = hr.to(device)
        sr = model(lr)

        loss = LSD(sr, hr) - SNR(sr, hr)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_iter += 1

    model.eval()
    val_iter, val_loss = 0, 0
    with torch.no_grad():
        for lr, hr in val_loader:
            lr = lr.to(device)
            hr = hr.to(device)
            sr = model(lr)[:, :len(hr[0])]

            loss = LSD(sr, hr) - SNR(sr, hr)

            val_loss += loss.item()
            val_iter += 1

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    print(f'EPOCH {epoch}/{max_epoch} |',
          f'Avg train loss: {avg_train_loss} |',
          f'Avg val loss: {avg_val_loss}')
    print(f'This epoch took {time.time() - start_epoch} seconds')

    writer.add_scalar('train_loss', avg_train_loss, epoch)
    writer.add_scalar('val_loss', avg_val_loss, epoch)
    writer.flush()

    ckpt = {'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss}
    torch.save(ckpt, last_path)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(ckpt, best_path)
        print('New Best Model!')
