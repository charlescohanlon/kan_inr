# Python script version of demo.ipynb notebook
# %%
import numpy as np
import torch

# Data parameters
data_root = "./data/"
data_filename = "nucleon_41x41x41_uint8.raw"
data_shape = (41, 41, 41)
data_dtype = np.uint8

batch_size = 1024
num_workers = 8
num_epochs = 24
lr = 0.01
lr_decay = 18
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
run_dtype = torch.float32  # or torch.float32 for half precision

# %%
from networks import INR_Base

network_type = "mlp"
use_native_encoder = not torch.cuda.is_available()
use_native_network = not torch.cuda.is_available() or network_type == "kan"

# Create model
model = INR_Base(
    n_neurons=32,
    n_hidden_layers=3,
    n_levels=8,
    n_features_per_level=8,
    per_level_scale=2.0,
    log2_hashmap_size=19,
    base_resolution=16,
    native_encoder=use_native_encoder,
    native_network=use_native_network,
    network_type=network_type,
)

import time
from tqdm import tqdm, trange

DEVICE = torch.device("cuda")

def Tensor(*args, **kwargs):
    return torch.tensor(*args, **kwargs, device=DEVICE)

def mse2psnr(x, data_range=1.0):
    x = x / data_range * data_range
    return (-10. * torch.log(x) / torch.log(Tensor(10.))) if torch.is_tensor(x) else (-10. * np.log(x) / np.log(10.))

def l1_loss(x, y):
    return torch.nn.L1Loss()(x, y) if torch.is_tensor(x) else np.absolute(x - y).mean()

def mse_loss(x, y):
    return torch.nn.MSELoss()(x, y) if torch.is_tensor(x) else np.mean((x - y) ** 2)

epoch_losses = []

def train(sampler, query, verbose=True, max_steps=50, lrate=1e-3, lrate_decay=500, batchsize=1024*64, **kwargs):
    global epoch_losses

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lrate, 
        betas=(0.9, 0.999), eps=1e-09, # weight_decay=1e-15, 
        # amsgrad=True, foreach=True #, fused=True,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lrate_decay, gamma=0.5)

    # Training
    step = 0
    def task():
        nonlocal step

        optimizer.zero_grad()

        # Reconstruction Loss
        with torch.no_grad():
            # NOTE: you dont need to do random samples really, I just took my old codes
            coords, targets = sampler.get_random_samples(batchsize)

        values = query(coords)
        loss = torch.nn.L1Loss()(values, targets)

        # Backpropagation
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Evaluation
        with torch.no_grad():
            psnr = mse2psnr(mse_loss(values, targets))

        step += 1
        return loss, psnr

    # Training Loop
    time0 = time.time()

    progress = trange(1, max_steps+1) if verbose else range(1, max_steps+1)
    for i in progress:
        loss, psnr = task()
        if verbose: progress.set_postfix_str(
            f'Loss: {loss:7.6f}, PSNR: {psnr:5.3f}dB, lrate: {scheduler.get_last_lr()[0]:5.3f}', refresh=True
        ) 
        epoch_losses.append(loss.item())

    total_time = time.time()-time0
    if verbose: print(f'[info] total training time: {total_time:5.3f}s, steps: {step}')
    return total_time

# %%
from pathlib import Path

from torch.utils.data import DataLoader
from volumetric_dataset import VolumetricDataset

print(f"Loading {data_filename}...")
data_filename = Path(data_root, data_filename)
# Create dataset and dataloader
dataset = VolumetricDataset(
    data_filename,
    data_shape,
    data_dtype,
    batch_size,
    normalize_coords=True,
    normalize_values=True,
    initial_shuffle=True,  # Shuffle dataset once initially
)
print(
    f"Dataset loaded with shape {dataset.data_shape} and type {np.dtype(dataset.data_type).name}"
)

# pin = device.type == "cuda"
# dataloader = DataLoader(
#     dataset, batch_size=None, num_workers=num_workers, pin_memory=pin
# )

# %%
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from tqdm import tqdm


# model.to(device, run_dtype)
# optimizer = AdamW(model.parameters(), lr=lr)
# scheduler = StepLR(optimizer, step_size=lr_decay, gamma=0.1)
# loss_fn = nn.functional.mse_loss

# # Training loop
# epoch_losses = []
# for epoch in range(num_epochs):
#     loss_total = 0.0
#     for x, y_hat in tqdm(dataloader):
#         x = x.to(device, run_dtype, non_blocking=True)
#         y_hat = y_hat.to(device, run_dtype, non_blocking=True)
#         y = model(x).to(dtype=run_dtype, non_blocking=True)
#         loss = loss_fn(y.squeeze(), y_hat)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         loss_total += loss.item()
# 
#     scheduler.step()
#     avg_loss = loss_total / len(dataloader)
#     epoch_losses.append(avg_loss)
#     print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")

# %%
gt_data = torch.as_tensor(
    dataset.volume_data(), device=device, dtype=run_dtype
).contiguous()

from samplers import *

sampler = VolumeSampler(data_shape, 'uint8')
sampler.load_from_ndarray(gt_data.flatten())

# ...
time = train(sampler, model, max_steps=10000)

with torch.no_grad():
    mse = sampler.compute_mse(model, verbose=True)
print(f"[info] mse: {mse}, psnr {mse2psnr(mse)}")


# %%
import matplotlib.pyplot as plt

# Plot training loss
plt.figure()
plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker="o")
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("loss.png")

# %%
from torchmetrics.functional import mean_squared_error
from torchmetrics.functional.image import peak_signal_noise_ratio
from torchmetrics.functional.image import structural_similarity_index_measure

# Direct reconstruction
with torch.no_grad():
    # NOTE: you dont need to do things the same way. I am substracting 0.5 because I needed to match CUDA texture sampling conventions.
    # So I am using a cell-centered voxel definition.
    xs = np.linspace(0.5/data_shape[0], (data_shape[0]-0.5)/data_shape[0], data_shape[0])
    ys = np.linspace(0.5/data_shape[1], (data_shape[1]-0.5)/data_shape[1], data_shape[0])
    zs = np.linspace(0.5/data_shape[2], (data_shape[2]-0.5)/data_shape[2], data_shape[0])
    coords = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).transpose(2, 1, 0, 3).reshape(-1, 3)
    coords = torch.tensor(coords, device="cuda")

    reconst_data = model(coords).reshape(data_shape).float()
    direct_psnr = peak_signal_noise_ratio(gt_data, reconst_data, data_range=1.0)
    print(f"Direct Reconstruction PSNR = {direct_psnr}dB")


# reconst_dataset = VolumetricDataset(
#     file_path=data_filename,
#     data_shape=data_shape,
#     data_type=data_dtype,
#     batch_size=batch_size,
#     normalize_coords=True,
#     normalize_values=True,
#     initial_shuffle=False,  # No shuffle for reconstruction
# )
# reconst_dataloader = DataLoader(
#     reconst_dataset,
#     batch_size=None,  # batching done in IterableDataset
#     num_workers=0,  # Single process for eval
#     pin_memory=pin,
# )
# Reconstruct the entire volume from the INR
model.eval()
# with torch.no_grad():
#     data_shape_tensor = torch.as_tensor(data_shape, device=device, dtype=run_dtype)
#     reconst_data = torch.zeros(data_shape, device=device, dtype=run_dtype)
#     for x, _ in tqdm(reconst_dataloader):
#         x = x.to(device, dtype=run_dtype, non_blocking=True)
#         y = model(x).to(dtype=run_dtype, non_blocking=True)
# 
#         indices = (x * (data_shape_tensor - 1)).long()
#         i, j, k = indices.split(1, dim=-1)
# 
#         # (batch_size,)
#         i, j, k = i.squeeze(), j.squeeze(), k.squeeze()
#         y = y.squeeze()
# 
#         reconst_data[i, j, k] = y


# Process metrics slice-by-slice (more stable for 3D volumes)
psnr_values = []
mse_values = []
ssim_values = []

# Iterate through one dimension
num_slices = gt_data.shape[2]
for i in tqdm(range(num_slices)):

    # (1, 1, H, W)
    gt_slice = gt_data[:, :, i].unsqueeze(0).unsqueeze(0)
    reconst_slice = reconst_data[:, :, i].unsqueeze(0).unsqueeze(0)

# Calculate metrics for this slice
psnr_values.append(peak_signal_noise_ratio(reconst_slice, gt_slice, data_range=1.0))
mse_values.append(mean_squared_error(reconst_slice, gt_slice))
ssim_values.append(
    structural_similarity_index_measure(reconst_slice, gt_slice, data_range=1.0)
)

# Average the metrics across all slices
psnr = torch.stack(psnr_values).mean().item()
mse = torch.stack(mse_values).mean().item()
ssim = torch.stack(ssim_values).mean().item()
print(f"PSNR: {psnr}, MSE: {mse}, SSIM: {ssim}")
