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

network_type = "kan"
use_native_encoder = not torch.cuda.is_available()
use_native_network = not torch.cuda.is_available() or network_type == "kan"

# Create model
model = INR_Base(
    n_neurons=16,
    n_hidden_layers=1,
    n_levels=1,
    n_features_per_level=8,
    per_level_scale=1.1,
    log2_hashmap_size=19,
    base_resolution=int((1 << 19) ** (1 / 3)),
    native_encoder=use_native_encoder,
    native_network=use_native_network,
    network_type=network_type,
)

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

pin = device.type == "cuda"
dataloader = DataLoader(
    dataset, batch_size=None, num_workers=num_workers, pin_memory=pin
)

# %%
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from tqdm import tqdm


model.to(device, run_dtype)
optimizer = AdamW(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=lr_decay, gamma=0.1)
loss_fn = nn.functional.mse_loss

# Training loop
epoch_losses = []
for epoch in range(num_epochs):
    loss_total = 0.0
    for x, y_hat in tqdm(dataloader):
        x = x.to(device, run_dtype, non_blocking=True)
        y_hat = y_hat.to(device, run_dtype, non_blocking=True)
        y = model(x).to(dtype=run_dtype, non_blocking=True)
        loss = loss_fn(y.squeeze(), y_hat)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_total += loss.item()

    scheduler.step()
    avg_loss = loss_total / len(dataloader)
    epoch_losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")

# %%
import matplotlib.pyplot as plt

# Plot training loss
plt.figure()
plt.plot(range(1, num_epochs + 1), epoch_losses, marker="o")
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# %%
from torchmetrics.functional import mean_squared_error
from torchmetrics.functional.image import peak_signal_noise_ratio
from torchmetrics.functional.image import structural_similarity_index_measure

reconst_dataset = VolumetricDataset(
    file_path=data_filename,
    data_shape=data_shape,
    data_type=data_dtype,
    batch_size=batch_size,
    normalize_coords=True,
    normalize_values=True,
    initial_shuffle=False,  # No shuffle for reconstruction
)
reconst_dataloader = DataLoader(
    reconst_dataset,
    batch_size=None,  # batching done in IterableDataset
    num_workers=0,  # Single process for eval
    pin_memory=pin,
)
# Reconstruct the entire volume from the INR
model.eval()
with torch.no_grad():
    data_shape_tensor = torch.as_tensor(data_shape, device=device, dtype=run_dtype)
    reconst_data = torch.zeros(data_shape, device=device, dtype=run_dtype)
    for x, _ in tqdm(reconst_dataloader):
        x = x.to(device, dtype=run_dtype, non_blocking=True)
        y = model(x).to(dtype=run_dtype, non_blocking=True)

        indices = (x * (data_shape_tensor - 1)).long()
        i, j, k = indices.split(1, dim=-1)

        # (batch_size,)
        i, j, k = i.squeeze(), j.squeeze(), k.squeeze()
        y = y.squeeze()

        reconst_data[i, j, k] = y

# %%
gt_data = torch.as_tensor(
    dataset.volume_data(), device=device, dtype=run_dtype
).contiguous()
reconst_data = torch.clamp(reconst_data, 0.0, 1.0).contiguous()

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
