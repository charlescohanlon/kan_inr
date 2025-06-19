# %%
import numpy as np
from torch.utils.data import random_split, DataLoader
from src.volumetric_dataset import VolumetricDataset
import os

# Data
data_path = "/data/nucleon/nucleon_41x41x41_uint8.raw"
data_shape = (41, 41, 41)
data_type = np.uint8
output_filename = "nucleon_output.csv"
inr_name = "nucleon_kan"
verbose = True
enable_pbar = True

# Train
batch_size = int(8 * 10**4.8)
shuffle = True
num_workers = 48
pin_memory = True
num_epochs = 100

print("Creating datasets")
dataset = VolumetricDataset(
    data_path, data_shape, data_type, normalize_coords=True, normalize_values=True
)
loader = DataLoader(dataset, batch_size, shuffle, num_workers=num_workers)

# %%
from torch.optim import AdamW
from src.efficient_kan.kan import KAN
import torch
import torch.nn as nn

lr = 0.008
device = torch.device("cuda")
dtype = torch.float32

print("Creating model and optimizer")
model = KAN(
    layers_hidden=[3, 16, 16, 1],
    grid_size=5,
    spline_order=3,
    scale_noise=0.1,
    scale_base=1,
    scale_spline=1,
    base_activation=nn.SiLU,
    grid_eps=0.02,
    grid_range=[-1, 1],
)
model.to(device, dtype)
print(model)
optimizer = AdamW(model.parameters(), lr=lr)
print(optimizer)

# %%
import torch
import torch.nn as nn
import torcheval.metrics.functional as tmf
from einops import rearrange
from time import time
from tqdm import tqdm

data_range = 1.0
loss_fn = nn.functional.mse_loss

with open(output_filename, "w") as output_file:
    output_file.write("epoch,avg_loss\n")


def train():
    print("Beginning training")
    model.train()
    for epoch in range(num_epochs):
        start = time()
        total_loss = 0
        for x, y_hat in tqdm(loader, disable=not enable_pbar):
            x = x.to(device, dtype)
            y_hat = y_hat.to(device, dtype)
            y = model(x)
            loss = loss_fn(y, y_hat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        total_observations = len(loader) * loader.batch_size
        avg_loss = total_loss / total_observations
        duration = time() - start
        if verbose:
            print(f"(epoch {epoch}): avg loss = {avg_loss}, duration = {duration}")
        with open(output_filename, "a") as output_file:
            output_file.write(f"{epoch},{avg_loss}\n")


def eval(save_model=False):
    print("Beginning eval")
    model_name = f"{inr_name}_KAN_INR.pt"
    if save_model:
        torch.save(model, model_name)
    with torch.no_grad():
        reconst_data = torch.zeros(dataset.data_shape, device=device, dtype=dtype)
        loader.dataset.return_indices = True
        for x, _, (i, j, k) in loader:
            x = x.to(device, dtype)
            y = model(x)
            for y_s, i_s, j_s, k_s in zip(y, i, j, k):
                reconst_data[i_s.item(), j_s.item(), k_s.item()] = y_s.item()

        gt_data = torch.tensor(dataset.volume_data(), device=device, dtype=dtype)

        reconst_data = rearrange(reconst_data, "h w c -> 1 c h w")
        gt_data = rearrange(gt_data, "h w c -> 1 c h w")
        psnr = tmf.peak_signal_noise_ratio(reconst_data, gt_data, data_range)

    print(f"psnr = {psnr}")


train()
eval(save_model=True)
