"""
A bare-bones script for demonstrating INR training and evaluation,
supporting both MLP and KAN architectures.
"""

import torch
import numpy as np
from networks import INR_Base

from samplers import VolumeSampler
from benchmark import parse_filename, train, reconstruct, compute_metrics, save

# Data and network parameters
data_path = "./data/nucleon_41x41x41_uint8.raw"
network_type = "kan"  # "kan" or "mlp"
n_hidden_layers = 4
n_neurons = 32

# see multi-resolution hash encoding https://arxiv.org/abs/2201.05989
n_levels = 4
n_features_per_level = 8
log2_hashmap_size = 19
base_resolution = 16
per_level_scale = 2.0

lrate = 0.001
lrate_decay = 16
epochs = 20
batch_size = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
native_encoder = device.type == "cpu"
native_network = device.type == "cpu" or network_type == "kan"

data_name, data_shape, data_type = parse_filename(data_path)

# Save parameters
save_dir = "./"
name = "my_inr_name"
save_model = False
save_reconstruction = False  # CHANGE ME

# For some reason the TCNN multi-resolution encoder seems to be numerically
# unstable with KAN on tiny datasets (e.g., nucleon_41x41x41_uint8.raw).
# This isn't observed during benchmarking with larger datasets, or with the native encoder.
# Suppressing NaNs is a work-around good enough for the demo script.
suppress_encoder_nan = not native_encoder and network_type == "kan"

model = INR_Base(
    n_input_dims=3,
    n_output_dims=1,
    native_encoder=native_encoder,
    native_network=native_network,
    network_type=network_type,
    n_hidden_layers=n_hidden_layers,
    n_neurons=n_neurons,
    n_levels=n_levels,
    n_features_per_level=n_features_per_level,
    log2_hashmap_size=log2_hashmap_size,
    base_resolution=base_resolution,
    per_level_scale=per_level_scale,
    suppress_encoder_nan=suppress_encoder_nan,
)
model.to(device, non_blocking=True)

# Create optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=lrate,
    betas=(0.9, 0.999),
    eps=1e-09,
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lrate_decay, gamma=0.5)

print(f"Loading {data_name} with shape {data_shape} and type {data_type}...")
data = np.fromfile(data_path, dtype=data_type)

# Prepare data and sampler
sampler = VolumeSampler(data_shape, data_type, device)
sampler.load_from_ndarray(data)

# Training loop
print("Starting training...")
for i in range(epochs):
    loss, psnr, duration = train(
        sampler, model, optimizer, batch_size, enable_pbar=True
    )
    print(
        f"Epoch {i + 1}/{epochs}: "
        f"Loss = {loss:.4f}, "
        f"PSNR = {psnr:.2f}dB, "
        f"Duration = {duration:.2f}sec"
    )
    scheduler.step()

# Data decompression
print("Reconstructing data...")
reconst_data = reconstruct(
    model,
    data_shape,
    device,
    batch_size,
    enable_pbar=True,
)

# Compute metrics
print("Computing metrics...")
gt_data = sampler.gt.data.reshape(data_shape)
psnr, ssim, mse = compute_metrics(
    gt_data,
    reconst_data,
    enable_pbar=True,
)
print(f"Decompression results: PSNR = {psnr}, SSIM = {ssim}, MSE = {mse}")

# Save the INR and/or decompressed data
save(
    save_dir,
    model,
    reconst_data,
    name,
    data_type,
    data_shape,
    save_model=save_model,
    save_reconstruction=save_reconstruction,
)
