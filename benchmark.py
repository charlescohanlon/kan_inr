"""
Benchmark script for comparing MLP and KAN Implicit Neural Representations (INRs).

This script provides functionality to train and evaluate different types of neural
representations on volumetric data, supporting both single-GPU and multi-GPU
distributed training via PyTorch's DistributedDataParallel (DDP).
"""

from dataclasses import dataclass, field
from functools import partial
import gc
from typing import List, Optional, Tuple
from pathlib import Path
from uuid import uuid4
import hydra
from hydra.core.config_store import ConfigStore
import numpy as np
from omegaconf import OmegaConf
from tqdm import trange
from time import time
import math

import torch
from torchmetrics.functional.image import structural_similarity_index_measure

import os
import shutil
from math import log

from networks import INR_Base
from samplers import VolumeSampler

from tempfile import TemporaryDirectory
from pprint import pprint

# DDP imports
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import hashlib

# Set seeds for reproducibility across all devices
torch.manual_seed(0)  # for reproducibility
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)


@dataclass
class BenchmarkConfig:
    """
    Configuration dataclass for benchmark runs.

    Attributes:
        params_file: Path to YAML file containing hyperparameter configurations
        network_types: List of network architectures to benchmark (e.g., ["mlp", "kan"])
        home_dir: Base directory for all paths
        data_path: Relative path from home_dir to data directory
        batch_size: Training batch size (calculated automatically if None)
        output_filename: Name for results CSV file (auto-generated if None)
        enable_pbar: Whether to show progress bars during training
        repeats: Number of training runs to average for each configuration
        save_mode: "largest", "smallest", or a specific hashtable size to save, None to skip
        safety_margin: Memory safety factor for batch size calculation (0-1)
        dataset: Filter to only benchmark this specific dataset
        hashmap_size: Filter to only use this specific log2 hashmap size
        epochs: Override epochs for all runs (uses params_file value if None)
        ssd_dir: Optional SSD directory for faster I/O during training
        test_epoch_run: If True, only run one epoch to estimate time (for job scheduling)
        output_dir: Directory to save result CSV
    """

    params_file: str = "params"
    network_types: List[str] = field(default_factory=lambda: ["fkan", "mlp"])
    home_dir: str = "/grand/insitu/cohanlon"
    data_path: str = "datasets/raw"
    params_path: str = "kan_inr/params"
    batch_size: Optional[int] = None
    output_filename: Optional[str] = None
    enable_pbar: bool = True
    repeats: int = 1
    save_mode: Optional[str] = None
    safety_margin: float = 0.8
    dataset: Optional[str] = None
    hashmap_size: Optional[int] = None
    epochs: Optional[int] = None
    ssd_dir: Optional[str] = None
    test_epoch_run: bool = False
    output_dir: str = str(Path(home_dir) / "results")


@dataclass
class KANParams:
    """
    Parameters specific to KAN networks.

    Attributes:
        grid_radius: Initial grid radius for KAN (can also be a list of two bounds)
        grid_radius_step: Step size for grid radius adjustment
        num_grids: Number of grids in KAN (can also be a list of two bounds)
        num_grids_step: Step size for num_grids adjustment
        use_base_update: Whether to use base update in KAN
    """

    grid_radius: float = 1.0
    grid_radius_step: float = 1.0
    num_grids: int = 8
    num_grids_step: int = 1
    use_base_update: bool = True


@dataclass
class RunParams:
    """
    Parameters for a single benchmark run.

    Attributes:
        dataset_name: Name of the dataset (e.g., "richtmyer_meshkov")
        network_type: Type of network architecture ("mlp" or "kan")
        lrate: Initial learning rate
        lrate_decay: Number of epochs between learning rate decay steps
        epochs: Total number of training epochs
        n_neurons: Number of neurons per hidden layer
        n_hidden_layers: Number of hidden layers in the network
        n_levels: Number of hash table levels (for multi-resolution encoding)
        n_features_per_level: Features per hash table level
        per_level_scale: Scale factor between consecutive levels
        log2_hashmap_size: Log2 of the hash table size (can also be list of two bounds)
        log2_hashmap_size_step: Step size for sweeping log2 hashmap sizes
        base_resolution: Base resolution for the hash encoding
        zfp_enc: ZFP compression parameter for encoder (unused in current implementation)
        zfp_mlp: ZFP compression parameter for MLP (unused in current implementation)
        kan_params: Optional dictionary of KAN-specific parameters
        special_mode: Special mode for additional configurations e.g., SIREN, etc.
    """

    dataset_name: str
    network_type: str
    lrate: float
    lrate_decay: int
    epochs: int
    n_neurons: int
    n_hidden_layers: int
    n_levels: int
    n_features_per_level: int
    per_level_scale: float
    log2_hashmap_size: int
    base_resolution: str
    zfp_enc: float
    zfp_mlp: float
    log2_hashmap_size_step: int = 1
    kan_params: Optional[KANParams] = None
    special_mode: Optional[str] = None

    # "None", "Sigmoid", "ReLU"
    activation: str = "ReLU"
    output_activation: str = "None"

    def epoch_time_hash(
        self, safety_margin: int, dataset_name: str, ssd_dir_provided: bool
    ) -> int:
        """
        Custom hash function for a run's epoch time to uniquely identify configurations.
        Considers parameters relevant to training epoch time, excluding those that don't affect it.

        Args:
            safety_margin: Safety margin for the configuration
            dataset_name: Name of the dataset being used
            ssd_dir_provided: Whether an SSD directory is provided for I/O
        """
        kan_params = (
            [
                self.kan_params.grid_radius,
                self.kan_params.num_grids,
                self.kan_params.use_base_update,
                # NOTE: may add ekan-specific params in future.
                # But this isn't a problem b/c network_type="ekan" should
                # hash differently anyway.
            ]
            if self.kan_params
            else []
        )

        # Would be empty string if None so use "None"
        special_mode = self.special_mode if self.special_mode else "None"

        config_tuple = (
            self.dataset_name,
            self.network_type,
            self.lrate,
            self.lrate_decay,
            self.n_neurons,
            self.n_hidden_layers,
            self.n_levels,
            self.n_features_per_level,
            self.per_level_scale,
            self.log2_hashmap_size,
            self.base_resolution,
            self.activation,
            self.output_activation,
            special_mode,
            dataset_name,
            safety_margin,
            ssd_dir_provided,
            *kan_params,
        )
        param_str = repr(config_tuple)

        # Deterministic hash
        return int(hashlib.md5(param_str.encode("utf-8")).hexdigest(), 16)


def run_benchmark(
    data_path: Path,
    params: RunParams,
    cfg: BenchmarkConfig,
    should_save: bool = False,
    rank: int = 0,
    world_size: int = 1,
    local_rank: int = 0,
):
    """
    Execute a single benchmark run with the specified parameters.

    This function handles the complete training and evaluation pipeline for one
    configuration, including multiple training runs for averaging if specified.

    Args:
        data_path: Path to the raw volumetric data file
        params: Configuration parameters for this run
        cfg: Global benchmark configuration
        should_save: Whether to save the trained model and reconstruction
        rank: Global rank for distributed training
        world_size: Total number of processes
        local_rank: Local rank for GPU assignment
    """
    # Check if we're in DDP mode
    is_ddp = dist.is_initialized()
    is_main_process = rank == 0

    # Set device and dtype based on availability
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(local_rank)
    else:
        device = torch.device("cpu")

    # Parse dataset information from filename
    _, data_shape, data_type = parse_filename(data_path)

    # Prepare data and sampler for coordinate-based training
    sampler = VolumeSampler(data_shape, data_type, device)
    sampler.load_from_file(filename=data_path)

    # Use bounding boxes in 3D to partition sampling among ranks for DDP
    # Each rank will train on a different spatial region of the volume
    top_corner, bottom_corner = partition_volume(data_shape, rank, world_size)
    sampler.set_bounds([top_corner, bottom_corner])
    print(f"Rank {rank}: Sampler bounds set to {top_corner} - {bottom_corner}")

    # Determine whether to use native PyTorch implementations
    # KAN networks and CPU training require native implementations
    native_encoder = device.type == "cpu"
    native_network = device.type == "cpu" or params.network_type.lower() != "mlp"

    psnrs, ssims, mses = [], [], []
    num_repeats = cfg.repeats

    # Run multiple training sessions and average the results
    for repeat in range(num_repeats):  # averages results over num_repeats many runs
        if is_main_process:
            print(f"\nRunning repeat {repeat + 1}/{num_repeats}")

        # Create model with specified architecture
        model = INR_Base(
            n_input_dims=3,  # 3D coordinates (x, y, z)
            n_output_dims=1,  # Single scalar output per voxel
            native_encoder=native_encoder,
            native_network=native_network,
            network_type=params.network_type,
            n_hidden_layers=params.n_hidden_layers,
            n_neurons=params.n_neurons,
            n_levels=params.n_levels,
            n_features_per_level=params.n_features_per_level,
            log2_hashmap_size=params.log2_hashmap_size,
            base_resolution=params.base_resolution,
            per_level_scale=params.per_level_scale,
            kan_params=params.kan_params,
        )
        model.to(device, non_blocking=True)

        # Create optimizer with Adam
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=params.lrate,
            betas=(0.9, 0.999),
            eps=1e-09,  # weight_decay=1e-15,
            # amsgrad=True, foreach=True #, fused=True,
        )
        # Learning rate scheduler for decay
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=params.lrate_decay, gamma=0.5
        )

        # Calculate optimal batch size if not specified
        if cfg.batch_size is None:
            # Compute batch size based on dataset and model
            # Use the underlying model for batch size calculation
            if is_main_process:
                batch_size = calculate_batch_size(
                    model,
                    device,
                    data_shape,
                    is_training=True,
                    optimizer=optimizer,
                    sampler=sampler,
                    safety_margin=cfg.safety_margin,
                )
            else:
                # Initialize batch_size for non-main processes before broadcast
                batch_size = 0
            # Broadcast batch size to all ranks
            if is_ddp:
                batch_size = torch.tensor(batch_size, device=device)
                dist.broadcast(batch_size, src=0)
                batch_size = batch_size.item()
        else:
            # Doesn't need to be broadcast if fixed
            batch_size = cfg.batch_size

        # Wrap model with DDP if in distributed mode
        if is_ddp:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)

        if is_main_process:
            print("Training INR...")

        if is_ddp:
            dist.barrier()  # Ensure all ranks start training together

        # Training loop
        durations = []
        for i in range(params.epochs):
            # Train the model (using Qi's sampler training code)
            loss, psnr, duration = train(
                sampler,
                model,
                optimizer,
                batch_size,
                enable_pbar=cfg.enable_pbar,
                rank=rank,
                world_size=world_size,
            )

            if cfg.test_epoch_run:
                return  # Just testing epoch time

            if is_ddp:
                torch.cuda.synchronize()  # Ensure CUDA ops complete
                dist.barrier()  # Ensure all ranks finish epoch

            durations.append(duration)
            if is_main_process:
                print(
                    f"Epoch {i + 1}/{params.epochs}: "
                    f"Loss = {loss:.4f}, "
                    f"PSNR = {psnr:.2f}dB, "
                    f"Duration = {duration:.2f}sec"
                )
            scheduler.step()  # Step the learning rate scheduler

        # Reconstruct and compute metrics when finished training
        if is_main_process:
            print("Reconstructing INR volume...")
            reconst_data = reconstruct(
                model.module if is_ddp else model,  # Unwrap DDP model if needed
                data_shape,
                device,
                batch_size=cfg.batch_size,
                safety_margin=cfg.safety_margin,
                enable_pbar=cfg.enable_pbar,
            )

            print("Computing metrics...")
            gt_data = sampler.gt.data.reshape(data_shape)
            psnr, ssim, mse = compute_metrics(
                gt_data,
                reconst_data,
                enable_pbar=cfg.enable_pbar,
            )

            psnrs.append(psnr)
            ssims.append(ssim)
            mses.append(mse)

            print(
                f"Repeat {repeat + 1}/{num_repeats}: PSNR = {psnr}, SSIM = {ssim}, MSE = {mse}"
            )
            if repeat + 1 < num_repeats:
                del reconst_data

        # Clean up memory after all repeats except last
        if repeat + 1 < num_repeats:
            del model
            gc.collect()
            torch.cuda.empty_cache()

        # Synchronize all processes before next repeat
        if is_ddp:
            dist.barrier()

    # Only compute and save results on main process
    if is_main_process:
        # Calculate average metrics across all repeats
        avg_psnr = sum(psnrs) / num_repeats
        std_dev_psnr = np.std(psnrs)
        avg_ssim = sum(ssims) / num_repeats
        std_dev_ssim = np.std(ssims)
        avg_mse = sum(mses) / num_repeats
        std_dev_mse = np.std(mses)

        print("Final Results")
        print("  Avg PSNR:", avg_psnr)
        print("  Std Dev PSNR:", std_dev_psnr)
        print("  Avg SSIM:", avg_ssim)
        print("  Std Dev SSIM:", std_dev_ssim)
        print("  Avg MSE:", avg_mse)
        print("  Std Dev MSE:", std_dev_mse)

        # Compute compression ratio
        with TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            name = hash(model) % (10**8)
            inr_path = tmpdir_path / f"{name}.pt"
            torch.save(model.state_dict(), inr_path)
            model_size = inr_path.stat().st_size

        volume_size = sampler.numvoxels * np.dtype(data_type).itemsize
        compression_ratio = volume_size / model_size
        print(
            f"Volume size: {volume_size:.2f} bytes, "
            f"Model size: {model_size:.2f} bytes, "
            f"Compression ratio: {compression_ratio:.2f}",
        )

        avg_time_per_epoch = sum(durations) / len(durations)
        num_gpus = torch.cuda.device_count()

        if params.kan_params is not None:
            kan_grid_radius = params.kan_params.grid_radius
            kan_grid_radius_step = params.kan_params.grid_radius_step
            kan_num_grids = params.kan_params.num_grids
            kan_num_grids_step = params.kan_params.num_grids_step
        else:
            kan_grid_radius = None
            kan_grid_radius_step = None
            kan_num_grids = None
            kan_num_grids_step = None

        pbs_job_id = os.getenv("PBS_JOBID")
        pbs_array_index = os.getenv("PBS_ARRAY_INDEX", 0)
        epoch_time_hash = params.epoch_time_hash(
            safety_margin=cfg.safety_margin,
            dataset_name=params.dataset_name,
            ssd_dir_provided=cfg.ssd_dir is not None,
        )

        # Set up output directory and file
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        if cfg.output_filename is None:
            # Create a unique filename
            uid = uuid4().hex
            unique_filename = uid + "_results.csv"
            output_path = Path(cfg.output_dir) / unique_filename
            print(f"No output filename specified, using {unique_filename}")
        else:
            output_path = Path(cfg.output_dir) / cfg.output_filename

        # Write results to CSV
        with open(output_path, "a") as f:
            f.write(
                ",".join(
                    [
                        # the (ordering of) datapoints collected
                        "dataset_name",
                        "data_size",
                        "inr_size",
                        "network_type",
                        "epoch_count",
                        "num_neurons",
                        "num_hidden_layers",
                        "num_levels",
                        "num_features_per_level",
                        "per_level_scale",
                        "log2_hashmap_size",
                        "base_resolution",
                        "num_repeats",
                        "avg_time_per_epoch",
                        "num_gpus",
                        "compression_ratio",
                        "kan_grid_radius",
                        "kan_grid_radius_step",
                        "kan_num_grids",
                        "kan_num_grids_step",
                        "batch_size",
                        "pbs_job_id",
                        "pbs_array_index",
                        "epoch_time_hash",
                        "avg_psnr",
                        "std_dev_psnr",
                        "avg_ssim",
                        "std_dev_ssim",
                        "avg_mse",
                        "std_dev_mse",
                    ]
                )
                + "\n"
                + ",".join(
                    map(
                        str,
                        [
                            # all the datapoints collected
                            params.dataset_name,
                            volume_size,
                            model_size,
                            params.network_type,
                            params.epochs,
                            params.n_neurons,
                            params.n_hidden_layers,
                            params.n_levels,
                            params.n_features_per_level,
                            params.per_level_scale,
                            params.log2_hashmap_size,
                            params.base_resolution,
                            cfg.repeats,
                            avg_time_per_epoch,
                            num_gpus,
                            compression_ratio,
                            kan_grid_radius,
                            kan_grid_radius_step,
                            kan_num_grids,
                            kan_num_grids_step,
                            batch_size,
                            pbs_job_id,
                            pbs_array_index,
                            epoch_time_hash,
                            avg_psnr,
                            std_dev_psnr,
                            avg_ssim,
                            std_dev_ssim,
                            avg_mse,
                            std_dev_mse,
                        ],
                    )
                )
            )

        # Save model and reconstruction if requested
        if should_save:
            # Un-normalize data before saving
            data_min, data_max = sampler.gt.minmax
            if 0 <= reconst_data.min() and reconst_data.max() <= 1.0:
                reconst_data = (  # un-normalize data (and move to cpu) before saving
                    reconst_data.cpu() * (data_max - data_min) + data_min
                )
            inr_name = "_".join(  # made up of parameters being swept
                [
                    params.dataset_name,
                    params.network_type,
                    str(params.log2_hashmap_size),
                ]
            )
            save(
                cfg.home_dir,
                model.module if is_ddp else model,
                reconst_data,
                inr_name,
                save_type=data_type,
                save_shape=data_shape,
            )


def mse2psnr(x):
    """Convert MSE to PSNR (Peak Signal-to-Noise Ratio)."""
    if torch.is_tensor(x):
        return -10.0 * torch.log(x) / log(10.0)

    return -10.0 * np.log(x) / log(10.0)


def mse_loss(x, y):
    """Calculate MSE loss between predictions and targets."""
    return torch.nn.MSELoss()(x, y) if torch.is_tensor(x) else np.mean((x - y) ** 2)


def train(
    sampler: VolumeSampler,
    model: INR_Base,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    enable_pbar: bool = True,
    rank: int = 0,
    world_size: int = 1,
    **kwargs,
):
    """
    Train the INR model for one epoch using random coordinate sampling.

    Args:
        sampler: Volume sampler for generating coordinate-value pairs
        model: INR model to train
        optimizer: PyTorch optimizer
        batch_size: Number of coordinates to sample per iteration
        enable_pbar: Whether to show progress bar
        rank: Process rank for distributed training
        world_size: Total number of processes
        **kwargs: Additional unused arguments for compatibility

    Returns:
        tuple: (final_loss, final_psnr, epoch_duration) metrics from the epoch
    """
    is_main_process = rank == 0
    num_voxels = sampler.numvoxels

    # Calculate number of steps for approximately one epoch
    # Divided amongst ranks for distributed training
    num_steps = math.ceil(num_voxels / (batch_size * world_size))

    # Training step function
    def task():
        """Execute one training step."""
        optimizer.zero_grad()

        # Sample random coordinates and their corresponding values
        with torch.no_grad():
            coords, targets = sampler.get_random_samples(batch_size)

        # Forward pass
        values = model(coords)
        # L1 loss typically works better than MSE for INRs
        loss = torch.nn.L1Loss()(values, targets)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Evaluation metrics (no gradients needed)
        with torch.no_grad():
            psnr = mse2psnr(mse_loss(values, targets))

        return loss, psnr

    # Setup progress tracking
    verbose = enable_pbar and is_main_process
    time0 = time()

    # Training Loop for one epoch
    progress = trange(1, num_steps + 1) if verbose else range(1, num_steps + 1)
    for _ in progress:
        loss, psnr = task()
        if verbose:
            progress.set_postfix_str(
                f"Loss: {loss:7.6f}, PSNR: {psnr:5.3f}dB",
                refresh=True,
            )
    duration = time() - time0
    return loss, psnr, duration


@torch.no_grad()
def reconstruct(
    model: INR_Base,
    data_shape: Tuple[int, int, int],
    device: torch.device,
    batch_size: int = None,
    safety_margin: float = 0.99,
    enable_pbar: bool = True,
):
    """
    Reconstruct the full volume using the trained INR model.

    Performs batched inference to avoid out-of-memory errors when reconstructing
    large volumes. Generates a regular grid of coordinates and evaluates the
    model at each point.

    Args:
        model: Trained INR model
        data_shape: Shape of the volume to reconstruct (x, y, z)
        device: Device to perform reconstruction on
        batch_size: Number of coordinates to process at once (auto-calculated if None)
        safety_margin: Memory safety factor for batch size calculation
        enable_pbar: Whether to show progress bar

    Returns:
        torch.Tensor: Reconstructed volume of shape data_shape, clamped to [0, 1]
    """
    model.eval()

    # Calculate batch size if not provided
    if batch_size is None:
        # Use the calculate_batch_size function for evaluation
        batch_size = calculate_batch_size(
            model=model,
            device=device,
            data_shape=data_shape,
            is_training=False,
            safety_margin=safety_margin,
        )
        print(f"Using batch size {batch_size:,} for reconstruction")

    # Generate regular grid of coordinates
    # Using cell-centered coordinates (hence the 0.5 offset)
    xs = np.linspace(
        0.5 / data_shape[0],
        (data_shape[0] - 0.5) / data_shape[0],
        data_shape[0],
    )
    ys = np.linspace(
        0.5 / data_shape[1],
        (data_shape[1] - 0.5) / data_shape[1],
        data_shape[1],
    )
    zs = np.linspace(
        0.5 / data_shape[2],
        (data_shape[2] - 0.5) / data_shape[2],
        data_shape[2],
    )

    # Create meshgrid and reshape to list of coordinates
    coords = (
        np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1)
        .transpose(2, 1, 0, 3)  # Rearrange spatial dims to z, y, x
        .reshape(-1, 3)
    )
    num_coords = coords.shape[0]
    reconst_list = []

    # Process coordinates in batches
    for i in trange(0, num_coords, batch_size, disable=not enable_pbar):
        end_idx = min(i + batch_size, num_coords)
        batch_coords = torch.tensor(
            coords[i:end_idx],
            dtype=torch.float32,
            device=device,
        )
        # Forward pass for this batch
        batch_output = model(batch_coords)

        # Store parts on cpu (OOM for some reason if don't do this)
        reconst_list.append(batch_output.cpu())

        # Clean up memory
        del batch_coords
        if i % (batch_size * 10) == 0:  # Periodic cache clear
            torch.cuda.empty_cache()

    # Combine all batches and reshape to volume
    reconst_data = (
        torch.cat(reconst_list, dim=0).reshape(data_shape).to(device).contiguous()
    )
    reconst_data.clamp_(0.0, 1.0)  # Clamp to valid range [0, 1]
    return reconst_data


@torch.no_grad()
def compute_metrics(
    gt_data: torch.Tensor, reconst_data: torch.Tensor, enable_pbar: bool = True
):
    """
    Compute reconstruction quality metrics (PSNR, SSIM, MSE).

    Processes the volume slice-by-slice for numerical stability with large 3D volumes.
    All input data must be normalized to [0, 1] range.

    NOTE: ssim metric is computed slice-by-slice along the z-axis and then averaged.

    Args:
        gt_data: Ground truth volume tensor in [0, 1] range
        reconst_data: Reconstructed volume tensor in [0, 1] range
        enable_pbar: Whether to show progress bar

    Returns:
        tuple: (psnr, mse, ssim) averaged across all slices

    Raises:
        ValueError: If data is not in [0, 1] range, shapes mismatch, or not 3D
    """
    # Validate inputs
    if gt_data.min() < 0.0 or gt_data.max() > 1.0:
        raise ValueError("Ground truth data must be in [0, 1] range")
    if reconst_data.min() < 0.0 or reconst_data.max() > 1.0:
        raise ValueError("Reconstructed data must be in [0, 1] range")
    if gt_data.shape != reconst_data.shape:
        raise ValueError(f"Shape mismatch: {gt_data.shape} vs {reconst_data.shape}")
    if len(gt_data.shape) != 3:
        raise ValueError("Metrics computation only supports 3D volumes")

    mse = mse_loss(reconst_data, gt_data)
    psnr = mse2psnr(mse)

    # Compute SSIM slice-by-slice
    ssim_values = []
    num_slices = gt_data.shape[2]
    for i in trange(num_slices, disable=not enable_pbar):

        # Shape: (1, 1, X, Y), Z axis is sliced
        gt_slice = gt_data[:, :, i].unsqueeze(0).unsqueeze(0)
        reconst_slice = reconst_data[:, :, i].unsqueeze(0).unsqueeze(0)

        ssim_values.append(
            structural_similarity_index_measure(reconst_slice, gt_slice, data_range=1.0)
        )

    avg_ssim = torch.stack(ssim_values).mean()

    return psnr.item(), avg_ssim.item(), mse.item()


@torch.no_grad()
def save(
    save_dir: Path,
    model: INR_Base,
    reconst_data: torch.Tensor,
    inr_name: str,
    save_type: np.dtype,
    save_shape: Tuple[str, str, str],
    save_order: str = "C",
    save_model: bool = True,
    save_reconstruction: bool = True,
):
    """
    Save the trained INR model and/or reconstructed volume to disk.

    Args:
        save_dir: Base directory for saving
        model: Trained INR model
        reconst_data: Reconstructed volume tensor
        inr_name: Base name for saved files
        save_type: NumPy dtype for saving reconstruction
        save_shape: Shape of the volume (for filename)
        save_order: Memory ordering for saving ('C' or 'F')
        save_model: Whether to save the model weights
        save_reconstruction: Whether to save the reconstructed volume
    """
    save_dir = Path(save_dir)

    if save_model:
        # Save the INR model weights
        save_path = save_dir / "models_saved" / f"{inr_name}.pt"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    if save_reconstruction:
        # Save the reconstruction as .raw file with metadata in filename
        shape_str = "x".join(map(str, save_shape))
        type_str = np.dtype(save_type).name
        reconst_path = (
            save_dir
            / "reconstructions_saved"
            / f"{inr_name}_{shape_str}_{type_str}.raw"
        )
        reconst_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to bytes with original data type
        reconst_bytes = (
            reconst_data.flatten()
            .contiguous()
            .cpu()
            .numpy()
            .astype(save_type)
            .tobytes(save_order)
        )
        # Write as binary .raw file
        with open(reconst_path, "wb") as f:
            f.write(reconst_bytes)
        print(f"Reconstruction saved to {reconst_path}")


def calculate_batch_size(
    model: INR_Base,
    device,
    data_shape,
    is_training=True,
    optimizer=None,
    sampler=None,
    min_batch=1,
    max_batch=10_000_000,
    safety_margin=0.99,
):
    """
    Automatically determine maximum batch size that fits in GPU memory.
    Uses actual train() and reconstruct() functions to empirically test
    batch sizes rather than simulating memory usage.

    Args:
        model: The INR model to test
        device: Device to test on (CPU or CUDA)
        data_shape: Shape of the data (for reconstruction memory estimation)
        is_training: Whether to test training (with gradients) or inference
        optimizer: Optimizer to test with (required for training)
        sampler: VolumeSampler for training tests (required for training)
        min_batch: Minimum batch size to consider
        max_batch: Maximum batch size to consider
        safety_margin: Safety factor to apply to final batch size (0-1)

    Returns:
        int: Maximum safe batch size for the given configuration
    """
    if device.type != "cuda":
        return 100_000  # Reasonable default for CPU

    # Validation
    if is_training and (optimizer is None or sampler is None):
        raise ValueError(
            "Optimizer and sampler required for training batch size calculation"
        )

    # Save original states on cpu to restore later
    original_model_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
    original_optimizer_state = None
    if optimizer is not None:
        original_optimizer_state = optimizer.state_dict()

    try:
        # First, do a coarse search to find the right order of magnitude
        test_sizes = [
            100,
            1_000,
            10_000,
            50_000,
            100_000,
            500_000,
            1_000_000,
            5_000_000,
        ]
        working_batch = 1

        for test_batch in test_sizes:
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                if is_training:
                    # Run one train step
                    optimizer.zero_grad()

                    # Sample real data
                    with torch.no_grad():
                        coords, targets = sampler.get_random_samples(test_batch)

                    # Forward pass
                    values = model(coords)
                    loss = torch.nn.L1Loss()(values, targets)

                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    # Clean up
                    del coords, targets, values, loss
                else:
                    # Test reconstruction with actual function logic
                    with torch.no_grad():
                        # Generate test coordinates like reconstruct() does
                        test_coords = torch.rand(
                            test_batch, 3, device=device, dtype=torch.float32
                        )

                        # Forward pass
                        output = model(test_coords)

                        # Also test if we can allocate reconstruction tensor
                        if test_batch > 100_000:  # Only test for larger batches
                            test_reconst = torch.zeros(
                                data_shape, device=device, dtype=torch.float32
                            )
                            del test_reconst

                        del test_coords, output

                working_batch = test_batch

            except RuntimeError as e:
                if "out of memory" in str(e) or "CUDA" in str(e):
                    torch.cuda.empty_cache()
                else:
                    raise e

        # Now binary search between working_batch and the next level up
        if working_batch == test_sizes[-1]:
            # We succeeded at the highest test, search higher
            min_batch = working_batch
            max_batch = max_batch
        else:
            # Search between working and failed size
            min_batch = working_batch
            max_batch = min(working_batch * 10, max_batch)

        # Binary search for the optimal batch size
        best_batch = working_batch

        while min_batch <= max_batch:
            mid_batch = (min_batch + max_batch) // 2

            # Skip if we've already tested something very close
            if abs(mid_batch - best_batch) < best_batch * 0.01:  # Within 1%
                break

            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                if is_training:
                    # Run training step
                    optimizer.zero_grad()

                    with torch.no_grad():
                        coords, targets = sampler.get_random_samples(mid_batch)

                    values = model(coords)
                    loss = torch.nn.L1Loss()(values, targets)
                    loss.backward()
                    optimizer.step()

                    # Success!
                    best_batch = mid_batch
                    min_batch = mid_batch + 1

                    # Clean up
                    del coords, targets, values, loss
                    optimizer.zero_grad()
                else:
                    # Test reconstruction
                    with torch.no_grad():
                        # Simulate reconstruct() function's memory usage
                        # Generate coordinates
                        num_test_coords = min(mid_batch, np.prod(data_shape))
                        test_coords = torch.rand(
                            num_test_coords, 3, device=device, dtype=torch.float32
                        )

                        # Forward pass
                        output = model(test_coords)

                        # Test intermediate storage (simulating reconst_list)
                        # reconstruct() stores outputs on CPU then moves back
                        cpu_output = output.cpu()

                        # Test if we can allocate the full reconstruction tensor
                        # This is critical for the reconstruct function
                        if mid_batch > best_batch * 1.2:  # Significant increase
                            test_reconst = torch.zeros(
                                data_shape, device=device, dtype=torch.float32
                            )
                            del test_reconst

                        # Success!
                        best_batch = mid_batch
                        min_batch = mid_batch + 1

                        # Clean up
                        del test_coords, output, cpu_output

            except RuntimeError as e:
                if "out of memory" in str(e) or "CUDA" in str(e):
                    # Failed, try smaller batch
                    max_batch = mid_batch - 1
                    torch.cuda.empty_cache()
                else:
                    raise e

    finally:
        # Move model state back to original device
        original_model_state = {
            k: v.to(device) for k, v in original_model_state.items()
        }

        # Restore original states
        model.load_state_dict(original_model_state)
        if optimizer is not None and original_optimizer_state is not None:
            optimizer.load_state_dict(original_optimizer_state)

        # Clean up the saved states
        del original_model_state
        if original_optimizer_state is not None:
            del original_optimizer_state

        torch.cuda.empty_cache()

    # Apply safety margin to avoid occasional OOM during actual usage
    final_batch = max(1, int(best_batch * safety_margin))

    # Sanity check
    if final_batch < 1000 and device.type == "cuda":
        print(
            f"Warning: Unexpectedly small batch size {final_batch} for INR on GPU. "
            f"Memory may be fragmented or model may be very large."
        )

    return final_batch


def partition_volume(data_shape, rank, world_size):
    """
    Partition a 3D volume among multiple processes for distributed training.

    Uses a simple spatial partitioning strategy that splits the volume along
    x, then y, then z dimensions as needed to create equal partitions.

    Args:
        data_shape: tuple of (x_len, y_len, z_len) - dimensions of the volume
        rank: current rank (0 to world_size-1)
        world_size: total number of ranks (preferably power of 2)

    Returns:
        tuple: (top_corner, bottom_corner) where:
            - top_corner: [x_min, y_min, z_min] of the partition
            - bottom_corner: [x_max, y_max, z_max] of the partition
    """
    x_len, y_len, z_len = data_shape

    # Determine grid dimensions based on world_size
    # For common configurations:
    # - 2 ranks: split x dimension only (2x1x1 grid)
    # - 4 ranks: split x and y dimensions (2x2x1 grid)
    # - 8 ranks: split all three dimensions (2x2x2 grid)
    # - Higher: progressively split dimensions

    # Calculate grid dimensions
    if world_size == 1:
        grid_x, grid_y, grid_z = 1, 1, 1
    elif world_size == 2:
        grid_x, grid_y, grid_z = 2, 1, 1
    elif world_size == 4:
        grid_x, grid_y, grid_z = 2, 2, 1
    elif world_size == 8:
        grid_x, grid_y, grid_z = 2, 2, 2
    elif world_size == 16:
        grid_x, grid_y, grid_z = 4, 2, 2
    elif world_size == 32:
        grid_x, grid_y, grid_z = 4, 4, 2
    elif world_size == 64:
        grid_x, grid_y, grid_z = 4, 4, 4
    else:
        # General case: factor world_size as evenly as possible
        grid_x = 2 ** math.ceil(math.log2(world_size) / 3)
        grid_y = (
            2 ** math.ceil(math.log2(world_size // grid_x) / 2)
            if world_size > grid_x
            else 1
        )
        grid_z = world_size // (grid_x * grid_y)

    # Calculate 3D grid indices for this rank
    idx_x = rank % grid_x
    idx_y = (rank // grid_x) % grid_y
    idx_z = rank // (grid_x * grid_y)

    # Calculate partition boundaries
    size_x = x_len // grid_x
    size_y = y_len // grid_y
    size_z = z_len // grid_z

    x_start = idx_x * size_x
    y_start = idx_y * size_y
    z_start = idx_z * size_z

    # Give remainder pixels to the last partition in each dimension
    x_end = x_start + size_x if idx_x < grid_x - 1 else x_len
    y_end = y_start + size_y if idx_y < grid_y - 1 else y_len
    z_end = z_start + size_z if idx_z < grid_z - 1 else z_len

    top_corner = [x_start, y_start, z_start]
    bottom_corner = [x_end, y_end, z_end]

    return top_corner, bottom_corner


def parse_filename(data_path: str | Path):
    """
    Parse dataset information from filename convention.

    Expected format: datasetname_XxYxZ_dtype.raw
    Example: richtmyer_meshkov_2048x2048x1920_uint8.raw

    Args:
        data_path: Path to the data file

    Returns:
        tuple: (dataset_name, data_shape, data_type) where:
            - dataset_name: Name of the dataset
            - data_shape: Tuple of (x, y, z) dimensions
            - data_type: NumPy dtype string (e.g., 'uint8', 'float32')
    """
    # Extract stem and split by underscores
    # richtmyer_meshkov_2048x2048x1920_uint8 -> richtmyer_meshkov, (2048, 2048, 1920), uint8
    dataset_info = Path(data_path).stem.split("_")

    # Everything except last 2 parts is the dataset name
    data_name = "_".join(dataset_info[:-2])

    # Parse dimensions from second-to-last part
    data_shape = tuple(map(int, dataset_info[-2].split("x")))

    # Data type is the last part
    data_type = dataset_info[-1]  # e.g., uint8, uint16

    return data_name, data_shape, data_type


def parse_run_params(cfg: BenchmarkConfig) -> List[RunParams]:
    """
    Parse parameter sweep configurations from YAML file.

    Generates all combinations of parameters specified in the params file,
    including parameter sweeps over hashmap sizes and network types.

    Args:
        cfg: Benchmark configuration containing params file path

    Returns:
        List[RunParams]: List of all run configurations to benchmark

    Raises:
        ValueError: If no valid runs are found with the specified filters
    """
    home_dir = Path(cfg.home_dir)
    params_dir = home_dir / cfg.params_path
    filename = cfg.params_file
    if not filename.endswith(".json"):
        filename += ".json"
    params_file = OmegaConf.load(params_dir / filename)
    runs: List[RunParams] = []

    # Iterate through each dataset in the params file
    for dataset_name, params in params_file.items():
        for param in params:
            # Parameter sweeps
            hashmap_sizes = param["log2_hashmap_size"]
            log2_hashmap_size_bounds = (
                [hashmap_sizes] if isinstance(hashmap_sizes, int) else hashmap_sizes
            )

            n_neurons = param["n_neurons"]
            n_neurons_bounds = [n_neurons] if isinstance(n_neurons, int) else n_neurons

            n_hidden_layers = param["n_hidden_layers"]
            n_hidden_layers_bounds = (
                [n_hidden_layers]
                if isinstance(n_hidden_layers, int)
                else n_hidden_layers
            )

            n_levels = param["n_levels"]
            n_levels_bounds = [n_levels] if isinstance(n_levels, int) else n_levels

            n_features_per_level = param["n_features_per_level"]
            n_features_per_level_bounds = (
                [n_features_per_level]
                if isinstance(n_features_per_level, int)
                else n_features_per_level
            )

            grid_radius = KANParams.grid_radius
            grid_radius_step = KANParams.grid_radius_step
            num_grids = KANParams.num_grids
            num_grids_step = KANParams.num_grids_step
            if "kan_params" in param:
                kan_params = param["kan_params"]
                if "grid_radius" in kan_params:
                    grid_radius = kan_params.grid_radius
                if "grid_radius_step" in kan_params:
                    grid_radius_step = kan_params.grid_radius_step
                if "num_grids" in kan_params:
                    num_grids = kan_params.num_grids
                if "num_grids_step" in kan_params:
                    num_grids_step = kan_params.num_grids_step

            if isinstance(grid_radius, float) or isinstance(grid_radius, int):
                grid_radius_bounds = [grid_radius]
            else:
                grid_radius_bounds = grid_radius

            num_grids_bounds = [num_grids] if isinstance(num_grids, int) else num_grids

            # Handle dynamic base resolution calculation
            dependent_base_resolution = False
            if param["base_resolution"] == "(int)cbrt(1<<log2_hashmap_size)":
                # Compute base resolution from hashmap size
                # This creates a resolution that scales with the hash table
                dependent_base_resolution = True
            elif not isinstance(param["base_resolution"], int):
                raise ValueError(
                    "base_resolution must be an int or (int)cbrt(1<<log2_hashmap_size)"
                )

            log2_hashmap_size_step = param.get("log2_hashmap_size_step", 1)

            # Generate runs for all combinations
            for log2_hashmap_size in range(
                log2_hashmap_size_bounds[0],
                log2_hashmap_size_bounds[-1] + log2_hashmap_size_step,
                log2_hashmap_size_step,
            ):
                if dependent_base_resolution:
                    # Calculate cube root of 2^size for base resolution
                    base_resolution = int((1 << log2_hashmap_size) ** (1 / 3))
                else:
                    base_resolution = param["base_resolution"]

                for n_neurons in range(n_neurons_bounds[0], n_neurons_bounds[-1] + 1):
                    for n_hidden_layers in range(
                        n_hidden_layers_bounds[0], n_hidden_layers_bounds[-1] + 1
                    ):
                        for n_levels in range(
                            n_levels_bounds[0], n_levels_bounds[-1] + 1
                        ):
                            for n_features_per_level in range(
                                n_features_per_level_bounds[0],
                                n_features_per_level_bounds[-1] + 1,
                            ):
                                for network_type in cfg.network_types:
                                    partial_run_params = partial(
                                        RunParams,
                                        dataset_name=dataset_name,
                                        network_type=network_type,
                                        lrate=param["lrate"],
                                        lrate_decay=param["lrate_decay"],
                                        epochs=param["epochs"],
                                        n_neurons=n_neurons,
                                        n_hidden_layers=n_hidden_layers,
                                        n_levels=n_levels,
                                        n_features_per_level=n_features_per_level,
                                        per_level_scale=param["per_level_scale"],
                                        base_resolution=base_resolution,
                                        log2_hashmap_size=log2_hashmap_size,
                                        zfp_enc=param["zfp_enc"],
                                        zfp_mlp=param["zfp_mlp"],
                                    )
                                    if "kan" not in network_type.lower():
                                        run_params = partial_run_params(kan_params=None)
                                        runs.append(run_params)
                                        continue

                                    # KAN parameter sweeps
                                    for grid_radius in np.arange(
                                        grid_radius_bounds[0],
                                        grid_radius_bounds[-1] + grid_radius_step,
                                        grid_radius_step,  # Use arange for float steps
                                    ):
                                        for num_grids in range(
                                            num_grids_bounds[0],
                                            num_grids_bounds[-1] + num_grids_step,
                                            num_grids_step,
                                        ):
                                            kan_params = KANParams(
                                                grid_radius=float(grid_radius),
                                                grid_radius_step=float(
                                                    grid_radius_step
                                                ),
                                                num_grids=int(num_grids),
                                                num_grids_step=int(num_grids_step),
                                            )
                                            run_params = partial_run_params(
                                                kan_params=kan_params
                                            )
                                            runs.append(run_params)

    # Apply filters if specified
    if cfg.dataset is not None:
        runs = [r for r in runs if r.dataset_name == cfg.dataset]
    if cfg.hashmap_size is not None:
        runs = [r for r in runs if r.log2_hashmap_size == cfg.hashmap_size]

    # Ensure we have valid runs
    if len(runs) == 0:
        raise ValueError(
            f"No runs found with specified restrictions: "
            f"dataset={cfg.dataset}, hashmap_size={cfg.hashmap_size}"
        )

    return runs


def setup_ddp():
    """
    Initialize the distributed training environment for multi-GPU training.

    Sets up NCCL backend for GPU communication and configures the process group
    based on environment variables set by torchrun or mpirun.

    Returns:
        tuple: (rank, world_size, local_rank) where:
            - rank: Global rank of this process across all nodes
            - world_size: Total number of processes
            - local_rank: Local rank on this node (for GPU assignment)
    """
    # Get rank and world size from environment variables (set by torchrun or mpirun)
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Set device for this process
    torch.cuda.set_device(local_rank)

    # Initialize the process group if not already initialized
    if not dist.is_initialized():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    return rank, world_size, local_rank


def cleanup_ddp():
    """
    Clean up the distributed training environment.

    Properly destroys the process group to free resources and ensure
    clean shutdown of distributed training.
    """
    if dist.is_initialized():
        dist.destroy_process_group()


# Register configuration schema with Hydra at import time
cs = ConfigStore.instance()
cs.store(name="benchmark_schema", node=BenchmarkConfig)


@hydra.main(version_base="1.3", config_path="conf")
def main(cfg: BenchmarkConfig):
    """
    Main entry point for the benchmark script.

    Coordinates the entire benchmarking process including:
    1. Parsing run configurations from params file
    2. Setting up distributed training if available
    3. Selecting appropriate run based on job array index
    4. Training and evaluating the model
    5. Saving results and optionally the trained model

    Args:
        cfg: Hydra-managed configuration object
    """
    # Check if we should use DDP (multi-GPU training)
    use_ddp = (
        torch.cuda.is_available()
        and torch.cuda.device_count() > 1
        and "RANK" in os.environ
    )

    if use_ddp and cfg.test_epoch_run:
        raise ValueError("Cannot use DDP with test_epoch_run=True")

    if use_ddp:
        rank, world_size, local_rank = setup_ddp()
    else:
        rank, world_size, local_rank = 0, 1, 0

    is_main_process = rank == 0

    # Only print configuration on main process to avoid duplicate output
    if is_main_process:
        print("Benchmark Configuration:\n" + OmegaConf.to_yaml(cfg))
        if use_ddp:
            print(f"Using DDP with {world_size} GPUs")

    # Parse all possible run configurations from the params file
    runs_list = parse_run_params(cfg)
    # Select the run parameters based on the job array index (for HPC job arrays)
    job_array_idx = int(os.environ.get("PBS_ARRAY_INDEX", 0))
    if is_main_process:
        print(f"Running job array index: {job_array_idx}")
    params = runs_list[job_array_idx]

    # Override epochs if specified in config
    if cfg.epochs is not None:
        params.epochs = cfg.epochs  # Override epochs if specified in config

    # Find the dataset path based on the dataset name (from the params file keys)
    home_dir = Path(cfg.home_dir)
    data_dir = home_dir / cfg.data_path
    data_path = None

    if not data_dir.exists():
        raise FileNotFoundError(f"Data path {data_dir} does not exist.")

    # Search for dataset file matching the dataset name
    for dataset in os.listdir(data_dir):
        if dataset.startswith(params.dataset_name):
            data_path = data_dir / dataset
            break
    if data_path is None:
        raise FileNotFoundError(
            f"Dataset {params.dataset_name} not found in {cfg.data_path}"
        )

    # If a ssd_dir is specified, copy data to the SSD for faster I/O
    if is_main_process and cfg.ssd_dir is not None:
        ssd_data_path = Path(cfg.ssd_dir) / data_path.name
        shutil.copy(data_path, ssd_data_path)
        data_path = ssd_data_path

    # Check if we should save the model based on the save_mode
    should_save = False
    if cfg.save_mode is not None:

        # Find the target hashmap size based on save mode
        if cfg.save_mode == "largest":
            specific_hashmap_size = max(
                run.log2_hashmap_size
                for run in runs_list
                if run.dataset_name == params.dataset_name
            )
        elif cfg.save_mode == "smallest":
            specific_hashmap_size = min(
                run.log2_hashmap_size
                for run in runs_list
                if run.dataset_name == params.dataset_name
            )
        elif cfg.save_mode.isdigit():
            specific_hashmap_size = int(cfg.save_mode)
        else:
            raise ValueError(
                f"Invalid save_mode: {cfg.save_mode}. Must be 'largest', 'smallest', or an integer."
            )

        should_save = params.log2_hashmap_size == specific_hashmap_size

    if is_main_process:
        start_time = time()
        print(f"Running w/ parameters:")
        pprint(params)
        print("Saving INR:", should_save)

    try:
        # Run the actual benchmark
        run_benchmark(
            data_path,
            params,
            cfg,
            should_save,
            rank,
            world_size,
            local_rank,
        )
    finally:
        # Ensure cleanup happens even if errors occur
        cleanup_ddp()
        end_time = time()
        if is_main_process:
            print(f"Total time: {(end_time - start_time) / 60:.2f} minutes")


if __name__ == "__main__":
    main()
