from dataclasses import dataclass
import gc
from typing import List, Optional
from pathlib import Path
import hydra
from hydra.core.config_store import ConfigStore
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
from time import time

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torchmetrics.functional import mean_squared_error as mse_metric
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr_metric
from torchmetrics.functional.image import (
    structural_similarity_index_measure as ssim_metric,
)

import os
import shutil

from networks import INR_Base
from volumetric_dataset import VolumetricDataset

from tempfile import TemporaryDirectory
from pprint import pprint

# DDP imports
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

torch.manual_seed(0)  # for reproducibility
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)


@dataclass
class BenchmarkConfig:
    params_file: str
    model_types: List[str]
    home_dir: str
    data_path: str
    batch_size: Optional[int] = None  # If None, is calculated
    output_filename: Optional[str] = None
    enable_pbar: bool = True
    shuffle: bool = True
    num_workers: int = 0
    pin_memory: bool = True
    run_dtype: str = "float32"  # Data type for all computations
    loss_fn: str = "MSE"
    repeats: int = 1  # Number of times to repeat each run for averaging results
    prefetch_factor: int = 2  # Prefetch factor for DataLoader workers
    count_configs: bool = False  # If True, only count configurations to run
    save_mode: Optional[str] = (
        None  # Save Modes: None = don't save, "largest" = save largest
    )
    checkpoint_freq: Optional[int] = None  # if None, won't checkpoint
    checkpoint_save: bool = (
        False  # If True, save model and reconstruction on checkpoint
    )
    checkpoint_eval: bool = False  # If True, evaluate model on checkpoint
    # hashtable size, "smallest" = save smallest hashtable size
    safety_margin: float = 0.70
    dataset: Optional[str] = None  # Filter to only include this dataset
    hashmap_size: Optional[int] = None  # Filter to only include this hashmap size
    epochs: Optional[int] = None  # Override the number of epochs for all runs
    ssd_dir: Optional[str] = (
        None  # Directory for SSD storage, will first copy data here if provided
    )


@dataclass
class RunParams:
    dataset_name: str  # e.g., "richtmyer_meshkov"
    model_type: str
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


# Register once at import time
cs = ConfigStore.instance()
cs.store(name="benchmark_schema", node=BenchmarkConfig)


def setup_ddp():
    """Initialize the distributed environment."""
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
    """Clean up the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


@hydra.main(version_base="1.3", config_path="conf")
def main(cfg: BenchmarkConfig):
    runs_list = parse_run_params(cfg)

    # Filter for specific dataset or specific hashmap size
    if cfg.dataset is not None:
        runs_list = [r for r in runs_list if r.dataset_name == cfg.dataset]
    if cfg.hashmap_size is not None:
        runs_list = [r for r in runs_list if r.log2_hashmap_size == cfg.hashmap_size]
    if len(runs_list) == 0:
        raise ValueError(
            f"No runs found with specified restrictions: "
            f"{cfg.dataset}, {cfg.hashmap_size}"
        )

    if cfg.count_configs:
        print(len(runs_list))
        return

    if (
        cfg.checkpoint_freq is not None
        and not cfg.checkpoint_save
        and not cfg.checkpoint_eval
    ):
        raise ValueError(
            "Checkpointing is enabled but neither save nor eval set. So it's not doing anything."
        )

    # Check if we should use DDP
    use_ddp = (
        torch.cuda.is_available()
        and torch.cuda.device_count() > 1
        and "RANK" in os.environ
    )

    if use_ddp:
        rank, world_size, local_rank = setup_ddp()
    else:
        rank, world_size, local_rank = 0, 1, 0

    if rank == 0:
        print("Benchmark Configuration:\n" + OmegaConf.to_yaml(cfg))
        if use_ddp:
            print(f"Using DDP with {world_size} GPUs")

    # Select the run parameters based on the job array index
    job_array_idx = int(os.environ.get("PBS_ARRAY_INDEX", 0))
    params = runs_list[job_array_idx]

    if cfg.epochs is not None:
        params.epochs = cfg.epochs  # Override epochs if specified in config

    # Find the dataset path based on the dataset name (from the params file keys)
    home_dir = Path(cfg.home_dir)
    data_dir = home_dir / cfg.data_path
    data_path = None
    if not data_dir.exists():
        raise FileNotFoundError(f"Data path {data_dir} does not exist.")
    for dataset in os.listdir(data_dir):
        if dataset.startswith(params.dataset_name):
            data_path = data_dir / dataset
            break
    if data_path is None:
        raise FileNotFoundError(
            f"Dataset {params.dataset_name} not found in {cfg.data_path}"
        )

    # If a ssd_dir is specified, copy data to the SSD
    if rank == 0 and cfg.ssd_dir is not None:
        ssd_data_path = Path(cfg.ssd_dir) / data_path.name
        shutil.copy(data_path, ssd_data_path)
        data_path = ssd_data_path

    output_dir = home_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    if cfg.output_filename is None:
        # Create a unique filename based on the parameter sweeping
        unique_filename = (
            "_".join(
                [params.dataset_name, params.model_type, str(params.log2_hashmap_size)]
            )
            + "_results.csv"
        )
        output_path = output_dir / unique_filename
    else:
        output_path = output_dir / cfg.output_filename

    # Only write header on rank 0
    if rank == 0:
        with open(output_path, "w") as f:
            f.write("model,dataset,log2_hashmap_size,compression_ratio,psnr,ssim,mse\n")

    # Check if we should save the model based on the save_mode
    should_save = False
    if cfg.save_mode is not None:
        if cfg.save_mode != "largest" and cfg.save_mode != "smallest":
            raise ValueError(
                f"Invalid save_mode: {cfg.save_mode}. Must be 'largest' or 'smallest'."
            )
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
        should_save = params.log2_hashmap_size == specific_hashmap_size

    if rank == 0:
        start_time = time()
        print(f"Running w/ parameters:")
        pprint(params)
        print("Saving INR:", should_save)

    try:
        run_benchmark(
            data_path,
            output_path,
            params,
            cfg,
            should_save,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
        )
    finally:
        cleanup_ddp()
        end_time = time()
        if rank == 0:
            print(f"Total time: {end_time - start_time:.2f} seconds")


def run_benchmark(
    data_path: Path,
    output_path: Path,
    params: RunParams,
    cfg: BenchmarkConfig,
    should_save: bool = False,
    rank: int = 0,
    world_size: int = 1,
    local_rank: int = 0,
):
    # Check if we're in DDP mode
    is_ddp = dist.is_initialized()
    is_main_process = rank == 0

    # Set device and dtype
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(local_rank)
    else:
        device = torch.device("cpu")
    run_dtype = getattr(torch, cfg.run_dtype)

    # Create dataset
    dataset_info = data_path.stem.split("_")
    # richtmyer_meshkov_2048x2048x1920_uint8 -> richtmyer_meshkov, (2048, 2048, 1920), np.uint8
    data_name = "_".join(dataset_info[:-2])
    data_shape = tuple(map(int, dataset_info[-2].split("x")))
    data_dtype = np.dtype(dataset_info[-1])

    inr_name = "_".join(  # made up of parameters being swept
        [
            params.dataset_name,
            params.model_type,
            str(params.log2_hashmap_size),
        ]
    )

    # use native encoder (i.e., don't use TCNN) for KAN
    native_encoder = device.type == "cpu"
    native_network = device.type == "cpu" or params.model_type != "mlp"

    cum_psnr, cum_ssim, cum_mse = 0.0, 0.0, 0.0
    num_repeats = cfg.repeats
    for repeat in range(num_repeats):
        if is_main_process:
            print(f"\nRunning repeat {repeat + 1}/{num_repeats}")

        # Create model and optimizer stuff
        model = INR_Base(
            n_input_dims=3,
            n_output_dims=1,
            native_encoder=native_encoder,
            native_network=native_network,
            network_type=params.model_type,
            n_hidden_layers=params.n_hidden_layers,
            n_neurons=params.n_neurons,
            n_levels=params.n_levels,
            n_features_per_level=params.n_features_per_level,
            log2_hashmap_size=params.log2_hashmap_size,
            base_resolution=params.base_resolution,
            per_level_scale=params.per_level_scale,
        )
        model.to(device, dtype=run_dtype, non_blocking=True)

        # Wrap model with DDP if in distributed mode
        if is_ddp:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)

        optimizer = AdamW(model.parameters(), lr=params.lrate)
        scheduler = StepLR(optimizer, step_size=params.lrate_decay, gamma=0.1)

        if cfg.batch_size is None:
            # Compute batch size based on dataset and model
            # Use the underlying model for batch size calculation
            base_model = model.module if is_ddp else model
            training_batch_size = calculate_batch_size(
                base_model,
                device,
                data_shape,
                run_dtype,
                is_training=True,  # Training mode
                optimizer=optimizer,
                safety_margin=cfg.safety_margin,
            )
            if is_main_process:
                print(
                    f"Calculated batch size: {training_batch_size:,} per GPU with "
                    f"safety margin {cfg.safety_margin} for training"
                )
        else:
            training_batch_size = cfg.batch_size

        dataset = VolumetricDataset(
            file_path=data_path,
            data_shape=data_shape,
            data_type=data_dtype,
            batch_size=training_batch_size,
            normalize_coords=True,
            normalize_values=True,
            initial_shuffle=cfg.shuffle,
        )

        if cfg.loss_fn != "MSE":
            raise ValueError(f"Unsupported loss function: {cfg.loss_fn}")
        loss_fn = nn.MSELoss()
        pin = device.type == "cuda" and cfg.pin_memory

        # For DDP with IterableDataset, adjust workers
        num_workers = cfg.num_workers
        if is_ddp and num_workers > 0:
            # Ensure at least 1 worker per process
            num_workers = max(1, num_workers // world_size)

        dataloader = DataLoader(
            dataset,
            batch_size=None,  # Use IterableDataset's batch size
            num_workers=num_workers,
            pin_memory=pin,
            prefetch_factor=None if num_workers == 0 else cfg.prefetch_factor,
            persistent_workers=num_workers > 0,
        )

        if is_main_process:
            print("Training INR...")
        model.train()
        for epoch in range(params.epochs):
            total_batch_loss = 0
            num_batches = 0

            for x, y_hat in tqdm(
                dataloader, disable=not cfg.enable_pbar or not is_main_process
            ):
                x = x.to(device, dtype=run_dtype, non_blocking=True)
                y_hat = y_hat.to(device, dtype=run_dtype, non_blocking=True)
                y = model(x).to(dtype=run_dtype, non_blocking=True)
                loss = loss_fn(y.squeeze(), y_hat)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_batch_loss += loss.item()
                num_batches += 1

            scheduler.step()
            if num_batches > 0:
                avg_loss = total_batch_loss / num_batches
                if is_main_process:
                    print(f"(epoch {epoch + 1}): avg_loss = {avg_loss}")

            # Save checkpoint every cfg.checkpoint_freq epochs
            if (
                is_main_process
                and cfg.checkpoint_freq is not None
                and (epoch + 1) % cfg.checkpoint_freq == 0
                and epoch + 1 < params.epochs
            ):
                reconst_data, reconst_dataset = reconstruct(
                    cfg, model.module if is_ddp else model, dataset, device, run_dtype
                )
                if cfg.checkpoint_eval:
                    psnr, ssim, mse = compute_metrics(
                        cfg,
                        reconst_data,
                        reconst_dataset,
                        device,
                        run_dtype,
                    )
                    print(
                        f"Checkpoint {epoch + 1}: PSNR = {psnr}, SSIM = {ssim}, MSE = {mse}"
                    )
                if cfg.checkpoint_save:
                    epoch_name = inr_name + f"_epoch_{epoch + 1}"
                    save(
                        cfg,
                        model.module if is_ddp else model,
                        reconst_dataset,
                        reconst_data,
                        epoch_name,
                    )

            # Synchronize at end of epoch
            if is_ddp:
                dist.barrier()

        # Reconstruct and compute metrics when finished training
        if is_main_process:
            print("Reconstructing INR volume...")
            reconst_data, reconst_dataset = reconstruct(
                cfg,
                model.module if is_ddp else model,
                dataset,
                device,
                run_dtype,
                enable_pbar=cfg.enable_pbar,
            )

            print("Computing metrics...")
            psnr, ssim, mse = compute_metrics(
                cfg,
                reconst_data,
                reconst_dataset,
                device,
                run_dtype,
            )

            cum_psnr += psnr
            cum_ssim += ssim
            cum_mse += mse

            print(
                f"Repeat {repeat + 1}/{num_repeats}: PSNR = {psnr}, SSIM = {ssim}, MSE = {mse}"
            )
            if repeat + 1 < num_repeats:
                del reconst_data, reconst_dataset

        # Clean up memory after all repeats except last
        if repeat + 1 < num_repeats:
            del dataset, model, optimizer, scheduler, dataloader, loss_fn
            gc.collect()
            torch.cuda.empty_cache()

        # Synchronize all processes before next repeat
        if is_ddp:
            dist.barrier()

    # Only compute and save results on main process
    if is_main_process:
        avg_psnr = cum_psnr / num_repeats
        avg_ssim = cum_ssim / num_repeats
        avg_mse = cum_mse / num_repeats
        print(f"Averages: PSNR = {avg_psnr}, SSIM = {avg_ssim}, MSE = {avg_mse}")

        # Compute compression ratio
        with TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            name = hash(model) % (10**8)
            inr_path = tmpdir_path / f"{name}.pt"
            torch.save(model.state_dict(), inr_path)
            model_size = inr_path.stat().st_size

        volume_size = (
            reconst_dataset.data.nbytes
            if "reconst_dataset" in locals()
            else dataset.data.nbytes
        )
        compression_ratio = volume_size / model_size
        print(f"Compression ratio: {compression_ratio:.2f}")

        with open(output_path, "a") as f:
            f.write(
                f"{params.model_type},{params.dataset_name},{params.log2_hashmap_size},"
                f"{compression_ratio},{avg_psnr},{avg_ssim},{avg_mse}\n"
            )

        if should_save:
            save(
                cfg,
                model.module if is_ddp else model,
                reconst_dataset,
                reconst_data,
                inr_name,
            )


@torch.no_grad()
def reconstruct(
    cfg: BenchmarkConfig,
    model: INR_Base,
    dataset: VolumetricDataset,
    device: torch.device,
    run_dtype: torch.dtype,
    enable_pbar: bool = False,
):
    if cfg.batch_size is None:
        eval_batch_size = calculate_batch_size(
            model,
            device,
            dataset.data_shape,
            run_dtype,
            is_training=False,  # Evaluation mode
            safety_margin=cfg.safety_margin,
        )
        print(
            f"Calculated batch size: {eval_batch_size:,} per GPU with "
            f"safety margin {cfg.safety_margin} for evaluation"
        )
    else:
        eval_batch_size = cfg.batch_size

    reconst_data = torch.zeros(dataset.data_shape, device=device, dtype=run_dtype)
    model.eval()

    data_path = dataset.file_path
    data_shape = dataset.data_shape
    data_dtype = dataset.data_type

    # Create new dataset for reconstruction (without shuffling to get all data)
    reconst_dataset = VolumetricDataset(
        file_path=data_path,
        data_shape=data_shape,
        data_type=data_dtype,
        batch_size=eval_batch_size,
        normalize_coords=True,
        normalize_values=True,
        initial_shuffle=False,  # No shuffle for reconstruction
    )

    pin = device.type == "cuda" and cfg.pin_memory
    eval_dataloader = DataLoader(
        reconst_dataset,
        batch_size=None,
        num_workers=0,  # Single process for eval
        pin_memory=pin,
    )

    data_shape_tensor = torch.as_tensor(
        reconst_dataset.data_shape, device=device, dtype=run_dtype
    )
    for x, _ in tqdm(eval_dataloader, disable=not enable_pbar or not cfg.enable_pbar):
        x = x.to(device, dtype=run_dtype, non_blocking=True)
        y = model(x).to(dtype=run_dtype, non_blocking=True)

        indices = (x * (data_shape_tensor - 1)).long()
        i, j, k = indices.split(1, dim=-1)

        # (batch_size,)
        i, j, k = i.squeeze(), j.squeeze(), k.squeeze()
        y = y.squeeze()

        reconst_data[i, j, k] = y

    return reconst_data, reconst_dataset


@torch.no_grad()
def compute_metrics(
    cfg: BenchmarkConfig,
    reconst_data: torch.Tensor,
    reconst_dataset: VolumetricDataset,
    device: torch.device,
    run_dtype: torch.dtype,
):
    # Get the data
    gt_data = torch.as_tensor(
        reconst_dataset.volume_data(), device=device, dtype=run_dtype
    ).contiguous()
    reconst_data = torch.clamp(reconst_data, 0.0, 1.0).contiguous()

    # Process metrics slice-by-slice (more stable for 3D volumes)
    psnr_values = []
    mse_values = []
    ssim_values = []

    # Iterate through one dimension
    num_slices = gt_data.shape[2]
    for i in tqdm(range(num_slices), disable=not cfg.enable_pbar):

        # (1, 1, H, W)
        gt_slice = gt_data[:, :, i].unsqueeze(0).unsqueeze(0)
        reconst_slice = reconst_data[:, :, i].unsqueeze(0).unsqueeze(0)

        # Calculate metrics for this slice
        psnr_values.append(psnr_metric(reconst_slice, gt_slice, data_range=1.0))
        mse_values.append(mse_metric(reconst_slice, gt_slice))
        ssim_values.append(ssim_metric(reconst_slice, gt_slice, data_range=1.0))

    # Average the metrics across all slices
    psnr = torch.stack(psnr_values).mean().item()
    mse = torch.stack(mse_values).mean().item()
    ssim = torch.stack(ssim_values).mean().item()

    return psnr, mse, ssim


@torch.no_grad()
def save(
    cfg: BenchmarkConfig,
    model: INR_Base,
    reconst_dataset: VolumetricDataset,
    reconst_data: torch.Tensor,
    inr_name: str,
):
    home_dir = Path(cfg.home_dir)
    # Save the INR
    save_path = home_dir / "saved_models" / f"{inr_name}.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # Save the reconstruction as .raw file
    save_order = reconst_dataset.order
    save_type = reconst_dataset.data_type
    save_shape = reconst_dataset.data_shape
    shape_str = "x".join(map(str, save_shape))
    type_str = np.dtype(save_type).name
    reconst_path = (
        home_dir / "saved_reconstructions" / f"{inr_name}_{shape_str}_{type_str}.raw"
    )
    reconst_path.parent.mkdir(parents=True, exist_ok=True)

    reconst_data = torch.clamp(reconst_data, 0.0, 1.0)

    # restore data range to original
    reconst_data = reconst_dataset.unnormalize(reconst_data)

    reconst_bytes = (
        reconst_data.flatten()
        .contiguous()
        .cpu()
        .numpy()
        .astype(save_type)
        .tobytes(save_order)
    )
    # write as .raw file
    with open(reconst_path, "wb") as f:
        f.write(reconst_bytes)
    print(f"Reconstruction saved to {reconst_path}")


def parse_run_params(cfg: BenchmarkConfig) -> List[RunParams]:
    home_dir = Path(cfg.home_dir)
    params_file = OmegaConf.load(home_dir / cfg.params_file)
    runs = []
    for dataset_name, params in params_file.items():
        for param in params:
            hashmap_sizes = param["log2_hashmap_size"]
            size_sweep = (
                [hashmap_sizes] if isinstance(hashmap_sizes, int) else hashmap_sizes
            )

            dependent_base_resolution = False
            if param["base_resolution"] == "(int)cbrt(1<<log2_hashmap_size)":
                # compute base resolution from hashmap size
                dependent_base_resolution = True
            elif not isinstance(param["base_resolution"], int):
                raise ValueError(
                    "base_resolution must be an int or (int)cbrt(1<<log2_hashmap_size)"
                )

            for size in range(size_sweep[0], size_sweep[-1] + 1):
                if dependent_base_resolution:
                    base_resolution = int((1 << size) ** (1 / 3))
                else:
                    base_resolution = param["base_resolution"]

                for model_type in cfg.model_types:
                    run_params = RunParams(
                        dataset_name=dataset_name,
                        model_type=model_type,
                        lrate=param["lrate"],
                        lrate_decay=param["lrate_decay"],
                        epochs=param["epochs"],
                        n_neurons=param["n_neurons"],
                        n_hidden_layers=param["n_hidden_layers"],
                        n_levels=param["n_levels"],
                        n_features_per_level=param["n_features_per_level"],
                        per_level_scale=param["per_level_scale"],
                        base_resolution=base_resolution,
                        log2_hashmap_size=size,
                        zfp_enc=param["zfp_enc"],
                        zfp_mlp=param["zfp_mlp"],
                    )
                    runs.append(run_params)

    return runs


def calculate_batch_size(
    model,
    device,
    data_shape,
    run_dtype,
    is_training=True,
    optimizer=None,
    min_batch=1,
    max_batch=10_000_000,  # 10 million max to cover your use case
    safety_margin=0.99,
):
    """
    Use binary search to find maximum batch size that fits in memory.
    For INRs with coordinate inputs, batch sizes can be very large (tens of thousands to millions).
    """
    if device.type != "cuda":
        return 100_000  # Reasonable default for CPU

    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # First, do a coarse search to find the right order of magnitude
    test_sizes = [100, 1_000, 10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000]
    working_batch = 1

    for test_batch in test_sizes:
        try:
            torch.cuda.empty_cache()
            test_input = torch.randn(test_batch, 3, device=device, dtype=run_dtype)

            if is_training:
                output = model(test_input)
                loss = output.mean()
                loss.backward()
                model.zero_grad()
            else:
                with torch.no_grad():
                    output = model(test_input)

            working_batch = test_batch
            del test_input, output
            if is_training:
                del loss

        except RuntimeError as e:
            if "out of memory" in str(e) or "CUDA" in str(e):
                torch.cuda.empty_cache()
                break
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
        max_batch = working_batch * 10

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

            # Test this batch size
            test_input = torch.randn(mid_batch, 3, device=device, dtype=run_dtype)

            if is_training:
                output = model(test_input)
                loss = output.mean()
                loss.backward()

                if optimizer:
                    optimizer.step()
                    optimizer.zero_grad()
                else:
                    model.zero_grad()
            else:
                with torch.no_grad():
                    output = model(test_input)

                    # For eval, test reconstruction tensor allocation periodically
                    if mid_batch > best_batch * 1.5:  # Significant increase
                        test_reconst = torch.zeros(
                            data_shape, device=device, dtype=run_dtype
                        )
                        del test_reconst

            # Success! Try larger batch
            best_batch = mid_batch
            min_batch = mid_batch + 1

            # Cleanup
            del test_input, output
            if is_training:
                del loss

        except RuntimeError as e:
            if "out of memory" in str(e) or "CUDA" in str(e):
                # Failed, try smaller batch
                max_batch = mid_batch - 1
                torch.cuda.empty_cache()
            else:
                raise e

    final_batch = max(1, int(best_batch * safety_margin))

    # Quick sanity check - if we're getting a tiny batch size for an INR, something's wrong
    if final_batch < 1000 and device.type == "cuda":
        print(
            f"Warning: Unexpectedly small batch size {final_batch} for INR on GPU. "
            f"Memory may be fragmented or model may be very large."
        )

    return final_batch


if __name__ == "__main__":
    main()
