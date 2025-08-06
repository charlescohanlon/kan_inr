from dataclasses import dataclass
from typing import List, Optional

from pathlib import Path
import hydra
from hydra.core.config_store import ConfigStore
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torchmetrics.functional import (
    mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)

from einops import rearrange
import os

from networks import INR_Base
from volumetric_dataset import VolumetricDataset

from uuid import uuid4
from tempfile import TemporaryDirectory

torch.manual_seed(0)  # for reproducibility
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)


@dataclass
class BenchmarkConfig:
    params_file: str
    batch_size: int
    model_types: List[str]
    home_dir: str
    data_path: str
    output_filename: Optional[str] = None
    enable_pbar: bool = True
    shuffle: bool = True
    num_workers: int = 0
    pin_memory: bool = True
    run_dtype: str = "float32"  # Data type for all computations
    loss_fn: str = "MSE"
    save_model: bool = True
    repeats: int = 1  # Number of times to repeat each run for averaging results
    prefetch_factor: int = 2  # Prefetch factor for DataLoader workers
    only_count_runs: bool = False  # If True, only count runs without executing them


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


@hydra.main(version_base="1.3", config_path="conf")
def main(cfg: BenchmarkConfig):
    runs_list = parse_run_params(cfg)
    if cfg.only_count_runs:
        print(len(runs_list))
        return

    print("Benchmark Configuration:\n" + OmegaConf.to_yaml(cfg))

    # Select the run parameters based on the job array index
    job_array_idx = int(os.environ.get("PBS_ARRAY_INDEX", 0))
    params = runs_list[job_array_idx]

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

    output_dir = home_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    if cfg.output_filename is None:
        # Create a unique filename based on the parameter sweeping
        unique_filename = (
            "_".join([params.dataset_name, params.model_type, str(params.log2_hashmap_size)])
            + "_results.csv"
        )
        output_path = output_dir / unique_filename
    else:
        output_path = output_dir / cfg.output_filename

    with open(output_path, "w") as f:
        f.write("model,dataset,compression_ratio,psnr,ssim,mse\n")

    print(f"Running benchmark for: {params}")
    run_benchmark(data_path, output_path, params, cfg)


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


def run_benchmark(
    data_path: Path, output_path: Path, params: RunParams, cfg: BenchmarkConfig
):
    # Create dataset
    dataset_info = data_path.stem.split("_")
    # richtmyer_meshkov_2048x2048x1920_uint8 -> richtmyer_meshkov, (2048, 2048, 1920), np.uint8
    data_name = "_".join(dataset_info[:-2])
    data_shape = tuple(map(int, dataset_info[-2].split("x")))
    data_dtype = np.dtype(dataset_info[-1])
    dataset = VolumetricDataset(
        file_path=data_path,
        data_shape=data_shape,
        data_type=data_dtype,
        normalize_coords=True,
        normalize_values=True,
        return_coords=False,
        order="F",  # col major order
    )

    # Set device and dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dtype = getattr(torch, cfg.run_dtype)
    native_encoder = device.type == "cpu"
    # use native encoder (i.e., don't use TCNN) for KAN
    native_network = device.type == "cpu" or params.model_type != "mlp"

    cum_psnr, cum_ssim, cum_mse = 0.0, 0.0, 0.0
    num_repeats = cfg.repeats
    for repeat in range(num_repeats):
        print(f"Running repeat {repeat + 1}/{num_repeats} for {params}")

        # Create model and optimizer stuff
        model = INR_Base(
            n_input_dims=3,
            n_output_dims=1,
            native_encoder=native_encoder,
            native_network=native_network,
            network=params.model_type,
            n_hidden_layers=params.n_hidden_layers,
            n_neurons=params.n_neurons,
            n_levels=params.n_levels,
            n_features_per_level=params.n_features_per_level,
            log2_hashmap_size=params.log2_hashmap_size,
            base_resolution=params.base_resolution,
            per_level_scale=params.per_level_scale,
        )
        model.to(device, dtype=run_dtype, non_blocking=True)
        optimizer = AdamW(model.parameters(), lr=params.lrate)
        scheduler = StepLR(optimizer, step_size=params.lrate_decay, gamma=0.1)

        if cfg.loss_fn != "MSE":
            raise ValueError(f"Unsupported loss function: {cfg.loss_fn}")
        loss_fn = nn.MSELoss()
        pin = device.type == "cuda" and cfg.pin_memory
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle,
            num_workers=cfg.num_workers,
            pin_memory=pin,
            prefetch_factor=None if cfg.num_workers == 0 else cfg.prefetch_factor,
            persistent_workers=cfg.num_workers > 0,
        )

        # Fit INR
        model.train()
        for epoch in range(params.epochs):
            total_batch_loss = 0

            for x, y_hat in tqdm(dataloader, disable=not cfg.enable_pbar):
                x = x.to(device, dtype=run_dtype, non_blocking=True)
                y_hat = y_hat.to(device, dtype=run_dtype, non_blocking=True)
                y = model(x)
                loss = loss_fn(y, y_hat)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_batch_loss += loss.item()

            scheduler.step()
            avg_loss = total_batch_loss / len(dataloader)
            print(f"(epoch {epoch}): {loss_fn.__name__} = {avg_loss}")

        # Evaluate INR
        with torch.no_grad():
            model.eval()
            dataloader.dataset.return_coords = False
            reconst_data = torch.zeros(
                dataset.data_shape, device=device, dtype=run_dtype
            )
            for x, _, (i, j, k) in tqdm(dataloader, disable=not cfg.enable_pbar):
                x = x.to(device, dtype=run_dtype, non_blocking=True)
                y = model(x).to(dtype=run_dtype, non_blocking=True)
                reconst_data[i, j, k] = y.squeeze()

            gt_data = torch.tensor(
                dataset.volume_data(), device=device, dtype=run_dtype
            )
            reconst_data = (
                rearrange(reconst_data, "h w c -> 1 c h w")
                .to(device=device, dtype=run_dtype, non_blocking=True)
                .contiguous()
            )
            gt_data = rearrange(gt_data, "h w c -> 1 c h w").contiguous()

            # Calculate metrics on reconstructed (decompressed) data
            psnr = peak_signal_noise_ratio(reconst_data, gt_data)
            ssim = structural_similarity_index_measure(reconst_data, gt_data)
            mse = mean_squared_error(reconst_data, gt_data)

            cum_psnr += psnr.item()
            cum_ssim += ssim.item()
            cum_mse += mse.item()

    avg_psnr = cum_psnr / num_repeats
    avg_ssim = cum_ssim / num_repeats
    avg_mse = cum_mse / num_repeats
    print(
        f"Average PSNR: {avg_psnr}, SSIM: {avg_ssim}, MSE: {avg_mse} "
        f"over {num_repeats} repeats for {params}"
    )

    # Compute compression ratio
    with TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        uid = uuid4()
        inr_name = f"{params.dataset_name}_{params.model_type}_{uid}"
        inr_path = tmpdir_path / f"{inr_name}.pt"
        torch.save(model.state_dict(), inr_path)
        model_size = inr_path.stat().st_size
    volume_size = dataset.data.nbytes
    compression_ratio = volume_size / model_size

    with open(output_path, "a") as f:
        f.write(
            f"{params.model_type},{params.dataset_name},{compression_ratio},"
            f"{avg_psnr},{avg_ssim},{avg_mse}\n"
        )


if __name__ == "__main__":
    main()
