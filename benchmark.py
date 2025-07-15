from dataclasses import dataclass
from functools import partial
from typing import List, Optional, Tuple, Callable

from pathlib import Path
from xml.parsers.expat import model
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from volumetric_dataset import VolumetricDataset
import torch.nn as nn
from torchmetrics.functional import mean_squared_error
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from einops import rearrange
import os
import torch.multiprocessing as mp

from networks import FKAN_Native, MLP_Native

SAVED_INR_PATH = Path("/grand/insitu/cohanlon/alcf_kan_inr/inrs")

torch.manual_seed(0)  # for reproducibility
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)


@dataclass
class MLPINR_Config:
    n_neurons: int
    n_hidden_layers: int
    in_features: int = 3  # x, y, z coordinates
    out_features: int = 1
    activation: str = "ReLU"
    bias: bool = True


@dataclass
class KAINRConfig:
    n_neurons: int
    n_hidden_layers: int
    in_features: int = 3
    out_features: int = 1


@dataclass
class BenchmarkConfig:
    batch_size: int
    num_epochs: int
    model_types: List[str]
    mlp_params: List[MLPINR_Config]
    kan_params: List[KAINRConfig]
    data_path: str = "/grand/insitu/cohanlon/datasets/raw"
    data_shape: Optional[Tuple[int, int, int]] = None  # inferred from data_path if none
    data_dtype: Optional[str] = None  # inferred from data_path if none
    output_filename: Optional[str] = None
    enable_pbar: bool = True
    shuffle: bool = True
    num_workers: int = 0
    pin_memory: bool = True
    lr: float = 1e-3
    model_dtype: str = "float32"
    loss_fn: str = "MSE"
    save_model: bool = True


# Register once at import time
cs = ConfigStore.instance()
cs.store(name="benchmark_schema", node=BenchmarkConfig)


@hydra.main(version_base="1.3", config_path="conf")
def main(cfg: BenchmarkConfig):
    print("Benchmark Configuration:\n" + OmegaConf.to_yaml(cfg))

    data_path = Path(cfg.data_path)
    if not data_path.exists():
        raise ValueError(f"Data path does not exist: {cfg.data_path}")
    if data_path.is_dir():
        # If data_path is a directory, look for all raw files inside it
        raw_files = list(data_path.glob("*.raw"))
        if not raw_files:
            raise ValueError(f"No .raw files found in directory: {data_path}")
    else:
        # If data_path is a file, use it directly
        raw_files = [data_path]

    print(f"Found files: {[file.name for file in raw_files]}\n")

    if cfg.output_filename:
        with open(cfg.output_filename, "w") as f:
            f.write("model,dataset,compression_ratio,psnr,ssim,mse\n")

    for file_path in raw_files:
        print(f"Running benchmark for dataset: {file_path}")
        run_benchmark(file_path, cfg)


def run_benchmark(data_path: Path, cfg: BenchmarkConfig):
    # Create dataset
    dataset_name = data_path.name.split(".")[0]
    ds_info = dataset_name.split("_")

    data_shape = (
        tuple(map(int, ds_info[-2].split("x")))  # e.g., "41x41x41" -> (41, 41, 41)
        if cfg.data_shape is None
        else cfg.data_shape
    )
    data_dtype = ds_info[-1] if cfg.data_dtype is None else cfg.data_dtype
    dataset = VolumetricDataset(
        file_path=data_path,
        data_shape=data_shape,
        data_type=data_dtype,
        normalize_coords=True,
        normalize_values=True,
    )

    # Create model(s)
    models = []
    for param_set in cfg.mlp_params:
        models.append(
            MLP_Native(
                bias=param_set.bias,
                n_hidden_layers=param_set.n_hidden_layers,
                n_neurons=param_set.n_neurons,
                activation=param_set.activation,
            )
        )
    for param_set in cfg.kan_params:
        models.append(
            FKAN_Native(
                n_hidden_layers=param_set.n_hidden_layers,
                n_neurons=param_set.n_neurons,
                activation="SiLU",
            )
        )

    dtype = getattr(torch, cfg.model_dtype)
    if not torch.cuda.is_available():
        devices = [torch.device("cpu")]
    else:
        num_gpus = torch.cuda.device_count()
        devices = [f"cuda:{i}" for i in range(num_gpus)]

    mp.set_start_method("spawn", force=True)
    q = mp.Queue()
    outfile_lock = mp.Lock() if devices[0] != "cpu" else None

    for model in models:
        q.put(
            partial(
                run,
                model=model,
                dataset=dataset,
                cfg=cfg,
                dtype=dtype,
                dataset_name=dataset_name,
            )
        )

    processes = []
    for device in devices:
        run_fn = q.get()
        p = mp.Process(
            target=run_fn, args=(device,), kwargs={"outfile_lock": outfile_lock}
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


def run(
    device: str,
    model: nn.Module,
    dataset: VolumetricDataset,
    cfg: BenchmarkConfig,
    dtype: torch.dtype,
    dataset_name: str,
    outfile_lock=None,
):
    model_uid = hash(model) % (10**6)
    model_type = model.__class__.__name__
    inr_name = model_type + "_inr" + str(model_uid)

    fit_inr(
        cfg=cfg,
        model=model,
        device=device,
        dtype=dtype,
        dataset=dataset,
        dataset_name=dataset_name,
        inr_name=inr_name,
    )
    evaluate_inr(
        cfg=cfg,
        model=model,
        device=device,
        dtype=dtype,
        dataset=dataset,
        dataset_name=dataset_name,
        inr_name=inr_name,
        outfile_lock=outfile_lock,
    )


def fit_inr(
    cfg: BenchmarkConfig,
    model: nn.Module,
    device: str,
    dtype: torch.dtype,
    dataset: VolumetricDataset,
    dataset_name: str,
    inr_name: str,
):
    print(f"Training {inr_name} on {dataset_name}")
    dataset.return_indices = False
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    model.to(device, dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    if cfg.loss_fn == "MSE":
        loss_fn = nn.functional.mse_loss
    else:
        raise ValueError(f"Unsupported loss function: {cfg.loss_fn}")

    model.train()
    for epoch in range(cfg.num_epochs):
        total_batch_loss = 0

        for x, y_hat in tqdm(loader, disable=not cfg.enable_pbar):
            x = x.to(device, dtype)
            y_hat = y_hat.to(device, dtype)
            y = model(x)
            loss = loss_fn(y, y_hat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_batch_loss += loss.item()

        avg_loss = total_batch_loss / len(loader)
        print(f"(epoch {epoch}): {loss_fn.__name__} = {avg_loss}")

    print(f"Finished training {inr_name} for {cfg.num_epochs} epochs.\n")


def evaluate_inr(
    cfg: BenchmarkConfig,
    model: nn.Module,
    device: str,
    dtype: torch.dtype,
    dataset: VolumetricDataset,
    dataset_name: str,
    inr_name: str,
    outfile_lock=None,
):
    print(f"Evaluating {inr_name}:")
    dataset.return_indices = True  # Ensure indices are returned for reconstruction
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True,  # Ensure only full batches for evaluation
    )
    inr_dir = SAVED_INR_PATH / dataset_name
    inr_dir.mkdir(exist_ok=True)
    inr_path = inr_dir / (str(inr_name) + ".pt")
    torch.save(model.state_dict(), inr_path)
    model_size = os.path.getsize(inr_path)
    dataset_size = dataset.volume_data().nbytes
    compression_ratio = dataset_size / model_size
    if not cfg.save_model:
        os.remove(inr_path)  # Remove model file if not saving

    model.to(device, dtype)
    model.eval()
    with torch.no_grad():
        reconst_data = torch.zeros(dataset.data_shape, device=device, dtype=dtype)
        for x, _, (i, j, k) in tqdm(loader, disable=not cfg.enable_pbar):
            x = x.to(device, dtype)
            y = model(x)
            for y_s, i_s, j_s, k_s in zip(y, i, j, k):
                reconst_data[i_s.item(), j_s.item(), k_s.item()] = y_s.item()

        gt_data = torch.tensor(dataset.volume_data(), device=device, dtype=dtype)
        reconst_data = (
            rearrange(reconst_data, "h w c -> 1 c h w")
            .to(device=device, dtype=dtype, non_blocking=True)
            .contiguous()
        )
        gt_data = rearrange(gt_data, "h w c -> 1 c h w").contiguous()

        print(f"{inr_name} - Evaluation Results:")
        psnr = PeakSignalNoiseRatio().to(device=device, dtype=dtype)
        psnr_value = psnr(reconst_data, gt_data)
        print(f"PSNR: {psnr_value.item()}")
        ssim = StructuralSimilarityIndexMeasure().to(device=device, dtype=dtype)
        ssim_value = ssim(reconst_data, gt_data)
        print(f"SSIM: {ssim_value.item()}")
        mse = mean_squared_error(reconst_data, gt_data)
        print(f"MSE: {mse.item()}")
        print("-" * 40, "\n")
        if cfg.output_filename:
            model_type = model.__class__.__name__
            metric_str = ",".join(
                [
                    model_type,
                    dataset_name,
                    str(compression_ratio),
                    str(psnr_value.item()),
                    str(ssim_value.item()),
                    str(mse.item()),
                ]
            )
            if outfile_lock:
                with outfile_lock:
                    with open(cfg.output_filename, "a") as f:
                        f.write(metric_str + "\n")
            else:
                # If no lock, write directly
                with open(cfg.output_filename, "a") as f:
                    f.write(metric_str + "\n")


if __name__ == "__main__":
    main()
