from dataclasses import dataclass
from typing import List, Optional, Tuple, Callable

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from alcf_kan_inr.volumetric_dataset import VolumetricDataset
import torch.nn as nn
import torcheval.metrics.functional as tmf
from einops import rearrange


@dataclass
class MLP_INR_Config:
    in_features: int = 3  # x, y, z coordinates
    out_features: int = 1
    hidden_features: int
    depth: int
    activation: str = "SiLU"
    bias: bool = True


@dataclass
class KAN_INR_Config:
    in_features: int = 3
    out_features: int = 1
    hidden_features: int
    depth: int


@dataclass
class BenchmarkConfig:
    # Required parameters
    data_path: str
    data_shape: Tuple[int, int, int]
    data_dtype: str
    batch_size: int
    num_epochs: int
    model_types: List[str]
    benchmark_metrics: List[str]
    mlp_params: MLP_INR_Config
    kan_params: KAN_INR_Config

    # Default parameters
    output_filename: Optional[str] = None
    verbose: bool = False
    enable_pbar: bool = True
    shuffle: bool = True
    num_workers: int = 0
    pin_memory: bool = True
    lr: float = 1e-3
    device: Optional[str] = None  # None means auto-detect
    model_dtype: str = "float32"
    loss_fn: str = "MSE"
    save_model: bool = True


# Register once at import time
cs = ConfigStore.instance()
cs.store(name="benchmark_schema", node=BenchmarkConfig)


@hydra.main(version_base="1.3", config_path="conf")
def main(cfg: BenchmarkConfig):
    if cfg.verbose:
        print("Benchmark Configuration:\n" + OmegaConf.to_yaml(cfg))

    # Initialize device
    if cfg.device is None:
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(cfg.device)
    dtype = getattr(torch, cfg.model_dtype)

    # Create dataset
    dataset = VolumetricDataset(
        file_path=cfg.data_path,
        data_shape=cfg.data_shape,
        data_type=cfg.data_dtype,
        normalize_coords=True,
        normalize_values=True,
    )

    # Create models
    models = create_models(cfg)

    # Fit the models
    fit_inrs(
        cfg=cfg,
        models=models,
        device=device,
        dtype=dtype,
        dataset=dataset,
    )

    # Evaluate the models
    evaluate_inrs(
        cfg=cfg,
        models=models,
        device=device,
        dtype=dtype,
        dataset=dataset,
    )


def create_models(cfg: BenchmarkConfig) -> List[nn.Module]:
    models = []
    for model_type in cfg.model_types:
        if model_type == "MLP":
            p = cfg.mlp_params
            activation_module = getattr(nn, cfg.mlp_params.activation)
            model = nn.Sequential(
                *[
                    nn.Sequential(
                        nn.Linear(
                            p.in_features,
                            (p.hidden_features if i < p.depth - 1 else p.out_features),
                            bias=p.bias,
                        ),
                        activation_module(),
                    )
                    for i in range(p.depth - 1)
                ]
            )
        elif model_type == "pykan":
            from alcf_kan_inr.pykan.kan.KANLayer import KANLayer

            p = cfg.kan_params
            model = nn.Sequential(
                *[
                    nn.Sequential(
                        KANLayer(
                            in_dim=p.in_features,
                            out_dim=(
                                p.hidden_features if i < p.depth - 1 else p.out_features
                            ),
                        ),
                    )
                    for i in range(p.depth - 1)
                ]
            )
        elif model_type in ["efficient-kan", "e-kan"]:
            from alcf_kan_inr.efficient_kan.src.efficient_kan.kan import KAN

            p = cfg.kan_params
            model = KAN(
                layers_hidden=[p.in_features]
                + [p.hidden_features] * (p.depth - 1)
                + [p.out_features],
            )
        elif model_type in ["fast-kan", "f-kan"]:
            from alcf_kan_inr.fast_kan.fastkan.fastkan import FastKAN

            p = cfg.kan_params
            model = FastKAN(
                layers_hidden=[p.in_features]
                + [p.hidden_features] * (p.depth - 1)
                + [p.out_features],
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        models.append(model)

    return models


def fit_inrs(
    cfg: BenchmarkConfig,
    models: List[nn.Module],
    device: torch.device,
    dtype: torch.dtype,
    dataset: VolumetricDataset,
):
    if cfg.output_filename is not None:  # NOTE: do we care about this?
        with open(cfg.output_filename, "w") as output_file:
            output_file.write("epoch,avg_loss\n")

    loader = DataLoader(dataset)
    for model in models:
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
            if cfg.verbose:
                print(f"(epoch {epoch}): avg loss = {avg_loss:.4f}")

            if cfg.output_filename is not None:
                with open(cfg.output_filename, "a") as output_file:
                    output_file.write(f"{epoch},{avg_loss:.4f}\n")


def evaluate_inrs(
    cfg: BenchmarkConfig,
    models: List[nn.Module],
    device: torch.device,
    dtype: torch.dtype,
    dataset: VolumetricDataset,
):
    metrics = get_metrics(cfg.benchmark_metrics)
    loader = DataLoader(dataset)
    for model in models:
        model_name = model.__class__.__name__ + "_inr.pt"
        if cfg.save_model:
            torch.save(model, model_name)

        model.to(device, dtype)

        model.eval()
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
            print(f"{model_name} - Evaluation Results:")
            for metric in metrics:
                result = metric(reconst_data, gt_data)
                print(f"{metric.__name__}: {result:.4f}")


def get_metrics(metric_names: List[str]) -> List[Callable]:
    metrics = []
    for name in metric_names:
        if name == "PSNR":
            metrics.append(tmf.peak_signal_noise_ratio)
        elif name == "MSE":
            metrics.append(tmf.mean_squared_error)
        else:
            raise ValueError(f"Unsupported metric: {name}")
    return metrics


if __name__ == "__main__":
    main()
