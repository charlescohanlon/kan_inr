from typing import Tuple, List
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class VolumetricDataset(Dataset):
    def __init__(
        self,
        file_path: Path,
        data_shape: Tuple[int, int, int],
        data_type: np.dtype,
        normalize_coords: bool = True,
        normalize_values: bool = True,
        return_coords: bool = False,
        order: str = "F",  # col major order
    ):
        self.data = np.fromfile(file_path, dtype=data_type)
        self.data_range = (np.min(self.data), np.max(self.data))

        if normalize_coords:
            xs = np.linspace(0, 1, data_shape[0])
            ys = np.linspace(0, 1, data_shape[1])
            zs = np.linspace(0, 1, data_shape[2])
        else:
            xs = np.arange(data_shape[0])
            ys = np.arange(data_shape[1])
            zs = np.arange(data_shape[2])
        self.coords = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(
            -1, 3
        )

        self.file_path = file_path
        self.data_shape = data_shape
        self.data_type = data_type
        self.normalize_coords = normalize_coords
        self.normalize_values = normalize_values
        self.return_coords = return_coords
        self.order = order

        if normalize_values:
            min, max = self.data_range
            self.data = (self.data - min) / (max - min)

    def __getitem__(self, idx):
        pair = (self.coords[idx], self.data[idx])
        if self.return_coords:
            if self.normalize_coords:  # denormalize coords
                denorm_coords = self.coords[idx] * np.array(self.data_shape)
                denorm_coords = denorm_coords.astype(np.int64)
                return pair, denorm_coords
        return pair

    def __len__(self):
        return self.data.shape[0]

    def volume_data(self):
        """Return the volume data in the original shape."""
        return self.data.reshape(self.data_shape, order=self.order)
