import math
from typing import Tuple, Optional
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info
from pathlib import Path


class VolumetricDataset(IterableDataset):
    def __init__(
        self,
        file_path: Path,
        data_shape: Tuple[int, int, int],
        data_type: np.dtype,
        normalize_coords: bool = True,
        normalize_values: bool = True,
        order: str = "F",  # col major order
        batch_size: Optional[int] = None,  # if None, will be set later
        initial_shuffle: bool = True,
    ):
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
        self.data = np.fromfile(file_path, dtype=data_type)
        self.data_range = (np.min(self.data), np.max(self.data))

        if initial_shuffle:
            self.perm = torch.randperm(self.data.shape[0])
            self.coords = self.coords[self.perm]
            self.data = self.data[self.perm]

        if normalize_values:
            min, max = self.data_range
            self.data = (self.data - min) / (max - min)

        self.file_path = file_path
        self.data_shape = data_shape
        self.data_type = data_type
        self.initial_shuffle = initial_shuffle
        self.normalize_coords = normalize_coords
        self.normalize_values = normalize_values
        self.order = order
        self.batch_size = batch_size
        self.generator = torch.Generator()

    def __len__(self):
        if self.batch_size is None:
            raise ValueError("Batch size must be set for length calculation.")
        return math.ceil(self.data.shape[0] / self.batch_size)

    def __iter__(self):
        if self.batch_size is None:
            raise ValueError("Batch size must be set for iteration.")
        worker_info = get_worker_info()
        num_batches = len(self)
        if worker_info is None:
            # Single-process data loading
            for i in range(num_batches):
                yield self._get_batch(i)
        else:
            # Multi-process data loading
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            batches_per_worker = num_batches // num_workers
            start_batch = worker_id * batches_per_worker
            end_batch = (
                (worker_id + 1) * batches_per_worker
                if worker_id < num_workers - 1
                else num_batches
            )
            for i in range(start_batch, end_batch):
                yield self._get_batch(i)

    def _get_batch(self, batch_index: int) -> Tuple[torch.Tensor, ...]:
        start = batch_index * self.batch_size
        end = min((batch_index + 1) * self.batch_size, self.data.shape[0])
        batch_data = self.data[start:end]
        batch_coords = torch.tensor(self.coords[start:end])
        batch_data = torch.tensor(batch_data)
        return batch_coords, batch_data

    def volume_data(self):
        """Return the volume data in the original shape."""
        return self.data.reshape(self.data_shape, order=self.order)

    def unshuffle(self, data: torch.Tensor) -> torch.Tensor:
        if self.initial_shuffle:
            data = data[self.perm]
        return data

    def unnormalize(self, data: torch.Tensor) -> torch.Tensor:
        if self.normalize_values:
            min_val, max_val = self.data_range
            data = data * (max_val - min_val) + min_val
        return data
