import math
from typing import Tuple, Optional
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info
from pathlib import Path
import torch.distributed as dist


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
            # Use a fixed seed for reproducible shuffling across all ranks
            rng = np.random.RandomState(0)  # Fixed seed
            self.perm = torch.from_numpy(rng.permutation(self.data.shape[0]))
            self.coords = self.coords[self.perm]
            self.data = self.data[self.perm]

        if normalize_values:
            min_val, max_val = self.data_range
            self.data = (self.data - min_val) / (max_val - min_val)

        self.file_path = file_path
        self.data_shape = data_shape
        self.data_type = data_type
        self.initial_shuffle = initial_shuffle
        self.normalize_coords = normalize_coords
        self.normalize_values = normalize_values
        self.order = order
        self.batch_size = batch_size
        self.generator = torch.Generator()

        # DDP-related attributes
        self.rank = 0
        self.world_size = 1
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def __len__(self):
        if self.batch_size is None:
            raise ValueError("Batch size must be set for length calculation.")
        # Total samples for THIS rank
        samples_per_rank = self._get_samples_per_rank()
        return math.ceil(samples_per_rank / self.batch_size)

    def _get_samples_per_rank(self):
        """Calculate number of samples for this rank"""
        total_samples = self.data.shape[0]
        samples_per_rank = total_samples // self.world_size
        # Handle remainder - last rank gets extra samples
        if self.rank == self.world_size - 1:
            samples_per_rank += total_samples % self.world_size
        return samples_per_rank

    def _get_rank_data_range(self):
        """Get the start and end indices for this rank's data"""
        total_samples = self.data.shape[0]
        samples_per_rank = total_samples // self.world_size

        start_idx = self.rank * samples_per_rank
        if self.rank == self.world_size - 1:
            # Last rank gets any remainder
            end_idx = total_samples
        else:
            end_idx = start_idx + samples_per_rank

        return start_idx, end_idx

    def __iter__(self):
        if self.batch_size is None:
            raise ValueError("Batch size must be set for iteration.")

        # Get this rank's portion of data
        rank_start, rank_end = self._get_rank_data_range()
        rank_samples = rank_end - rank_start
        num_batches = math.ceil(rank_samples / self.batch_size)

        worker_info = get_worker_info()

        if worker_info is None:
            # Single-process data loading for this rank
            for i in range(num_batches):
                yield self._get_batch(i, rank_start, rank_end)
        else:
            # Multi-process data loading - split this rank's data across workers
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

            # Calculate batches per worker
            batches_per_worker = num_batches // num_workers
            start_batch = worker_id * batches_per_worker

            if worker_id == num_workers - 1:
                # Last worker gets remainder
                end_batch = num_batches
            else:
                end_batch = start_batch + batches_per_worker

            for i in range(start_batch, end_batch):
                yield self._get_batch(i, rank_start, rank_end)

    def _get_batch(
        self, batch_index: int, rank_start: int, rank_end: int
    ) -> Tuple[torch.Tensor, ...]:
        # Calculate indices relative to this rank's data portion
        start = rank_start + (batch_index * self.batch_size)
        end = min(rank_start + ((batch_index + 1) * self.batch_size), rank_end)

        batch_coords = torch.tensor(self.coords[start:end], dtype=torch.float32)
        batch_data = torch.tensor(self.data[start:end], dtype=torch.float32)

        return batch_coords, batch_data

    def volume_data(self):
        """Return the FULL volume data in the original shape (for evaluation)."""
        # This returns ALL data, not just this rank's portion
        return self.data.reshape(self.data_shape, order=self.order)

    def unshuffle(self, data: torch.Tensor) -> torch.Tensor:
        if self.initial_shuffle:
            # Create inverse permutation
            inverse_perm = torch.zeros_like(self.perm)
            inverse_perm[self.perm] = torch.arange(len(self.perm))
            data = data[inverse_perm]
        return data

    def unnormalize(self, data: torch.Tensor) -> torch.Tensor:
        if self.normalize_values:
            min_val, max_val = self.data_range
            data = data * (max_val - min_val) + min_val
        return data
