import numpy as np
import torch
from torch.utils.data import Dataset


class VolumetricDataset(Dataset):
    def __init__(
        self,
        file_path,
        data_shape,
        data_type,
        normalize_coords=True,
        normalize_values=True,
        return_coords=False,
        order="F",  # col major order
    ):
        self.data = np.fromfile(file_path, dtype=data_type)
        if self.data.size != np.prod(data_shape):
            raise ValueError("Data shape does not match file size")
        self.data_shape = data_shape
        self.normalize_indices = normalize_coords
        self.normalize_values = normalize_values
        self.data_range = (np.min(self.data), np.max(self.data))
        if self.normalize_values:
            min, max = self.data_range
            self.data = (self.data - min) / (max - min)
        self.return_indices = return_coords
        self.order = order

    def __getitem__(self, index):
        # NOTE: NRRD004 files axis order is (left to right) fastest to slowest (column-major order)
        # see https://teem.sourceforge.net/nrrd/format.html#general.4
        d1, d2, d3 = self.data_shape
        i = index % d1
        j = (index // d1) % d2
        k = index // (d1 * d2)
        if self.return_indices:
            indices = (i, j, k)
        if self.normalize_indices:
            i = i / (d1 - 1)
            j = j / (d2 - 1)
            k = k / (d3 - 1)
        x = torch.tensor((i, j, k))
        y = torch.tensor(self.data[index])[None]
        if self.return_indices:
            return x, y, indices
        return x, y

    def __len__(self):
        return len(self.data)

    def volume_data(self):
        return self.data.reshape(self.data_shape, order=self.order)
