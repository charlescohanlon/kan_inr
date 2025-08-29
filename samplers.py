"""
Qi's optimized data sampler code and accompanying utilities.
"""

import os
from typing import List
import torch
import numpy as np


class VolumeDesc:
    def __init__(
        self, dims: List[int], dtype: str, device: str, numfields: int = 1, **kwargs
    ):
        self.device = torch.device(device)
        self._dims = dims
        self.dimx = dims[0]
        self.dimy = dims[1]
        self.dimz = dims[2]
        self.dims = torch.tensor(
            [self.dimx, self.dimy, self.dimz], dtype=torch.int32, device=self.device
        )
        self.numvoxels = dims[0] * dims[1] * dims[2]

        self._type = dtype
        if dtype == "uint8":
            _dtype = np.uint8
        elif dtype == "uint16":
            _dtype = np.uint16
        elif dtype == "uint32":
            _dtype = np.uint32
        elif dtype == "uint64":
            _dtype = np.uint64
        elif dtype == "int8":
            _dtype = np.int8
        elif dtype == "int16":
            _dtype = np.int16
        elif dtype == "int32":
            _dtype = np.int32
        elif dtype == "int64":
            _dtype = np.int64
        elif dtype == "float32":
            _dtype = np.float32
        elif dtype == "float64":
            _dtype = np.float64
        else:
            raise Exception(f"Unknown data type: {dtype}")
        self.type = _dtype

        self.numfields = numfields


class VolumeData(VolumeDesc):
    def __init__(self, *args, **kwargs):
        super(VolumeData, self).__init__(*args, **kwargs)
        self.data = None
        self.minmax = [None, None]

    def from_file(self, filename: str, offset: int = 0, **kwargs):
        fstat = os.stat(filename)
        expected_size = (
            self.dimx
            * self.dimy
            * self.dimz
            * np.dtype(self.type).itemsize
            * self.numfields
        )
        if fstat.st_size != expected_size:
            raise Exception(
                f"File size does not match: expected: {expected_size}, actual: {fstat.st_size}"
            )
        with open(filename, "rb") as file:
            self.data = np.fromfile(
                file,
                count=self.numvoxels * self.numfields,
                dtype=self.type,
                offset=offset,
            )
            self.data = self.data.reshape([self.numvoxels, self.numfields])
        self._normalize(**kwargs)

    def from_ndarray(self, data: np.ndarray, **kwargs):
        self.data = data
        self._normalize(**kwargs)

    def _normalize(self, minmax: List[float] = None, transform=None):
        """
        Need to normalize value to [0, 1]
        """
        self.data = torch.as_tensor(self.data, dtype=torch.float32, device=self.device)
        if minmax is None:
            dmax = torch.max(self.data)
            dmin = torch.min(self.data)
        else:
            dmin, dmax = minmax
        self.minmax = [dmin, dmax]
        self.data = (self.data - dmin) * (1.0 / (dmax - dmin))
        self.data = torch.clamp(self.data, min=0.0, max=1.0)
        if transform:  # Apply application defined transform (optional)
            self.data = transform(self.data)

    def _access(self, coordinates: torch.Tensor):
        """
        integer coordinates -->  torch.Tensor[N x 3]
        return --> torch.Tensor[N x numfields]
        """
        x, y, z = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
        indices = x + y * self.dimx + z * (self.dimx * self.dimy)
        return self.data[indices.int()]

    def nearest_interpolation(self, coordinates: torch.Tensor):
        """
        floating point coordinates -->  torch.Tensor[N x 3]
        return --> torch.Tensor[N x numfields]
        """
        assert torch.is_tensor(coordinates)
        assert len(coordinates.shape) == 2
        assert coordinates.shape[1] == 3
        xyz = coordinates * self.fdims
        return self._access(xyz.int())

    def linear_interpolation(self, coordinates: torch.Tensor):
        """
        floating point coordinates -->  torch.Tensor[N x 3]
        return --> torch.Tensor[N x numfields]
        """
        assert torch.is_tensor(coordinates)
        assert len(coordinates.shape) == 2
        assert coordinates.shape[1] == 3

        zeros3 = torch.zeros(3, dtype=torch.int32, device=self.device)

        fxyz = torch.clamp(coordinates * self.dims - 0.5, zeros3, self.dims - 1)
        fxyz0 = fxyz.int()
        fxyz1 = torch.clamp(fxyz0 + 1, zeros3, self.dims - 1)

        fx0 = fxyz0[:, 0]
        fy0 = fxyz0[:, 1]
        fz0 = fxyz0[:, 2]
        fx1 = fxyz1[:, 0]
        fy1 = fxyz1[:, 1]
        fz1 = fxyz1[:, 2]

        assert fx0.min() >= 0
        assert fy0.min() >= 0
        assert fz0.min() >= 0
        assert fx1.max() < self.dimx
        assert fy1.max() < self.dimy
        assert fz1.max() < self.dimz

        X = torch.stack((fx0, fx0, fx0, fx0, fx1, fx1, fx1, fx1), dim=1).flatten()
        Y = torch.stack((fy0, fy0, fy1, fy1, fy0, fy0, fy1, fy1), dim=1).flatten()
        Z = torch.stack((fz0, fz1, fz0, fz1, fz0, fz1, fz0, fz1), dim=1).flatten()
        coords = torch.stack((X, Y, Z), dim=1)

        grid = self._access(coords)  # torch.Size([-1, 3])
        grid = grid.reshape((-1, 8, self.numfields))

        fxyz -= fxyz0
        fx = fxyz[:, 0].unsqueeze(dim=-1)
        fy = fxyz[:, 1].unsqueeze(dim=-1)
        fz = fxyz[:, 2].unsqueeze(dim=-1)

        # step 1
        c00 = grid[:, 0] * (1 - fx) + grid[:, 4] * fx
        c01 = grid[:, 1] * (1 - fx) + grid[:, 5] * fx
        c10 = grid[:, 2] * (1 - fx) + grid[:, 6] * fx
        c11 = grid[:, 3] * (1 - fx) + grid[:, 7] * fx
        # step 2
        c0 = c00 * (1 - fy) + c10 * fy
        c1 = c01 * (1 - fy) + c11 * fy
        # step 3
        c = c0 * (1 - fz) + c1 * fz

        return c

    def sample(self, coordinates: torch.Tensor, trilinear: bool = True):
        if trilinear:
            ret = self.linear_interpolation(coordinates)
        else:
            ret = self.nearest_interpolation(coordinates)
        return ret

    def get_value_range(self):
        return [torch.min(self.data), torch.max(self.data)]

    def get_value_range_unnormalized(self):
        return self.minmax


class VolumeSampler(VolumeDesc):
    """
    Assuming cell-centered voxels
    """

    def __init__(self, *args, **kwargs):
        super(VolumeSampler, self).__init__(*args, **kwargs)
        self.set_bounds(**kwargs)
        self.gt = None

    def load_from_file(self, *args, **kwargs):
        self.gt = VolumeData(
            dims=self._dims, dtype=self._type, numfields=self.numfields
        )
        self.gt.from_file(*args, **kwargs)

    def load_from_ndarray(self, *args, **kwargs):
        self.gt = VolumeData(
            dims=self._dims,
            dtype=self._type,
            numfields=self.numfields,
            device=self.device,
        )
        self.gt.from_ndarray(*args, **kwargs)

    def load_from_callback(self, callback):
        self.gt = callback

    def set_bounds(self, bounds: List[List[float]] = None, **kwargs):
        """
        bounds: bounding box in the voxel (unnormalized) coordinate
        bounds_lo, bounds_hi: bounding box in the normalized local coordinate
        """
        if bounds:
            self.bounds = bounds
            self.bounds_lo = torch.tensor(
                np.asarray(bounds[0]) / np.asarray(self._dims),
                dtype=torch.float32,
                device=self.device,
            )
            self.bounds_hi = torch.tensor(
                np.asarray(bounds[1]) / np.asarray(self._dims),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            self.bounds = ((0, 0, 0), self._dims)
            self.bounds_lo = torch.zeros(3, device=self.device, dtype=torch.float32)
            self.bounds_hi = torch.ones(3, device=self.device, dtype=torch.float32)
        self.sampling_scale = self.bounds_hi - self.bounds_lo

    def get_samples(self, coordinates):
        return self.gt.sample(coordinates)

    def get_random_samples(self, numsamples: int):
        """Generate random coordinate and value pairs within the predefined bounding box"""
        coordinates = (
            torch.rand(numsamples, 3, dtype=torch.float32, device=self.device)
            * self.sampling_scale
            + self.bounds_lo
        )
        return coordinates, self.get_samples(coordinates)

    def get_decoding_inputs_bbox(self, lo: list, hi: list):
        linspace = lambda r0, r1, dim: torch.linspace(
            (r0 + 0.5) / dim,
            (r1 - 0.5) / dim,
            r1 - r0,
            dtype=torch.float32,
            device=self.device,
        )
        x, y, z = torch.meshgrid(
            linspace(lo[0], hi[0], self._dims[0]),
            linspace(lo[1], hi[1], self._dims[1]),
            linspace(lo[2], hi[2], self._dims[2]),
            indexing="xy",
        )
        coordinates = torch.stack((x, y, z), axis=3).permute(2, 0, 1, 3)
        return coordinates.reshape([-1, 3])

    def get_decoding_inputs_x_slice(self, x_value: float):
        x, y, z = torch.meshgrid(
            torch.tensor(x_value, dtype=torch.float32, device=self.device),
            torch.linspace(
                0.5 / self._dims[1],
                (self._dims[1] - 0.5) / self._dims[1],
                self._dims[1],
                dtype=torch.float32,
                device=self.device,
            ),
            torch.linspace(
                0.5 / self._dims[2],
                (self._dims[2] - 0.5) / self._dims[2],
                self._dims[2],
                dtype=torch.float32,
                device=self.device,
            ),
            indexing="xy",
        )
        coordinates = torch.stack((x, y, z), axis=3).permute(2, 0, 1, 3)
        return coordinates.reshape([-1, 3])

    def get_decoding_inputs_y_slice(self, y_value: float):
        x, y, z = torch.meshgrid(
            torch.linspace(
                0.5 / self._dims[0],
                (self._dims[0] - 0.5) / self._dims[0],
                self._dims[0],
                dtype=torch.float32,
                device=self.device,
            ),
            torch.tensor(y_value, dtype=torch.float32, device=self.device),
            torch.linspace(
                0.5 / self._dims[2],
                (self._dims[2] - 0.5) / self._dims[2],
                self._dims[2],
                dtype=torch.float32,
                device=self.device,
            ),
            indexing="xy",
        )
        coordinates = torch.stack((x, y, z), axis=3).permute(2, 0, 1, 3)
        return coordinates.reshape([-1, 3])

    def get_decoding_inputs_z_slice(self, z_value: float):
        x, y, z = torch.meshgrid(
            torch.linspace(
                0.5 / self._dims[0],
                (self._dims[0] - 0.5) / self._dims[0],
                self._dims[0],
                dtype=torch.float32,
                device=self.device,
            ),
            torch.linspace(
                0.5 / self._dims[1],
                (self._dims[1] - 0.5) / self._dims[1],
                self._dims[1],
                dtype=torch.float32,
                device=self.device,
            ),
            torch.tensor(z_value, dtype=torch.float32, device=self.device),
            indexing="xy",
        )
        coordinates = torch.stack((x, y, z), axis=3).permute(2, 0, 1, 3)
        return coordinates.reshape([-1, 3])

    def decode_volume(self, callback, network, verbose=True):
        from tqdm import tqdm, trange

        RANGE = trange if verbose else range
        for z in RANGE(self.bounds[0][2], self.bounds[1][2]):
            lo = (self.bounds[0][0], self.bounds[0][1], z)
            hi = (self.bounds[1][0], self.bounds[1][1], z + 1)
            coords = self.get_decoding_inputs_bbox(lo, hi)
            values = network(coords).cpu().numpy()
            values = np.ascontiguousarray(values)
            callback(values, z)

    def compute_mse(self, network_fn, step=4, verbose=True):
        from tqdm import tqdm, trange

        if self.gt is None:
            raise Exception("cannot compute MSE without a ground truth data")
        RANGE = trange if verbose else range
        progress = RANGE(self.bounds[0][2], self.bounds[1][2], step)
        # start computation
        mse = 0.0
        for z in progress:
            lo = (self.bounds[0][0], self.bounds[0][1], z)
            hi = (
                self.bounds[1][0],
                self.bounds[1][1],
                min(z + step, self.bounds[1][2]),
            )
            coords = self.get_decoding_inputs_bbox(lo, hi)
            targets = self.get_samples(coords)
            predictions = network_fn(coords)
            assert targets.shape == predictions.shape
            mse += torch.mean((targets - predictions) ** 2).cpu().numpy()
        mse /= (self._dims[2] + step - 1) / step
        return mse
