import os, math
from typing import Tuple
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from fastkan import FastKAN


if torch.cuda.is_available():
    import tinycudann
    from tinycudann.modules import _C as _native
    from tinycudann.modules import _torch_precision, _module_function

    class EncodingFunction(nn.Module):
        def __init__(self, n_input_dims, encoding_config, dtype=None):
            super(EncodingFunction, self).__init__()

            self.n_input_dims = n_input_dims
            self.encoding_config = encoding_config

            self._create_native_tcnn_module(dtype)

            self.dtype = _torch_precision(self.native_tcnn_module.param_precision())
            self.loss_scale = (
                128.0
                if self.native_tcnn_module.param_precision() == _native.Precision.Fp16
                else 1.0
            )

            self.n_output_dims = self.native_tcnn_module.n_output_dims()

        def _create_native_tcnn_module(self, dtype):
            if dtype is None:
                precision = _native.preferred_precision()
            else:
                if dtype == torch.float32:
                    precision = _native.Precision.Fp32
                elif dtype == torch.float16:
                    precision = _native.Precision.Fp16
                else:
                    raise ValueError(
                        f"Encoding only supports fp32 or fp16 precision, but got {dtype}"
                    )
            self.native_tcnn_module = _native.create_encoding(
                self.n_input_dims, self.encoding_config, precision
            )

        def initial_params(self, seed=1337):
            return self.native_tcnn_module.initial_params(seed)

        def n_params(self):
            return self.native_tcnn_module.n_params()

        def forward(self, x, params):
            if not x.is_cuda:
                warnings.warn(
                    "input must be a CUDA tensor, but isn't. This indicates suboptimal performance."
                )
                x = x.cuda()

            batch_size = x.shape[0]
            batch_size_granularity = int(_native.batch_size_granularity())
            padded_batch_size = (
                (batch_size + batch_size_granularity - 1)
                // batch_size_granularity
                * batch_size_granularity
            )

            x_padded = (
                x
                if batch_size == padded_batch_size
                else torch.nn.functional.pad(
                    x, [0, 0, 0, padded_batch_size - batch_size]
                )
            )
            output = _module_function.apply(
                self.native_tcnn_module,
                x_padded.to(torch.float).contiguous(),
                params.to(
                    _torch_precision(self.native_tcnn_module.param_precision())
                ).contiguous(),
                self.loss_scale,
            )
            return output[:batch_size, : self.n_output_dims]


class HashEmbedderTCNN(nn.Module):
    def __init__(
        self,
        n_pos_dims=3,
        seed=1337,
        # encoder parameters
        n_levels=16,
        n_features_per_level=2,
        log2_hashmap_size=19,
        base_resolution=16,
        per_level_scale=2.0,
    ):
        super(HashEmbedderTCNN, self).__init__()

        self.n_pos_dims = n_pos_dims

        encoding_config = {
            "otype": "HashGrid",
            "n_levels": n_levels,
            "n_features_per_level": n_features_per_level,
            "log2_hashmap_size": log2_hashmap_size,
            "base_resolution": base_resolution,
            "per_level_scale": per_level_scale,
        }

        self.encoding_config = encoding_config

        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.per_level_scale = per_level_scale

        self.embeddings = tinycudann.Encoding(
            n_input_dims=self.n_pos_dims, encoding_config=encoding_config, seed=seed
        )
        self.n_output_dims = self.embeddings.n_output_dims

    def forward(self, x):
        return self.embeddings(x)


class HashEmbedderNative(nn.Module):
    def __init__(
        self,
        n_pos_dims=3,
        # encoder parameters
        n_levels=16,
        n_features_per_level=4,
        log2_hashmap_size=19,
        base_resolution=16,
        per_level_scale=2.0,
    ):
        super(HashEmbedderNative, self).__init__()

        # only support 3 dimensional inputs for now
        assert n_pos_dims == 3

        self.n_pos_dims = n_pos_dims

        encoding_config = {
            "otype": "HashGrid",
            "n_levels": n_levels,
            "n_features_per_level": n_features_per_level,
            "log2_hashmap_size": log2_hashmap_size,
            "base_resolution": base_resolution,
            "per_level_scale": per_level_scale,
        }

        self.encoding_config = encoding_config

        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.per_level_scale = per_level_scale

        self.n_output_dims = self.n_levels * self.n_features_per_level

        embedding_offsets = []
        embedding_lengths = []
        offset = 0
        for i in range(self.n_levels):
            scale = self.grid_scale(i, self.per_level_scale, self.base_resolution)
            resolution = self.grid_resolution(scale)
            length = resolution**n_pos_dims
            length = (
                (length + 8 - 1) // 8 * 8
            )  # Make sure memory accesses will be aligned
            length = min(length, 1 << self.log2_hashmap_size)
            embedding_offsets.append(offset)
            embedding_lengths.append(length)
            offset += length
        self.embedding_offsets = embedding_offsets
        self.embedding_lengths = embedding_lengths

        # https://github.com/NVlabs/tiny-cuda-nn/blob/v1.6/include/tiny-cuda-nn/encodings/grid.h#L1355
        scale = 1.0
        self.params = nn.Parameter(
            data=torch.zeros(offset * self.n_features_per_level, dtype=torch.float32)
        )
        self.register_parameter("params", self.params)
        nn.init.uniform_(self.params, -1e-4 * scale, 1e-4 * scale)

    @staticmethod
    @torch.no_grad()
    def trilinear_interp_weights(weights):
        c0 = (
            (1 - weights[..., 0]) * (1 - weights[..., 1]) * (1 - weights[..., 2])
        )  # c00 c0
        c1 = (1 - weights[..., 0]) * (1 - weights[..., 1]) * weights[..., 2]  # c01 c1
        c2 = (1 - weights[..., 0]) * weights[..., 1] * (1 - weights[..., 2])  # c10 c0
        c3 = (1 - weights[..., 0]) * weights[..., 1] * weights[..., 2]  # c11 c1
        c4 = weights[..., 0] * (1 - weights[..., 1]) * (1 - weights[..., 2])  # c00 c0
        c5 = weights[..., 0] * (1 - weights[..., 1]) * weights[..., 2]  # c01 c1
        c6 = weights[..., 0] * weights[..., 1] * (1 - weights[..., 2])  # c10 c0
        c7 = weights[..., 0] * weights[..., 1] * weights[..., 2]  # c11 c1
        return torch.stack([c0, c1, c2, c3, c4, c5, c6, c7], dim=-1)

    @staticmethod
    @torch.no_grad()
    def grid_scale(level: int, per_level_scale: float, base_resolution: float):
        return np.power(
            np.float32(2), np.float32(level) * np.log2(np.float32(per_level_scale))
        ) * np.float32(base_resolution) - np.float32(1.0)

    @staticmethod
    @torch.no_grad()
    def grid_resolution(scale: float):
        return np.int32(np.ceil(np.float32(scale))) + 1

    # https://github.com/NVlabs/tiny-cuda-nn/blob/v1.6/include/tiny-cuda-nn/common_device.h#L403
    @staticmethod
    @torch.no_grad()
    def grid_indices(scale: int, coords: torch.Tensor):
        positions = (coords * scale + 0.5).to(torch.float32)
        indices = torch.floor(positions).to(torch.int32)  # shape => [B, 3]
        positions = positions - indices  # fractional part
        offsets = coords.new_tensor(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ],
            dtype=torch.int32,
        )  # shape => [8, 3]
        indices = indices.unsqueeze(-2) + offsets.unsqueeze(0)
        return indices, positions

    @staticmethod
    @torch.no_grad()
    def hash_it(hashmap_size: int, resolution: int, indices: torch.Tensor):
        """It is possible that the coordinate is larger than the domain size."""
        HASH = coherent_prime_hash  # TCNN provides multiple hash functions
        assert indices.shape[-1] == 3
        resolution = np.uint32(resolution)
        stride = np.uint32(1)
        output = torch.zeros_like(indices[..., 0])
        for dim in range(3):
            output += indices[..., dim] * stride
            stride *= resolution  # --> expecting integer overflow in scalar multiply
            if stride > hashmap_size:
                break
        if hashmap_size < stride:
            output = HASH(indices)
        return output % hashmap_size

    @torch.no_grad()
    def access(self, coords: torch.Tensor, level: int):
        scale = self.grid_scale(level, self.per_level_scale, self.base_resolution)
        resolution = self.grid_resolution(scale)
        hashmap_size = self.embedding_lengths[level]
        indices, fractions = self.grid_indices(scale, coords)
        offsets = self.hash_it(hashmap_size, resolution, indices)
        return offsets, fractions

    def forward(self, coords: torch.Tensor):
        coords = coords.contiguous().to(torch.float32)

        with torch.no_grad():
            weights_arr = []
            offsets_arr = []
            for i in range(self.n_levels):
                offsets, weights = self.access(coords, i)
                offsets_arr.append(offsets.unsqueeze(1))
                weights_arr.append(weights.unsqueeze(1))
            offsets_arr = torch.cat(offsets_arr, dim=1)
            weights_arr = torch.cat(weights_arr, dim=1)
            weights_arr = self.trilinear_interp_weights(weights_arr)

        embeds_arr = F.embedding(
            offsets_arr, self.params.reshape((-1, self.n_features_per_level))
        )
        out = (weights_arr.unsqueeze(-1) * embeds_arr).sum(dim=-2)
        return out.reshape(-1, self.n_output_dims)

    def extra_repr(self):
        return f"hyperparams={self.encoding_config}"


def find_activation(activation):
    if activation == "None":
        return lambda x: x
    if activation == "ReLU":
        return lambda x: F.relu(x, inplace=True)
    elif activation == "Sigmoid":
        return F.sigmoid


MLP_ALIGNMENT = 16


class MLP_Native(torch.nn.Module):
    def __init__(
        self,
        n_input_dims=3,
        n_output_dims=1,
        bias=False,
        n_hidden_layers=3,
        n_neurons=64,
        activation="ReLU",
        output_activation="None",
    ):
        super(MLP_Native, self).__init__()

        self.n_input_dims = n_input_dims
        self.n_output_dims = n_output_dims

        network_config = {
            "otype": "FullyFusedMLP",
            "activation": activation,
            "output_activation": output_activation,
            "n_neurons": n_neurons,
            "n_hidden_layers": n_hidden_layers,
            "feedback_alignment": False,
        }

        self.n_hidden_layers = n_hidden_layers
        self.n_neurons = n_neurons
        self.bias = bias

        self.network_config = network_config

        assert n_hidden_layers >= 0, "expect at least one hidden layer"

        self.first = nn.Linear(self.n_input_dims, self.n_neurons, bias=self.bias)
        self.hidden = nn.ModuleList(
            [
                nn.Linear(self.n_neurons, self.n_neurons, bias=self.bias)
                for _ in range(self.n_hidden_layers - 1)
            ]
        )
        self.last = nn.Linear(
            self.n_neurons,
            (self.n_output_dims + MLP_ALIGNMENT - 1) // MLP_ALIGNMENT * MLP_ALIGNMENT,
            bias=self.bias,
        )

        self.activation = find_activation(activation)
        self.output_activation = find_activation(output_activation)

    def forward(self, x):
        x = self.activation(self.first(x))
        for layer in self.hidden:
            x = self.activation(layer(x))
        return self.output_activation(self.last(x))[..., : self.n_output_dims]


class MLP_TCNN(torch.nn.Module):
    def __init__(
        self,
        n_input_dims=3,
        n_output_dims=1,
        bias=False,
        seed=1337,
        n_hidden_layers=3,
        n_neurons=64,
        activation="ReLU",
        output_activation="None",
    ):
        super(MLP_TCNN, self).__init__()

        self.n_input_dims = n_input_dims
        self.n_output_dims = n_output_dims

        network_config = {
            "otype": "FullyFusedMLP",
            "activation": activation,
            "output_activation": output_activation,
            "n_neurons": n_neurons,
            "n_hidden_layers": n_hidden_layers,
            "feedback_alignment": False,
        }

        self.n_hidden_layers = n_hidden_layers
        self.n_neurons = n_neurons
        self.bias = False

        self.network_config = network_config

        assert bias == False, "TCNN MLP doesnot contain bias"

        self.model = tinycudann.Network(
            n_input_dims=self.n_input_dims,
            n_output_dims=self.n_output_dims,
            network_config=network_config,
            seed=seed,
        )

    def forward(self, x):
        return self.model(x)


class INR_Base(nn.Module):
    def __init__(
        self,
        n_input_dims=3,
        n_output_dims=1,
        native_encoder=True,
        native_network=True,
        network_type="mlp",
        # network paramerers
        n_hidden_layers=3,
        n_neurons=64,
        # encoder paraparametersms
        n_levels=16,
        n_features_per_level=4,
        log2_hashmap_size=19,
        base_resolution=16,
        per_level_scale=2.0,
        activation="ReLU",
        output_activation="None",
    ):
        super(INR_Base, self).__init__()

        self.n_input_dims = n_input_dims
        self.n_output_dims = n_output_dims

        ENCODER = HashEmbedderNative if native_encoder else HashEmbedderTCNN
        if network_type.lower() == "mlp":
            NETWORK = MLP_Native if native_network else MLP_TCNN
        elif network_type.lower() == "kan":
            NETWORK = FKAN_Native
        else:
            raise ValueError(f"Unsupported network type {network_type}")

        # Make sure memory accesses will be aligned
        n_enc_out = n_levels * n_features_per_level
        n_enc_pad = (n_enc_out + MLP_ALIGNMENT - 1) // MLP_ALIGNMENT * MLP_ALIGNMENT
        self.n_pad = n_enc_pad - n_enc_out

        self.network = NETWORK(
            n_input_dims=n_enc_pad,
            n_output_dims=n_output_dims,
            n_hidden_layers=n_hidden_layers,
            n_neurons=n_neurons,
            activation=activation,
            output_activation=output_activation,
        )

        self.encoder = ENCODER(
            n_pos_dims=n_input_dims,
            n_levels=n_levels,
            n_features_per_level=n_features_per_level,
            log2_hashmap_size=log2_hashmap_size,
            base_resolution=base_resolution,
            per_level_scale=per_level_scale,
        )

        self.encoding_config = self.encoder.encoding_config
        self.network_config = self.network.network_config

    def forward(self, x):
        h = self.encoder(x)
        h = F.pad(h, (0, self.n_pad))
        return self.network(h)


class INR_TCNN(nn.Module):
    def __init__(
        self,
        n_input_dims=3,
        n_output_dims=1,
        seed=1337,
        # network paramerers
        n_hidden_layers=3,
        n_neurons=64,
        # encoder paraparametersms
        n_levels=16,
        n_features_per_level=4,
        log2_hashmap_size=19,
        base_resolution=16,
        per_level_scale=2.0,
        activation="ReLU",
        output_activation="None",
    ):
        super(INR_TCNN, self).__init__()

        self.n_input_dims = n_input_dims
        self.n_output_dims = n_output_dims

        encoding_config = {
            "otype": "HashGrid",
            "n_levels": n_levels,
            "n_features_per_level": n_features_per_level,
            "log2_hashmap_size": log2_hashmap_size,
            "base_resolution": base_resolution,
            "per_level_scale": per_level_scale,
        }

        network_config = {
            "otype": "FullyFusedMLP",
            "activation": activation,
            "output_activation": output_activation,
            "n_neurons": n_neurons,
            "n_hidden_layers": n_hidden_layers,
            "feedback_alignment": False,
        }

        self.encoding_config = encoding_config
        self.network_config = network_config

        self.model = tinycudann.NetworkWithInputEncoding(
            n_input_dims=n_input_dims,
            n_output_dims=n_output_dims,
            encoding_config=encoding_config,
            network_config=network_config,
            seed=seed,
        )

        self.encoding_func = EncodingFunction(
            n_input_dims=n_input_dims, encoding_config=encoding_config
        )

    def forward(self, x):
        return self.model(x)

    def n_encoding_params(self):
        return self.encoding_func.n_params()


class FKAN_Native(torch.nn.Module):
    def __init__(
        self,
        n_input_dims=3,
        n_output_dims=1,
        n_hidden_layers=3,
        n_neurons=64,
        activation="SiLU",
        output_activation="None",
    ):
        super(FKAN_Native, self).__init__()

        self.n_input_dims = n_input_dims
        self.n_output_dims = n_output_dims

        network_config = {
            "otype": "FastKAN",
            "activation": activation,
            "output_activation": output_activation,
            "n_neurons": n_neurons,
            "n_hidden_layers": n_hidden_layers,
            "feedback_alignment": False,
        }

        self.n_hidden_layers = n_hidden_layers
        self.n_neurons = n_neurons

        self.network_config = network_config

        assert n_hidden_layers >= 0, "expect at least one hidden layer"

        layers_hidden = [n_input_dims] + [n_neurons] * n_hidden_layers + [n_output_dims]
        self.fkan = FastKAN(
            layers_hidden=layers_hidden,
            grid_min=-1.0,
            grid_max=1.0,
            num_grids=8,
            use_base_update=True,
            base_activation=F.silu,
            spline_weight_init_scale=0.1,
        )

    def forward(self, x):
        return self.fkan(x)


def coherent_prime_hash(coords, log2_hashmap_size=0):
    """
    coords: this function can process upto 7 dim coordinates
    log2T:  logarithm of T w.r.t 2
    """
    primes = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]

    xor_result = torch.zeros_like(coords)[..., 0]
    for i in range(coords.shape[-1]):
        xor_result ^= coords[..., i] * primes[i]

    if log2_hashmap_size == 0:
        return xor_result
    else:
        return (
            torch.tensor((1 << log2_hashmap_size) - 1).to(xor_result.device)
            & xor_result
        )
