# ==============================================================
# Extracted FKAN nn module from 'FKAN_INR.ipynb'
# Classes included: FKAN_INR, FKANLayer, SineLayer
# Additional dependent classes: FourierKANLayer
# IPython magics and shell escapes are commented out.
# All notebook Markdown appears below as commented context.
# ==============================================================

# ----------------- Notebook Markdown (as comments) -----------------
# ## Defining Model

# ### Defining desired Positional Encoding

# ### Model Configureations

# ----------------- End Markdown -----------------


# ----------------- Imports (pruned) -----------------
import numpy as np
import torch
from torch import nn


# ----------------- Additional Dependent Classes -----------------
class FourierKANLayer(nn.Module):
    def __init__(
        self, input_dim, output_dim, gridsize, addbias=True, smooth_initialization=False
    ):
        super(FourierKANLayer, self).__init__()
        self.gridsize = gridsize
        self.addbias = addbias
        self.inputdim = input_dim
        self.outdim = output_dim

        grid_norm_factor = (
            (torch.arange(gridsize) + 1) ** 2
            if smooth_initialization
            else np.sqrt(gridsize)
        )

        # The normalization has been chosen so that if given inputs where each coordinate is of unit variance,
        # then each coordinates of the output is of unit variance
        # independently of the various sizes
        self.fouriercoeffs = torch.nn.Parameter(
            torch.randn(2, output_dim, input_dim, gridsize)
            / (np.sqrt(input_dim) * grid_norm_factor)
        )

        if self.addbias:
            self.bias = torch.nn.Parameter(torch.zeros(1, output_dim))

    def forward(self, x):

        xshp = x.shape

        outshape = xshp[0:-1] + (self.outdim,)
        x = torch.reshape(x, (-1, self.inputdim))

        # Starting at 1 because constant terms are in the bias
        k = torch.reshape(
            torch.arange(1, self.gridsize + 1, device=x.device),
            (1, 1, 1, self.gridsize),
        )

        xrshp = torch.reshape(x, (x.shape[0], 1, x.shape[1], 1))

        # This should be fused to avoid materializing memory
        c = torch.cos(k * xrshp)
        s = torch.sin(k * xrshp)

        # We compute the interpolation of the various functions defined by their fourier coefficient for each input coordinates and we sum them
        y = torch.sum(c * self.fouriercoeffs[0:1], (-2, -1))
        y += torch.sum(s * self.fouriercoeffs[1:2], (-2, -1))
        if self.addbias:
            y += self.bias
        # End fuse

        y = torch.reshape(y, outshape)
        return y


# ----------------- FKAN Module Classes -----------------
class FourierKAN_INR(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, grid):
        super(FourierKAN_INR, self).__init__()

        self.fkan = FKANLayer(in_features, hidden_features, grid)
        # self.fkan2=FKANLayer(hidden_features,out_features,grid)

        self.hid1 = SineLayer(hidden_features, 2 * hidden_features)

        self.hid2 = SineLayer(2 * hidden_features, 2 * hidden_features)

        self.hid3 = SineLayer(2 * hidden_features, 2 * hidden_features)

        self.hid4 = SineLayer(2 * hidden_features, 4 * hidden_features)
        # self.hid3=SineLayer(hidden_features,hidden_features)

        self.out = nn.Linear(4 * hidden_features, out_features)

        with torch.no_grad():
            const = np.sqrt(6 / hidden_features) / 30
            self.out.weight.uniform_(-const, const)

        # self.out=SineLayer(hidden_features,out_features)

        # self.net = nn.Sequential(
        # Bern layer
        #    FKANLayer(in_features, hidden_features,grid),
        # nn.ReLU(),

        # SineLayer(hidden_features,512),

        #    SineLayer(hidden_features,hidden_features),
        #    SineLayer(hidden_features,hidden_features),
        #    SineLayer(hidden_features,hidden_features),
        # First linear MLP layer
        # nn.Linear(hidden_features, hidden_features),
        # nn.ReLU(),

        #    nn.Linear(hidden_features,out_features),
        # nn.ReLU(),

        # Second linear MLP layer (output layer)
        # SineLayer(hidden_features, out_features)
        # FKANLayer(hidden_features, out_features,grid),
        # )

        # print(f"Network architecture:")
        # print(f"  FKAN layer: {in_features} -> {hidden_features}")
        # print(f"  MLP layer 1: {hidden_features} -> {hidden_features}")
        # print(f"  MLP layer 2 (output): {hidden_features} -> {out_features}")

    def forward(self, coords):

        x = self.fkan(coords)

        y1 = self.hid1(x)
        # y1=self.n1(y1)
        y2 = self.hid2(y1)

        y3 = self.hid3(y2)
        y4 = self.hid4(y3)
        # y2=self.n2(y2)

        y4 = self.out(y4)

        # return self.net(coords)
        return y4


class FKANLayer(nn.Module):
    def __init__(self, in_features, out_features, grid):
        super(FKANLayer, self).__init__()
        self.fkan = FourierKANLayer(in_features, out_features, grid)
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x):
        x = self.fkan(x)
        x = self.norm(x)
        return x


class SineLayer(nn.Module):
    """
    SineLayer is a custom PyTorch module that applies the Sinusoidal activation function to the output of a linear transformation.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool, optional): If True, the linear transformation includes a bias term. Default is True.
        is_first (bool, optional): If it is the first layer, we initialize the weights differently. Default is False.
        omega_0 (float, optional): Frequency scaling factor for the sinusoidal activation. Default is 30.
        scale (float, optional): Scaling factor for the output of the sine activation. Default is 10.0.
        init_weights (bool, optional): If True, initializes the layer's weights according to the SIREN paper. Default is True.

    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        is_first=False,
        omega_0=30,
        scale=10.0,
        init_weights=True,
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        # self.grid=grid
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        # k = torch.reshape( torch.arange(1,grid+1),(1,1,1,self.gridsize))

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0,
                )

    def forward(self, input):
        # k = torch.reshape( torch.arange(1,self.grid+1,device=input.device),(1,1,self.grid))

        # a=self.linear(input)
        # a=a[:,:,None]

        # print(a.shape)

        # s = torch.sin( k*a )

        # return torch.tanh(self.omega_0 *self.linear(input))
        #######################################################################

        # return (self.linear(input)+torch.sin(self.omega_0 *self.linear(input)))*nn.Sigmoid()(self.linear(input))

        # y=self.linear(input)
        ################################################################################
        return (
            self.linear(input) + torch.tanh(self.omega_0 * self.linear(input))
        ) * nn.Sigmoid()(self.linear(input))
