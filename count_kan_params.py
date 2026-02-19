"""Load a small efficient KAN model and print its parameter count."""

import torch
from efficient_kan import KAN


def count_parameters(model):
    """Count total and trainable parameters in a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    # Small KAN: 3 input dims -> 16 hidden -> 16 hidden -> 1 output
    layers_hidden = [3, 16, 16, 1]
    model = KAN(layers_hidden=layers_hidden, grid_size=5, spline_order=3)

    total, trainable = count_parameters(model)

    print(f"KAN Architecture: {layers_hidden}")
    print(f"Grid size: {model.grid_size}, Spline order: {model.spline_order}")
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print()

    # Per-layer breakdown
    for i, layer in enumerate(model.layers):
        layer_params = sum(p.numel() for p in layer.parameters())
        print(
            f"  Layer {i} ({layer.in_features} -> {layer.out_features}): {layer_params:,} params"
        )


if __name__ == "__main__":
    main()
