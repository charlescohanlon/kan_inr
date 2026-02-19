"""
Given a KAN configuration, find the multiresolution hash encoder size
(log2_hashmap_size) such that SIREN/MLP + encoder total params ≈ KAN params.

Usage:
    python match_params.py
"""

import math


# ─────────────────────────────────────────────────────
# 1. KAN parameter counting (from efficient_kan.py)
# ─────────────────────────────────────────────────────


def kan_layer_params(in_features, out_features, grid_size=5, spline_order=3):
    """Parameter count for a single KANLinear layer.

    Each KANLinear has:
      - spline_weight:  out_features × in_features × (grid_size + spline_order)
      - base_weight:    out_features × in_features
      - spline_scaler:  out_features × in_features
    """
    k = grid_size + spline_order
    spline = out_features * in_features * k
    base = out_features * in_features
    scaler = out_features * in_features
    return spline + base + scaler


def kan_total_params(layers_hidden, grid_size=5, spline_order=3):
    """Total parameter count for a KAN model given layer widths."""
    total = 0
    for i in range(len(layers_hidden) - 1):
        total += kan_layer_params(
            layers_hidden[i], layers_hidden[i + 1], grid_size, spline_order
        )
    return total


# ─────────────────────────────────────────────────────
# 2. MLP parameter counting (bias=False by default)
# ─────────────────────────────────────────────────────


def mlp_params(n_input_dims, n_output_dims, n_hidden_layers, n_neurons, bias=False):
    """Parameter count for MLP_Native (no bias by default)."""
    b = 1 if bias else 0
    total = 0
    # first layer
    total += n_input_dims * n_neurons + b * n_neurons
    # hidden layers (n_hidden_layers - 1 intermediate)
    for _ in range(n_hidden_layers - 1):
        total += n_neurons * n_neurons + b * n_neurons
    # last layer
    total += n_neurons * n_output_dims + b * n_output_dims
    return total


# ─────────────────────────────────────────────────────
# 3. SIREN parameter counting (always has bias=True)
# ─────────────────────────────────────────────────────


def siren_params(
    n_input_dims, n_output_dims, n_hidden_layers, n_neurons, is_residual=False
):
    """Parameter count for FieldNet (SIREN). Always uses bias.

    Non-residual SIREN:
      - Each layer: (in + 1) * out  (weight + bias)

    Residual SIREN:
      - First layer: (d_in + 1) * n_neurons  (SineLayer)
      - Hidden layers: 2 * (n_neurons + 1) * n_neurons  (ResidualSineLayer with two nn.Linear)
      - Last layer: (n_neurons + 1) * d_out  (nn.Linear with bias)
    """
    layers = [n_input_dims] + [n_neurons] * n_hidden_layers + [n_output_dims]
    n_layers = len(layers) - 1  # number of layer transitions

    total = 0
    for ndx in range(n_layers):
        layer_in = layers[ndx]
        layer_out = layers[ndx + 1]

        if ndx != n_layers - 1:
            if not is_residual:
                # SineLayer: nn.Linear with bias
                total += (layer_in + 1) * layer_out
            else:
                if ndx == 0:
                    # First layer: SineLayer
                    total += (layer_in + 1) * layer_out
                else:
                    # ResidualSineLayer: two nn.Linear(features, features, bias=True)
                    total += 2 * ((layer_in + 1) * layer_in)
        else:
            # Final nn.Linear with bias
            total += (layer_in + 1) * layer_out

    return total


# ─────────────────────────────────────────────────────
# 4. Hash encoder parameter counting
#    Theoretical computation matching the Instant NGP /
#    tinycudann spec. Uses Python arbitrary-precision ints.
# ─────────────────────────────────────────────────────


def grid_scale(level, per_level_scale, base_resolution):
    """Compute the continuous scale for a given level (Instant NGP formula)."""
    return (per_level_scale**level) * base_resolution - 1.0


def grid_resolution(scale):
    """Compute the integer grid resolution from a continuous scale."""
    return int(math.ceil(scale)) + 1


def encoder_params(
    n_pos_dims=3,
    n_levels=16,
    n_features_per_level=4,
    log2_hashmap_size=19,
    base_resolution=16,
    per_level_scale=2.0,
):
    """Compute the theoretical number of parameters in a multiresolution
    hash encoder (Instant NGP / tinycudann).

    For each level:
      - resolution = ceil(base_resolution * per_level_scale^level - 1) + 1
      - entries = min(resolution^n_pos_dims, 2^log2_hashmap_size)
      - aligned to multiples of 8
    Total params = sum(entries) * n_features_per_level
    """
    offset = 0
    level_details = []
    hashmap_size = 1 << log2_hashmap_size

    for i in range(n_levels):
        scale = grid_scale(i, per_level_scale, base_resolution)
        resolution = grid_resolution(scale)
        # Python ints: no overflow
        n_vertices = resolution**n_pos_dims
        # Align to 8
        n_vertices = (n_vertices + 7) // 8 * 8
        # Cap at hash table max
        n_vertices = min(n_vertices, hashmap_size)
        level_details.append((i, resolution, n_vertices))
        offset += n_vertices

    total = offset * n_features_per_level
    return total, level_details


# ─────────────────────────────────────────────────────
# 5. Solver: find log2_hashmap_size to match a target
# ─────────────────────────────────────────────────────


def find_log2_hashmap_size(
    target_encoder_params,
    n_pos_dims=3,
    n_levels=16,
    n_features_per_level=4,
    base_resolution=16,
    per_level_scale=2.0,
):
    """Binary search for the smallest log2_hashmap_size such that
    encoder_params >= target_encoder_params.

    Returns (log2_hashmap_size, actual_encoder_params).
    """
    if target_encoder_params <= 0:
        return 0, 0

    # Search range: log2_hashmap_size in [1, 30]
    best = None
    for l2hs in range(1, 31):
        ep, _ = encoder_params(
            n_pos_dims=n_pos_dims,
            n_levels=n_levels,
            n_features_per_level=n_features_per_level,
            log2_hashmap_size=l2hs,
            base_resolution=base_resolution,
            per_level_scale=per_level_scale,
        )
        if ep >= target_encoder_params:
            best = (l2hs, ep)
            break
        best = (l2hs, ep)

    return best


# ─────────────────────────────────────────────────────
# 6. Main: put it all together
# ─────────────────────────────────────────────────────


def match_params(
    # KAN config
    kan_layers_hidden,
    kan_grid_size=5,
    kan_spline_order=3,
    # Network config (SIREN/MLP that we're matching)
    network_type="siren",  # "siren" or "mlp"
    n_input_dims=3,
    n_output_dims=1,
    n_hidden_layers=3,
    n_neurons=64,
    mlp_bias=False,
    siren_residual=False,
    # Encoder config (everything except log2_hashmap_size)
    n_levels=16,
    n_features_per_level=4,
    base_resolution=16,
    per_level_scale=2.0,
):
    """Find log2_hashmap_size so that (network + encoder) params ≈ KAN params."""

    # ── KAN target ──
    kan_p = kan_total_params(kan_layers_hidden, kan_grid_size, kan_spline_order)

    # ── Network params (fixed) ──
    n_enc_out = n_levels * n_features_per_level  # encoder output feeds the network
    if network_type == "siren":
        net_p = siren_params(
            n_enc_out, n_output_dims, n_hidden_layers, n_neurons, siren_residual
        )
    elif network_type == "mlp":
        net_p = mlp_params(
            n_enc_out, n_output_dims, n_hidden_layers, n_neurons, mlp_bias
        )
    else:
        raise ValueError(f"Unknown network_type: {network_type}")

    # ── Encoder budget ──
    encoder_budget = kan_p - net_p

    # ── Report ──
    print("=" * 60)
    print("PARAMETER MATCHING: KAN ↔ Encoder + Network")
    print("=" * 60)
    print()
    print(
        f"  KAN config:  layers={kan_layers_hidden}  grid={kan_grid_size}  order={kan_spline_order}"
    )
    print(f"  KAN params:  {kan_p:,}")
    print()
    print(f"  Network type:      {network_type.upper()}")
    print(
        f"  Network input:     {n_enc_out}  (= {n_levels} levels × {n_features_per_level} feat/level)"
    )
    print(
        f"  Network arch:      {n_enc_out} → {'×'.join([str(n_neurons)]*n_hidden_layers)} → {n_output_dims}"
    )
    print(f"  Network params:    {net_p:,}")
    print()

    if encoder_budget <= 0:
        print(f"  ⚠ Network alone ({net_p:,}) already exceeds KAN ({kan_p:,}).")
        print(f"    → Consider using fewer neurons/layers, or a larger KAN.")
        return

    print(f"  Encoder budget:    {encoder_budget:,}  (= {kan_p:,} − {net_p:,})")
    print()

    # ── Solve for log2_hashmap_size ──
    l2hs, enc_p = find_log2_hashmap_size(
        encoder_budget,
        n_pos_dims=n_input_dims,
        n_levels=n_levels,
        n_features_per_level=n_features_per_level,
        base_resolution=base_resolution,
        per_level_scale=per_level_scale,
    )

    total = net_p + enc_p
    diff = total - kan_p

    print(f"  Encoder config:")
    print(f"    n_levels             = {n_levels}")
    print(f"    n_features_per_level = {n_features_per_level}")
    print(f"    base_resolution      = {base_resolution}")
    print(f"    per_level_scale      = {per_level_scale}")
    print(f"    ──────────────────────────")
    print(f"    log2_hashmap_size    = {l2hs}")
    print(f"    encoder params       = {enc_p:,}")
    print()

    # Per-level detail
    _, level_details = encoder_params(
        n_pos_dims=n_input_dims,
        n_levels=n_levels,
        n_features_per_level=n_features_per_level,
        log2_hashmap_size=l2hs,
        base_resolution=base_resolution,
        per_level_scale=per_level_scale,
    )
    print(f"  Per-level breakdown:")
    for lvl, res, length in level_details:
        capped = " (capped)" if length == (1 << l2hs) else ""
        print(f"    Level {lvl:2d}: resolution={res:6d}, entries={length:>10,}{capped}")
    print()

    print("─" * 60)
    print(f"  TOTAL (network + encoder) = {total:,}")
    print(f"  KAN target                = {kan_p:,}")
    print(f"  Difference                = {diff:+,}  ({diff/kan_p*100:+.2f}%)")
    print("─" * 60)

    # Also show the closest match one step below
    if l2hs > 1:
        enc_p_below, _ = encoder_params(
            n_pos_dims=n_input_dims,
            n_levels=n_levels,
            n_features_per_level=n_features_per_level,
            log2_hashmap_size=l2hs - 1,
            base_resolution=base_resolution,
            per_level_scale=per_level_scale,
        )
        total_below = net_p + enc_p_below
        diff_below = total_below - kan_p
        print(f"\n  Alternative (log2_hashmap_size = {l2hs - 1}):")
        print(f"    encoder params = {enc_p_below:,}")
        print(
            f"    total          = {total_below:,}  (diff = {diff_below:+,}, {diff_below/kan_p*100:+.2f}%)"
        )

    return {
        "kan_params": kan_p,
        "network_params": net_p,
        "log2_hashmap_size": l2hs,
        "encoder_params": enc_p,
        "total_params": total,
        "difference": diff,
    }


if __name__ == "__main__":
    # ──────────────────────────────────────────────
    # Example: Match a KAN [3, 16, 16, 1] with grid_size=5
    # against SIREN + hash encoder
    # ──────────────────────────────────────────────

    print("\n")
    result = match_params(
        # KAN config
        kan_layers_hidden=[3, 16, 16, 1],
        kan_grid_size=5,
        kan_spline_order=3,
        # SIREN config
        network_type="siren",
        n_input_dims=3,
        n_output_dims=1,
        n_hidden_layers=2,
        n_neurons=16,
        # Encoder config (all except log2_hashmap_size)
        n_levels=2,
        n_features_per_level=4,
        base_resolution=16,
        per_level_scale=2.0,
    )

    print("\n\n")

    # ──────────────────────────────────────────────
    # Same KAN, but matched against an MLP
    # ──────────────────────────────────────────────
    result = match_params(
        kan_layers_hidden=[3, 16, 16, 1],
        kan_grid_size=5,
        kan_spline_order=3,
        network_type="mlp",
        n_input_dims=3,
        n_output_dims=1,
        n_hidden_layers=2,
        n_neurons=16,
        mlp_bias=False,
        n_levels=2,
        n_features_per_level=4,
        base_resolution=16,
        per_level_scale=2.0,
    )
