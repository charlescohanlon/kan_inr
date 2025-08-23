# Kolmogorov-Arnold Neural Networks (KANs) for Implicit Neural Representations (INRs) of Volumetric Data

A high-performance framework for training and benchmarking Implicit Neural Representations (INRs) using both traditional Multi-Layer Perceptrons (MLPs) and novel Kolmogorov-Arnold Networks (KANs) on volumetric scientific data.

## Project Overview

This project implements a comprehensive benchmarking system for comparing different neural network architectures for volumetric data compression and reconstruction. The key innovation is the integration of KAN architectures, which use learnable spline-based activation functions instead of fixed activations, potentially offering better expressivity for complex 3D scalar fields.

### Key Features

- **Dual Architecture Support**: Seamlessly switch between MLP and KAN architectures for comparative analysis
- **Multi-Resolution Hash Encoding**: Implements the InstantNGP-style hash encoding for efficient spatial feature extraction
- **Distributed Training**: Full support for multi-GPU training via PyTorch's DistributedDataParallel (DDP)
- **Automated Benchmarking**: Comprehensive parameter sweep capabilities with automatic resource allocation
- **HPC Integration**: Built-in PBS job submission and management for high-performance computing clusters
- **Memory-Efficient**: Automatic batch size calculation and memory management for large volumes

### Applications

- **Medical Imaging**: Compress and reconstruct CT/MRI scans
- **Scientific Visualization**: Handle simulation outputs from CFD, climate modeling, etc.
- **Volume Rendering**: Efficient storage and transmission of volumetric data
- **Neural Compression**: Achieve high compression ratios while maintaining reconstruction quality

---

## Installation

### Prerequisites
- CUDA-capable GPU (recommended)
- Python 3.8+
- PyTorch 2.0+

### Setup

```bash
# Clone the repository
git clone git@github.com:charlescohanlon/alcf_kan_inr.git
cd alcf_kan_inr

# Create conda environment (or use micromamba)
conda env create -f environment.yml
conda activate alcf_kan_inr

# Install package in development mode
pip install -e .
```

---

## Repository Structure

```
alcf_kan_inr/
├── conf/                               # Hydra configuration files
│   └── config.yaml                     # Main configuration file
├── data/                               # Sample datasets
│   └── nucleon_41x41x41_uint8.raw     # Demo dataset
├── run_scripts/                        # Utility scripts
│   ├── debug.sh                        # Debug script
│   └── reconst.sh                      # Reconstruction example
├── benchmark.py                        # Main benchmarking script
├── demo.py                             # Simple demonstration script
├── fastkan.py                          # KAN implementation
├── networks.py                         # Network architectures
├── samplers.py                         # Data sampling utilities
├── submit_jobs.py                      # HPC job submission manager
├── params.json                         # Benchmark parameters
├── params_debug.json                   # Debug parameters
├── vis.ipynb                           # Visualization notebook
├── environment.yml                     # Conda environment
└── README.md                           # This file
```

---

## Quick Start

### Simple Demo

For a quick test with the included sample data:

```bash
python demo.py
```

This will train a KAN-based INR on the small nucleon dataset and report reconstruction metrics.

---

## Running Benchmarks

The `benchmark.py` script is the main entry point for training and evaluation. It supports both single-GPU and multi-GPU configurations.

### Single GPU Execution

```bash
# Basic single-GPU run
python benchmark.py -cn config \
    params_file="params.json" \
    dataset="nucleon" \
    network_types="[mlp]" \
    epochs=20 \
    enable_pbar=True
```

### Multi-GPU Execution with torchrun

For distributed training across multiple GPUs:

```bash
# Automatically detect and use all available GPUs
torchrun --standalone --nnodes=1 --nproc_per_node=gpu \
    benchmark.py -cn config \
    params_file="params.json" \
    dataset="magnetic_reconnection" \
    network_types="[mlp,kan]" \
    hashmap_size=16 \
    epochs=50 \
    repeats=3 \
    enable_pbar=True

# Explicitly specify number of GPUs
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    benchmark.py -cn config \
    params_file="params.json" \
    dataset="richtmyer_meshkov" \
    network_types="[kan]" \
    hashmap_size=19 \
    epochs=100 \
    batch_size=100000 \
    save_mode="largest"
```

### Advanced Options

```bash
# Full parameter sweep with automatic batch sizing
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    benchmark.py -cn config \
    params_file="params.json" \
    dataset="beechnut" \
    network_types="[mlp,kan]" \
    epochs=200 \
    repeats=5 \
    safety_margin=0.8 \
    save_mode="largest" \
    output_filename="beechnut_comparison.csv" \
    ssd_dir="/local/scratch/"
```

### Key Parameters

- `params_file`: YAML/JSON file containing hyperparameter configurations
- `dataset`: Specific dataset to benchmark (filters params file entries)
- `network_types`: List of architectures to test (`[mlp]`, `[kan]`, or `[mlp,kan]`)
- `hashmap_size`: Log2 of the hash table size (e.g., 16 means 2^16 entries)
- `epochs`: Number of training epochs (overrides params file)
- `repeats`: Number of runs to average for statistical stability
- `batch_size`: Training batch size (auto-calculated if not specified)
- `safety_margin`: Memory safety factor for batch size calculation (0-1)
- `save_mode`: Save strategy (`"largest"`, `"smallest"`, or specific size like `"18"`)
- `enable_pbar`: Show progress bars during training
- `ssd_dir`: Optional fast storage for I/O acceleration

---

## HPC Job Submission

The `submit_jobs.py` script automates PBS job submission with intelligent resource allocation.

### Basic Usage

```bash
# Submit all jobs from params file
python submit_jobs.py -cn config \
    params_file="params.json" \
    network_types="[mlp,kan]"

# Dry run to preview job configurations
python submit_jobs.py -cn config \
    params_file="params.json" \
    network_types="[mlp,kan]" \
    dry_run=True \
    verbose=True
```

### Filtered Submission

```bash
# Submit only specific dataset
python submit_jobs.py -cn config \
    params_file="params.json" \
    dataset="magnetic_reconnection" \
    network_types="[kan]" \
    epochs=100 \
    repeats=3

# Submit specific hashmap size sweep
python submit_jobs.py -cn config \
    params_file="params.json" \
    dataset="richtmyer_meshkov" \
    hashmap_size=18 \
    network_types="[mlp,kan]" \
    max_jobs=10
```

### Advanced HPC Configuration

```bash
# Full production run with queue management
python submit_jobs.py -cn config \
    params_file="params_production.json" \
    network_types="[mlp,kan]" \
    repeats=5 \
    max_jobs=20 \
    wait_time=120 \
    verbose=True \
    save_mode="largest" \
    output_filename="production_results.csv"
```

### Submission Parameters

- `max_jobs`: Maximum number of jobs to keep in queue (default: 20)
- `wait_time`: Seconds to wait when queue is full (default: 120)
- `dry_run`: Preview job configurations without submitting
- `verbose`: Print detailed job information
- All `benchmark.py` parameters are also supported

### Resource Estimation

The submission script automatically:
- Estimates GPU requirements based on dataset size and model complexity
- Calculates appropriate walltime based on empirical performance data
- Selects optimal compute nodes (Sophia for ≤4 GPUs, Polaris for >4)
- Manages queue slots to avoid overwhelming the scheduler

---

## Understanding the Parameters File

The `params.json` file defines the hyperparameter search space:

```json
{
    "dataset_name": {
        "lrate": 0.001,
        "lrate_decay": 16,
        "epochs": 100,
        "n_neurons": 64,
        "n_hidden_layers": 3,
        "n_levels": 16,
        "n_features_per_level": 4,
        "log2_hashmap_size": [14, 18],  // Range for sweep
        "per_level_scale": 2.0,
        "base_resolution": "(int)cbrt(1<<log2_hashmap_size)",
        "zfp_enc": 0.0,
        "zfp_mlp": 0.0
    }
}
```

Key fields:
- `log2_hashmap_size`: Can be a single value or `[min, max]` for sweeping
- `base_resolution`: Can be an integer or dynamic calculation
- Multiple datasets can be defined in a single file

---

## Output and Results

### Result Files

Benchmarks produce CSV files with the following metrics:
- `compression_ratio`: Size reduction compared to raw data
- `avg_psnr`: Peak Signal-to-Noise Ratio (higher is better)
- `avg_ssim`: Structural Similarity Index (0-1, higher is better)
- `avg_mse`: Mean Squared Error (lower is better)
- `avg_time_per_epoch`: Training efficiency metric
- Model configuration parameters for reproducibility

### Saved Models and Reconstructions

When `save_mode` is specified:
- Models saved to: `{home_dir}/saved_models/{name}.pt`
- Reconstructions saved to: `{home_dir}/saved_reconstructions/{name}_{shape}_{dtype}.raw`

---

## 🔬 Architecture Details

### Multi-Resolution Hash Encoding
- Implements InstantNGP-style spatial hash encoding
- Multiple resolution levels capture features at different scales
- Configurable hash table size trades memory for quality

### KAN vs MLP
- **MLP**: Traditional architecture with fixed ReLU activations
- **KAN**: Learnable spline-based activations on edges
  - Uses radial basis functions to approximate B-splines
  - Potentially better for smooth, continuous fields
  - Higher computational cost but potentially better compression

### Distributed Training
- Spatial domain decomposition for multi-GPU training
- Each GPU processes a different region of the volume
- Automatic synchronization and gradient aggregation

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce `batch_size` or use `safety_margin=0.7` |
| Slow training | Increase number of GPUs or reduce `epochs` |
| Poor reconstruction | Increase `hashmap_size` or `n_hidden_layers` |
| NaN in KAN training | Enable `suppress_encoder_nan` in network config |
| Queue full errors | Reduce `max_jobs` in submission script |
| vtkOpenGL errors | Use `jupyter lab --no-browser` for remote sessions |

---

## Citation

If you use this code in your research, please cite: (TODO: replace placeholders with actual values)

```bibtex
@software{kan_inr_benchmark,
  title = {KAN-INR: Kolmogorov-Arnold Networks for Implicit Neural Representations},
  author = {[Authors]},
  year = {2024},
  url = {https://github.com/charlescohanlon/alcf_kan_inr}
}
```

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

---

## 📧 Contact

For questions or collaborations, please open an issue on GitHub or contact the maintainers directly.