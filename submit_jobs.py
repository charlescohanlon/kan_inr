"""
PBS Job Submission Script with Dynamic Resource Allocation
Submits jobs with appropriate GPU and walltime requests based on job complexity
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, fields

import hydra
from hydra.core.config_store import ConfigStore
import numpy as np
from omegaconf import DictConfig, OmegaConf

# Import everything we need from benchmark.py
from benchmark import BenchmarkConfig, RunParams, parse_filename, parse_run_params


@dataclass
class SubmissionConfig(BenchmarkConfig):
    """Extended configuration that includes submission-specific parameters"""

    # Submission-specific parameters
    max_jobs: int = 20
    wait_time: int = 10 * 60  # seconds
    dry_run: bool = False
    verbose: bool = False
    array_index_subset: Optional[List[int]] = None


# Dataset sizes parsed from filenames for quick reference
DATASET_INFO = {
    "3d_neurons_15_sept_2016": (2048, 2048, 1718, "uint16"),
    "aneurism": (256, 256, 256, "uint8"),
    "backpack": (512, 512, 373, "uint16"),
    "beechnut": (1024, 1024, 1546, "uint16"),
    "blunt_fin": (256, 128, 64, "uint8"),
    "boston_teapot": (256, 256, 178, "uint8"),
    "bunny": (512, 512, 361, "uint16"),
    "carp": (256, 256, 512, "uint16"),
    "chameleon": (1024, 1024, 1080, "uint16"),
    "christmas_tree": (512, 499, 512, "uint16"),
    "csafe_heptane": (302, 302, 302, "uint8"),
    "dns": (10240, 7680, 1536, "float64"),
    "duct": (193, 194, 1000, "float32"),
    "engine": (256, 256, 128, "uint8"),
    "foot": (256, 256, 256, "uint8"),
    "frog": (256, 256, 44, "uint8"),
    "fuel": (64, 64, 64, "uint8"),
    "hcci_oh": (560, 560, 560, "float32"),
    "hydrogen_atom": (128, 128, 128, "uint8"),
    "jicf_q": (1408, 1080, 1100, "float32"),
    "kingsnake": (1024, 1024, 795, "uint8"),
    "lobster": (301, 324, 56, "uint8"),
    "magnetic_reconnection": (512, 512, 512, "float32"),
    "marmoset_neurons": (1024, 1024, 314, "uint8"),
    "marschner_lobb": (41, 41, 41, "uint8"),
    "miranda": (1024, 1024, 1024, "float32"),
    "mri_ventricles": (256, 256, 124, "uint8"),
    "mri_woman": (256, 256, 109, "uint16"),
    "mrt_angio": (416, 512, 112, "uint16"),
    "neghip": (64, 64, 64, "uint8"),
    "neocortical_layer_1_axons": (1464, 1033, 76, "uint8"),
    "nucleon": (41, 41, 41, "uint8"),
    "pancreas": (240, 512, 512, "int16"),
    "pawpawsaurus": (958, 646, 1088, "uint16"),
    "pig_heart": (2048, 2048, 2612, "int16"),
    "present": (492, 492, 442, "uint16"),
    "prone": (512, 512, 463, "uint16"),
    "richtmyer_meshkov": (2048, 2048, 1920, "uint8"),
    "rotstrat_temperature": (4096, 4096, 4096, "float32"),
    "shockwave": (64, 64, 512, "uint8"),
    "silicium": (98, 34, 34, "uint8"),
    "skull": (256, 256, 256, "uint8"),
    "spathorhynchus": (1024, 1024, 750, "uint16"),
    "stag_beetle": (832, 832, 494, "uint16"),
    "statue_leg": (341, 341, 93, "uint8"),
    "stent": (512, 512, 174, "uint16"),
    "synthetic_truss_with_five_defects": (1200, 1200, 1200, "float32"),
    "tacc_turbulence": (256, 256, 256, "float32"),
    "tooth": (103, 94, 161, "uint8"),
    "vertebra": (512, 512, 512, "uint16"),
    "vis_male": (128, 256, 256, "uint8"),
    "woodbranch": (2048, 2048, 2048, "uint16"),
    "zeiss": (680, 680, 680, "uint8"),
}


class JobSubmissionManager:
    def __init__(self, cfg: SubmissionConfig):
        self.cfg = cfg
        self.max_queued_jobs = cfg.max_jobs
        self.wait_time = cfg.wait_time
        self.dry_run = cfg.dry_run
        self.verbose = cfg.verbose
        self.home_dir = Path(cfg.home_dir)
        self.log_dir = self.home_dir / "kan_inr" / "logs"
        if self.cfg.dataset is not None:
            self.log_dir /= self.cfg.dataset
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def get_dataset_info(self, dataset_name: str) -> Tuple[Tuple[int, int, int], str]:
        """Get dataset shape and type from pre-parsed info or by finding the file"""

        # First check our pre-parsed dataset info
        if dataset_name in DATASET_INFO:
            shape, dtype = DATASET_INFO[dataset_name][:3], DATASET_INFO[dataset_name][3]
            return shape, dtype

        # If not found, try to find the actual file
        data_dir = self.home_dir / self.cfg.data_path
        if data_dir.exists():
            for dataset_file in os.listdir(data_dir):
                if dataset_file.startswith(dataset_name):
                    data_path = data_dir / dataset_file
                    _, shape, dtype = parse_filename(data_path)
                    return shape, dtype

        raise ValueError(f"Unknown dataset: {dataset_name}")

    def estimate_resources(self, params: RunParams) -> Tuple[int, str]:
        """
        Estimate required resources based on job parameters using A100 specifications.

        Based on empirical data from beechnut dataset runs:
        - KAN models are ~5-8x slower than MLP models
        - Larger hashmap sizes increase runtime exponentially for h>16
        - Complex networks (more layers/neurons) significantly increase runtime
        - Memory requirements scale with dataset size and model complexity

        Returns:
            (num_gpus, walltime_str) - Number of GPUs and walltime in HH:MM:SS format
        """
        # Get dataset information
        shape, dtype = self.get_dataset_info(params.dataset_name)

        # Calculate dataset size in GB
        dtype_sizes = {
            "uint8": 1,
            "int8": 1,
            "uint16": 2,
            "int16": 2,
            "float32": 4,
            "int32": 4,
            "float64": 8,
            "int64": 8,
        }
        bytes_per_element = dtype_sizes.get(dtype, 4)
        dataset_size_gb = (shape[0] * shape[1] * shape[2] * bytes_per_element) / (
            1024**3
        )
        total_voxels = shape[0] * shape[1] * shape[2]

        # Base runtime estimation (seconds per epoch)
        # Based on empirical data from beechnut (1.6B voxels)
        beechnut_voxels = 1621098496  # Reference dataset
        voxel_ratio = total_voxels / beechnut_voxels

        # Base time per epoch (calibrated from logs)
        if "kan" in params.network_type:
            # KAN: ~87-88 seconds per epoch for beechnut with simple network
            base_time_per_epoch = 87.0 * voxel_ratio
        else:  # mlp
            # MLP: ~10-11 seconds per epoch for beechnut with simple network
            base_time_per_epoch = 10.5 * voxel_ratio

        # Network complexity multiplier
        # Simple network baseline: 1 hidden layer, 16 neurons
        network_complexity = (params.n_hidden_layers * params.n_neurons) / 16.0

        # Adjust for very complex networks (exponential scaling observed)
        if params.n_hidden_layers >= 4:
            network_complexity *= 1.5 ** (params.n_hidden_layers - 3)

        # Hashmap size impact (exponential for large sizes)
        hashmap_multiplier = 1.0
        if params.log2_hashmap_size <= 14:
            hashmap_multiplier = 1.0 + (params.log2_hashmap_size - 10) * 0.02
        elif params.log2_hashmap_size <= 17:
            hashmap_multiplier = 1.08 + (params.log2_hashmap_size - 14) * 0.05
        else:  # h >= 18
            # Significant jump observed for h>=18, especially with complex networks
            hashmap_multiplier = 1.23 * (1.3 ** (params.log2_hashmap_size - 17))

        # Calculate total training time
        time_per_epoch = base_time_per_epoch * network_complexity * hashmap_multiplier
        training_time = time_per_epoch * params.epochs

        # Add reconstruction and metric computation overhead (~5-10% of training)
        total_time = training_time * 1.1

        # Account for repeats
        total_time *= self.cfg.repeats

        # Memory-based GPU allocation
        # Estimate model memory footprint
        hashmap_memory_gb = (
            2**params.log2_hashmap_size
            * params.n_features_per_level
            * params.n_levels
            * 2
        ) / 1024**3

        # Network parameter memory (rough estimate)
        if "kan" in params.network_type:
            # KAN networks use more memory per parameter
            params_memory_gb = (params.n_neurons * params.n_hidden_layers * 100 * 4) / (
                1024**3
            )
        else:
            params_memory_gb = (params.n_neurons * params.n_hidden_layers * 50 * 4) / (
                1024**3
            )

        # Working memory for batch processing (includes gradients, optimizer state)
        working_memory_gb = dataset_size_gb * 3  # Conservative estimate

        # Total memory requirement
        total_memory_gb = (
            hashmap_memory_gb + params_memory_gb + working_memory_gb + 5
        )  # +5GB overhead

        # Determine GPU count based on memory (A100 has 40GB)
        memory_per_gpu = 35  # Conservative limit (leave headroom)
        num_gpus = max(1, int(np.ceil(total_memory_gb / memory_per_gpu)))

        if "kan" in params.network_type:
            total_time *= 15  # KAN models are slower
        else:
            total_time *= 2  # MLP models are faster

        seconds_per_hour = 3600
        if total_time > seconds_per_hour * 2:
            num_gpus = max(2, num_gpus)
        if total_time > seconds_per_hour * 4:
            num_gpus = max(4, num_gpus)
        if total_time > seconds_per_hour * 8:
            num_gpus = 8

        # Adjust time estimate for multi-GPU (not perfect scaling)
        if num_gpus > 1:
            # Assume 80% scaling efficiency per additional GPU
            scaling_factor = 1 + (num_gpus - 1) * 0.8
            total_time = total_time / scaling_factor

        # Convert to walltime string with appropriate padding
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)

        # Round up to next 15-minute increment for scheduler efficiency
        if seconds > 0:
            minutes += 1
        if minutes % 15 != 0:
            minutes = ((minutes // 15) + 1) * 15
            if minutes >= 60:
                hours += minutes // 60
                minutes = minutes % 60

        # Cap at 24 hours and use appropriate format
        if hours >= 24:
            walltime = "24:00:00"
        else:
            walltime = f"{hours:02d}:{minutes:02d}:00"

        # Special cases for known difficult configurations
        # Very complex KAN networks with large hashmaps
        if (
            "kan" in params.network_type
            and params.n_hidden_layers >= 4
            and params.log2_hashmap_size >= 18
        ):
            # These take 6000+ seconds empirically
            if hours < 3:
                walltime = "03:00:00"

        # Quick MLP runs shouldn't request too much time
        if params.network_type == "mlp" and params.n_hidden_layers <= 2:
            if hours > 2:
                walltime = "02:00:00"

        # Minimum walltime of 30 minutes for stability
        if hours == 0 and minutes < 30:
            walltime = "00:30:00"

        return num_gpus, walltime

    def generate_pbs_script(
        self,
        run_index: int,
        params: RunParams,
        num_gpus: int,
        walltime: str,
    ) -> str:
        """Generate a customized PBS script for this specific job"""
        # Generate a descriptive job name (PBS limits to 15 chars)
        job_name = f"{params.network_type[:1]}_{params.dataset_name[:3]}_{run_index}"

        if params.log2_hashmap_size > 0:
            hashmap_size = "2^" + str(params.log2_hashmap_size)
        else:
            hashmap_size = "N/A"

        # Cast SubmissionConfig superclass to BenchmarkConfig for argument parsing
        submission_fields = {f.name for f in fields(SubmissionConfig)}
        benchmark_fields = {f.name for f in fields(BenchmarkConfig)}
        override_args = []
        for key in submission_fields:
            # If key exists in BenchmarkConfig and is not None, add to overrides
            if key in benchmark_fields and self.cfg[key] is not None:
                override_args.append(f'{key}={str(self.cfg[key]).replace(" ", "")}')

        override_str = " ".join(override_args).replace("'", '"')
        script = f"""#!/bin/bash
# ---------- PBS DIRECTIVES ----------
#PBS -A insitu
#PBS -q by-gpu
#PBS -l select=1:ncpus={num_gpus*8}:gputype=A100:system=sophia
#PBS -l walltime={walltime}
#PBS -l filesystems=home:grand
#PBS -N {job_name}
#PBS -o {self.log_dir}/
#PBS -e {self.log_dir}/
#PBS -m n
# -----------------------------------

# Job parameters
export PBS_ARRAY_INDEX={run_index}

source {self.home_dir}/miniconda3/etc/profile.d/conda.sh 
cd {self.home_dir}/kan_inr
conda activate kan_inr

# Get the number of GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "========================================="
echo "Job Information:"
echo "  Host: $(hostname)"
echo "  Job ID: $PBS_JOBID"
echo "  Job Array Index: $PBS_ARRAY_INDEX"
echo "  Number of GPUs detected: $NUM_GPUS"
echo "  Requested GPUs: {num_gpus}"
echo "  Walltime: {walltime}"
echo "  System: sophia"
echo "  Override args: {override_str}"
echo "========================================="
echo "Model Configuration:"
echo "  Model type: {params.network_type}"
echo "  Dataset: {params.dataset_name}"
echo "  Hashmap size: {hashmap_size}"
echo "  Epochs: {params.epochs}"
echo "  Hidden layers: {params.n_hidden_layers}"
echo "  Neurons per layer: {params.n_neurons}"
echo "  Repeats: {self.cfg.repeats}"
echo "========================================="
echo "Starting at: $(date)"

# Run the benchmark
torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS \\
    benchmark.py -cn config {override_str}

echo "Completed at: $(date)"
"""
        return script

    def submit_job(self, run_index: int, params: RunParams) -> bool:
        """Submit a single job and return success status"""
        num_gpus, walltime = self.estimate_resources(params)

        data_shape, data_type = self.get_dataset_info(params.dataset_name)

        if self.verbose or self.dry_run:
            print(f"\nJob {run_index}: {params.network_type} on {params.dataset_name}")
            print(f"  Dataset shape: {data_shape} ({data_type})")
            print(f"  Hashmap size: 2^{params.log2_hashmap_size}")
            print(
                f"  Network: {params.n_hidden_layers} layers x {params.n_neurons} neurons"
            )
            print(f"  Epochs: {params.epochs}, Repeats: {self.cfg.repeats}")
            print(f"  Resources: {num_gpus} GPUs, {walltime} walltime")

        if self.dry_run:
            return True

        # Generate PBS script
        script_content = self.generate_pbs_script(run_index, params, num_gpus, walltime)

        # Submit via stdin to qsub
        try:
            result = subprocess.run(
                ["qsub"],
                input=script_content,
                text=True,
                capture_output=True,
                check=True,
            )
            job_id = result.stdout.strip()

            if self.verbose:
                print(f"  Submitted: {job_id}")

            return True

        except subprocess.CalledProcessError as e:
            print(f"ERROR submitting job {run_index}: {e.stderr}")
            return False

    def get_queued_jobs(self) -> int:
        """Get the current number of queued jobs for the user"""
        try:
            result = subprocess.run(
                ["qstat", "-u", os.environ.get("USER", "")],
                capture_output=True,
                text=True,
                check=True,
            )
            # Count non-header lines
            lines = result.stdout.strip().split("\n")
            return max(0, len(lines) - 5)  # Skip header lines
        except:
            return 0

    def wait_for_slot(self, must_wait=False):
        """Wait for a queue slot to become available"""
        while True:
            current_jobs = self.get_queued_jobs()
            if current_jobs < self.max_queued_jobs and not must_wait:
                return

            print(
                f"Queue full ({current_jobs + 1}/{self.max_queued_jobs}). "
                f"Waiting {self.wait_time / 60:.1f} minutes before retrying..."
            )
            time.sleep(self.wait_time)
            must_wait = False  # Only force wait once

    def run(self):
        """Main execution loop"""
        # Use the parse_run_params function from benchmark.py
        runs_list = parse_run_params(self.cfg)
        total_runs = len(runs_list)

        print(f"Total runs to submit: {total_runs}")
        print(f"Repeats per run: {self.cfg.repeats}")

        if total_runs == 0:
            print("No runs found with specified filters")
            return

        # Print summary of datasets if verbose
        if self.verbose:
            datasets = set(r.dataset_name for r in runs_list)
            print(f"Datasets to process: {', '.join(sorted(datasets))}")

        # Initialize counters
        submitted = 0
        failed = 0

        # Submit jobs
        for run_index, params in enumerate(runs_list):
            if self.cfg.array_index_subset is not None:
                if run_index not in self.cfg.array_index_subset:
                    if self.verbose:
                        print(f"Skipping job {run_index} (not in subset)")
                    continue
            if not self.dry_run:
                self.wait_for_slot()  # Check if max_queued_jobs is reached

            success = False
            while not success:  # Retry until successful
                success = self.submit_job(run_index, params)
                if not self.dry_run:
                    self.wait_for_slot(must_wait=not success)

            if not self.dry_run:
                # Small delay to avoid overwhelming the scheduler
                time.sleep(1)

        # Final summary
        print("\n=== SUBMISSION SUMMARY ===")
        print(f"Total runs: {total_runs}")
        print(f"Submitted: {submitted}")
        print(f"Failed: {failed}")


# Register the config schema with Hydra as a structured config
cs = ConfigStore.instance()

# Override schema to interpret "--config-name config" as a SubmissionConfig (not a BenchmarkConfig)
cs.store(name="config", node=SubmissionConfig)


@hydra.main(version_base="1.3", config_path="conf")
def main(cfg: SubmissionConfig):
    """Main entry point with Hydra configuration"""

    # Print configuration if verbose
    if cfg.verbose:
        print("Submission Configuration:")
        print(f"  Max queued jobs: {cfg.max_jobs}")
        print(f"  Wait time: {cfg.wait_time} seconds")
        print(f"  Dry run: {cfg.dry_run}")
        print(f"  Verbose: {cfg.verbose}")
        if cfg.dataset:
            print(f"  Dataset filter: {cfg.dataset}")
        if cfg.hashmap_size:
            print(f"  Hashmap size filter: {cfg.hashmap_size}")
        if cfg.epochs:
            print(f"  Epochs override: {cfg.epochs}")
        print()

    # Create and run the submission manager
    manager = JobSubmissionManager(cfg)

    try:
        manager.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
