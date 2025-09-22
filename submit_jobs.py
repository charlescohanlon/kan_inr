from contextlib import nullcontext
from genericpath import exists
import multiprocessing
import os
import subprocess
import time
import math
import pickle
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass, fields
import uuid
import pandas as pd

import hydra
from hydra.core.config_store import ConfigStore

from benchmark import BenchmarkConfig, RunParams, parse_run_params


@dataclass
class SubmissionConfig(BenchmarkConfig):
    """Extended configuration that includes submission-specific parameters"""

    # Submission-specific parameters
    max_jobs: int = 18
    wait_time: int = 10 * 60
    dry_run: bool = False
    verbose: bool = False
    array_index_subset: Optional[List[int]] = None
    num_minutes_per_job: float = 3  # Expected minutes per job for test epoch
    compute_epoch_time_only: bool = False
    num_workers: int = 4  # Number of parallel workers test_epoch test epochs
    time_scaler: float = 1  # Scaling factor for estimating time

    # Directories for logs and memoization
    log_dir: Path = Path(BenchmarkConfig.home_dir) / "kan_inr" / "logs"
    memo_dir: Path = Path(BenchmarkConfig.home_dir) / "memo"
    test_epoch_dir: Path = log_dir / "test_epochs"
    test_epoch_memo_file: Path = memo_dir / "test_epoch_memo.pkl"
    use_memoization: bool = True  # Use memoization file for test epochs

    # Directory for operations with aggregated results
    aggregated_results_dir: Path = Path(BenchmarkConfig.home_dir) / "aggregated_results"
    aggregate_file: Optional[Path] = (
        None  # If None, will be auto-named the same as params file
    )
    filter_from_aggregated: bool = True  # Filter jobs already in aggregate file
    aggregate_results: bool = True  # Aggregate results after submission

    # Retry any failed jobs found in log directory
    retry_failed: bool = True

    # Option to filter out runs already running or queued
    filter_running_or_queued: bool = True


def get_test_epoch_key(cfg: SubmissionConfig, params: RunParams) -> str:
    """Get a unique key for test epoch results."""
    return str(
        params.epoch_time_hash(
            safety_margin=cfg.safety_margin,
            dataset_name=params.dataset_name,
            ssd_dir_provided=cfg.ssd_dir is not None,
        )
    )


def load_test_epoch_memoization(cfg: SubmissionConfig) -> dict:
    """Load existing test epoch results from memoization file"""
    memo_file_path = cfg.test_epoch_memo_file
    if memo_file_path.exists():
        if cfg.verbose:
            print(f"Loading test epochs from {memo_file_path}")
        try:
            with open(memo_file_path, "rb") as f:
                return pickle.load(f)
        except (pickle.PickleError, FileNotFoundError, EOFError) as e:
            print(f"Warning: could not load test epoch memoization file: {e}")
            return {}
    if cfg.verbose:
        print(f"No test epoch memoization file found at {memo_file_path}")
    return {}


def save_test_epoch_memoization(
    cfg: SubmissionConfig, test_epoch_dict_proxy, test_epoch_lock
) -> bool:
    """Save test_epoch results to memoization file when new ones computed"""
    with test_epoch_lock:
        try:
            with open(cfg.test_epoch_memo_file, "wb") as f:
                pickle.dump(dict(test_epoch_dict_proxy), f)
            return True
        except (pickle.PickleError, OSError) as e:
            print(f"Warning: Could not save test epoch memoization file: {e}")
            return False


def get_override_args(cfg: SubmissionConfig) -> List[str]:
    """
    Get command-line arguments for overriding default config values
    by only including fields that exist in BenchmarkConfig.
    """
    submission_fields = {f.name for f in fields(SubmissionConfig)}
    benchmark_fields = {f.name for f in fields(BenchmarkConfig)}
    override_args = []
    for key in submission_fields:
        if key in benchmark_fields and cfg[key] is not None:
            override_args.append(f'{key}={str(cfg[key]).replace(" ", "")}')
    return override_args


def submit_test_epoch_job(
    cfg: SubmissionConfig, run_indices: List[int], worker_id: int, output_file: Path
) -> str:
    """Create a bash script for testing single epochs"""
    # Estimate walltime for test epoch runs
    time_in_minutes = max(len(run_indices) * cfg.num_minutes_per_job, 5)
    walltime = f"{int(time_in_minutes // 60):02d}:{int(time_in_minutes % 60):02d}:00"
    job_name = output_file.stem

    override_args = get_override_args(cfg)
    override_args.append("test_epoch_run=True")  # Ensure test epoch mode
    override_str = " ".join(override_args).replace("'", '"')

    script = f"""#!/bin/bash
# ---------- PBS DIRECTIVES ----------
#PBS -A insitu
#PBS -q by-gpu
#PBS -l select=1:ncpus=8:gputype=A100:system=sophia
#PBS -l walltime={walltime}
#PBS -l filesystems=home:grand
#PBS -N {job_name}
#PBS -o /dev/null
#PBS -e /dev/null
#PBS -m n
# -----------------------------------

source /grand/insitu/cohanlon/miniconda3/etc/profile.d/conda.sh 
cd /grand/insitu/cohanlon/kan_inr
conda activate kan_inr
for run_index in {' '.join(map(str, run_indices))}; do
    START_TIME=$(date +%s.%N)
    PBS_ARRAY_INDEX=$run_index python benchmark.py -cn config {override_str}
    END_TIME=$(date +%s.%N)
    ELAPSED=$(echo "$END_TIME - $START_TIME" | bc)
    echo "run_index: $run_index, time_elapsed: $ELAPSED" >> {output_file}
done
"""
    try:
        # Submit script to PBS
        result = subprocess.run(
            ["qsub"],
            input=script,
            capture_output=True,
            text=True,
            check=True,
        )
        job_id = result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error submitting test epoch job: {e.stderr}")
        raise e

    return job_id


def get_pbs_walltime_and_gpus(
    cfg: SubmissionConfig, params: RunParams, epoch_time: float
) -> str:
    """
    Estimate walltime based on empirical test epoch results.
    """
    # Calculate total estimated time
    # NOTE: epoch_time includes other overhead which pads the estimate
    # +1 for reconstruction phase in real runs
    total_time = (params.epochs + 1) * cfg.repeats * epoch_time * cfg.time_scaler

    # Adjust for extra GPUs based on time thresholds
    num_gpus = 1
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

    # Cap at 24 hours
    if hours >= 24:
        walltime = "24:00:00"
    else:
        walltime = f"{hours:02d}:{minutes:02d}:00"

    # Minimum walltime of 30 minutes for stability
    if hours == 0 and minutes < 30:
        walltime = "00:30:00"

    return walltime, num_gpus


def submit_pbs_job(
    cfg: SubmissionConfig,
    run_index: int,
    params: RunParams,
    epoch_time: float,
    output_dir: Optional[Path],
    print_lock=None,
    worker_id: int = 0,
) -> str:
    walltime, num_gpus = get_pbs_walltime_and_gpus(cfg, params, epoch_time)
    job_name = f"{params.network_type[:2]}_{params.dataset_name[:3]}_{run_index}"

    with print_lock or nullcontext():
        print(f"\nJob {run_index}:")
        print("  Run Parameters")
        for field in fields(RunParams):
            value = getattr(params, field.name)
            print(f"    {field.name}: {value}")
        print(f"  Test epoch time: {epoch_time:.2f} seconds")
        print(f"  Resources: {num_gpus} GPUs, {walltime} walltime")
        print(f"  Job name: {job_name}")

    if cfg.dry_run:
        return "dry_run"

    # Get override args for submission script
    override_args = get_override_args(cfg)
    override_str = " ".join(override_args).replace("'", '"')

    script = f"""#!/bin/bash
# ---------- PBS DIRECTIVES ----------
#PBS -A insitu
#PBS -q by-gpu
#PBS -l select={num_gpus}:ncpus={num_gpus*8}:gputype=A100:system=sophia
#PBS -l walltime={walltime}
#PBS -l filesystems=home:grand
#PBS -N {job_name}
#PBS -o {output_dir}
#PBS -e {output_dir}
# -----------------------------------

# Job parameters
export PBS_ARRAY_INDEX={run_index}

source {cfg.home_dir}/miniconda3/etc/profile.d/conda.sh
cd {cfg.home_dir}/kan_inr
conda activate kan_inr

# Get the number of GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "========================================="
echo "Job Information:"
echo "  Host: $(hostname)"
echo "  Job ID: $PBS_JOBID"
echo "  Job Array Index: $PBS_ARRAY_INDEX"
echo "  Number of GPUs detected: $NUM_GPUS"
echo "  Epoch time: {epoch_time:.2f} seconds"
echo "  Walltime: {walltime}"
echo "  System: sophia"
echo "  Override args: {override_str}"
echo "========================================="
echo "Starting at: $(date)"

torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS \\
    benchmark.py -cn config {override_str}

echo "Completed at: $(date)"
"""

    def _wait_for_slot(must_wait=False):
        """Wait for a queue slot to become available"""

        while True:
            current_jobs = get_queued_jobs()
            if current_jobs < cfg.max_jobs and not must_wait:
                return

            if cfg.verbose and worker_id == 0:
                print(
                    f"\nQueue full ({current_jobs} jobs), waiting {cfg.wait_time // 60} minutes..."
                )
            time.sleep(cfg.wait_time)
            must_wait = False  # Only force wait once

    # Submit via stdin to qsub, wait for success (will fail if queue is full)
    while True:
        try:
            _wait_for_slot()
            proc = subprocess.run(
                ["qsub"],
                input=script,
                capture_output=True,
                text=True,
                check=True,  # Raise error on non-zero exit code
            )
            job_id = proc.stdout.strip()
            return job_id
        except subprocess.CalledProcessError as e:
            _wait_for_slot(must_wait=True)


def get_queued_jobs(return_num_running=False) -> int:
    """Get the current number of queued jobs for the user"""

    def _is_job_line(line: str) -> bool:
        return line and line[0].isdigit()

    def _is_running_job_line(line: str) -> bool:
        return _is_job_line(line) and " R " in line

    try:
        result = subprocess.run(
            ["qstat", "-u", os.environ.get("USER", "")],
            capture_output=True,
            text=True,
            check=True,
        )
        # Count non-header lines
        lines = result.stdout.strip().split("\n")
        # Count number of running jobs
        num_jobs = sum(1 for l in lines if _is_job_line(l))
        if return_num_running:
            num_running = sum(1 for l in lines if _is_running_job_line(l))
            return num_jobs, num_running
        return num_jobs
    except:
        return 0, 0


def create_dry_run_test_epoch_file(output_file: Path, compute_indices: List[int]):
    """Create a fake test epoch file for dry run mode"""
    fake_time = -1.0
    with open(output_file, "w") as f:
        for run_index in compute_indices:
            f.write(f"run_index: {run_index}, time_elapsed: {fake_time}\n")


def compute_and_submit_job_worker(
    cfg: SubmissionConfig,
    worker_id: int,
    jobs_to_run: List[Tuple[int, RunParams]],
    test_epoch_lock,
    test_epoch_dict: dict,
    print_lock,
    output_dir: Path,
):
    test_epoch_job_id = None
    jobs_run = set()
    try:
        # Then submit the test epoch script for runs needing computation
        job_idx_to_run = [run_index for run_index, _ in jobs_to_run]
        uid = str(uuid.uuid4())[:4]
        test_epoch_file = cfg.test_epoch_dir / f"w{worker_id}_{uid}.txt"
        if not cfg.dry_run:
            test_epoch_job_id = submit_test_epoch_job(
                cfg, job_idx_to_run, worker_id, test_epoch_file
            )
            while not test_epoch_file.exists():
                # Wait for file to be created (first test epoch to finish)
                time.sleep(30)
        else:
            create_dry_run_test_epoch_file(test_epoch_file, job_idx_to_run)

        while len(jobs_run) < len(job_idx_to_run):
            with open(test_epoch_file, "r") as f:
                lines = f.readlines()

            for line in lines:
                if not "run_index" in line or not "time_elapsed" in line:
                    continue

                parts = line.strip().split(",")
                run_index = int(parts[0].split(":")[1].strip())
                time_elapsed = float(parts[1].split(":")[1].strip())
                # Find corresponding RunParams
                params = next(p for i, p in jobs_to_run if i == run_index)

                # Save new test epoch results
                key = get_test_epoch_key(cfg, params)
                with test_epoch_lock:
                    test_epoch_dict[key] = time_elapsed

                if (run_index in jobs_run) or cfg.compute_epoch_time_only:
                    continue

                submit_pbs_job(
                    cfg,
                    run_index,
                    params,
                    time_elapsed,
                    output_dir,
                    print_lock,
                    worker_id,
                )
                jobs_run.add(run_index)

            if len(jobs_run) < len(job_idx_to_run):
                time.sleep(30)  # Wait for file to update
    except KeyboardInterrupt:
        # (Attempt) to terminate test epoch job
        if test_epoch_job_id is not None:
            subprocess.run(["qdel", test_epoch_job_id])
    finally:
        # Clean up test epoch file
        if test_epoch_file.exists():
            test_epoch_file.unlink()


def create_run_signature(cfg: BenchmarkConfig, params: RunParams) -> str:
    """
    Create a signature string that uniquely identifies a run based on
    the parameters that are saved to the CSV results file.
    """
    # Formatting function
    form = lambda x: str(float(x)) if not isinstance(x, str) else x

    # Handle KAN parameters
    if params.kan_params:
        kan_sig = (
            f"{form(params.kan_params.grid_radius)}_{form(params.kan_params.num_grids)}"
        )
    else:
        kan_sig = "None_None"

    # Create signature from key identifying parameters
    signature = (
        "_".join(
            map(
                form,
                [
                    params.dataset_name,
                    params.network_type,
                    params.epochs,
                    params.n_neurons,
                    params.n_hidden_layers,
                    params.n_levels,
                    params.n_features_per_level,
                    params.per_level_scale,
                    params.log2_hashmap_size,
                    params.base_resolution,
                    cfg.repeats,
                ],
            )
        )
        + f"_{kan_sig}"
    )

    return signature


def create_csv_row_signature(row: pd.Series, cfg: BenchmarkConfig) -> str:
    """
    Create a signature string from a CSV row that matches create_run_signature.
    """

    # Formatting function
    def form(x):
        if pd.isna(x):
            return "None"
        return str(float(x)) if not isinstance(x, str) else x

    # NOTE: not using any kind of step parameters here
    kan_sig = f"{form(row.get('kan_grid_radius', 'None'))}_{form(row.get('kan_num_grids', 'None'))}"
    signature = (
        "_".join(
            map(
                form,
                [
                    row["dataset_name"],
                    row["network_type"],
                    row["epoch_count"],
                    row["num_neurons"],
                    row["num_hidden_layers"],
                    row["num_levels"],
                    row["num_features_per_level"],
                    row["per_level_scale"],
                    row["log2_hashmap_size"],
                    row["base_resolution"],
                    cfg.repeats,
                ],
            )
        )
        + f"_{kan_sig}"
    )

    return signature


def filter_from_aggregate(
    cfg: SubmissionConfig, aggregate_file: Path, runs: List[Tuple[int, RunParams]]
):
    """
    Filter runs that have already been completed by comparing against aggregate results.

    Uses an efficient signature-based approach:
    1. Precompute signatures for all rows in aggregate_df (O(n))
    2. Create signatures for input runs and check against the set (O(1) per run)

    Args:
        cfg: Submission configuration
        aggregate_file: Path to aggregate results CSV
        runs: List of (index, RunParams) tuples to filter

    Returns:
        List of runs not found in the aggregate file
    """
    aggregate_df = pd.read_csv(aggregate_file)

    # Precompute signatures for all existing results (O(n))
    existing_signatures = set()
    for _, row in aggregate_df.iterrows():
        signature = create_csv_row_signature(row, cfg)
        existing_signatures.add(signature)

    def _run_not_in_aggregate(run: Tuple[int, RunParams]) -> bool:
        """Check if a run is not in the aggregate results (O(1) lookup)."""
        _, params = run
        run_signature = create_run_signature(cfg, params)
        return run_signature not in existing_signatures

    return [run for run in runs if _run_not_in_aggregate(run)]


def filter_running_or_queued(runs: List[Tuple[int, RunParams]]):
    """
    Filter runs already running or queued using PBS qstat.
    """
    current_jobs = get_queued_jobs()
    if current_jobs == 0:
        return runs

    result = subprocess.run(
        ["qstat", "-u", os.environ.get("USER", "")],
        capture_output=True,
        text=True,
        check=True,
    )
    idx_to_be_filtered = set()
    lines = result.stdout.strip().split("\n")
    for line in lines:
        if not line or not line[0].isdigit():  # Skip header or empty lines
            continue
        # Find Jobname
        parts = line.split()
        job_name = parts[3]

        # By convention the PBS_ARRAY_INDEX is at the end of the job name
        parts = job_name.split("_")
        if not parts[-1].isdigit():
            continue  # Skip non-array jobs
        array_index = int(parts[-1])
        idx_to_be_filtered.add(array_index)

    def _keep(run: Tuple[int, RunParams]) -> bool:
        """Check if a run is not currently running (O(1) lookup)."""
        run_index, _ = run
        return run_index not in idx_to_be_filtered

    filtered_runs = [run for run in runs if _keep(run)]
    return filtered_runs


def aggregate_results(cfg: SubmissionConfig, aggregate_file: Path):
    # Wait for all jobs to complete before aggregating
    csv_dir = Path(cfg.output_dir)
    csv_files = list(csv_dir.glob("*.csv"))
    if not csv_files:
        print(f"No result CSV files found to aggregate in {csv_dir}")
        return

    # Load and concatenate all CSV files
    df_list = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if df.empty:
            continue
        df_list.append(df)

    # Include existing aggregate file if it exists
    if aggregate_file.exists():
        existing_df = pd.read_csv(aggregate_file)
        df_list.insert(0, existing_df)

    if not df_list:
        print("No valid data found in CSV files.")
        return

    aggregated_df = pd.concat(df_list, ignore_index=True)

    # Save the aggregated results
    aggregated_df.to_csv(aggregate_file, index=False)
    if cfg.verbose:
        print(f"Aggregated results saved to {aggregate_file}")

    # Clean up original CSV files
    if cfg.verbose:
        print("Cleaning up individual CSV files...")
    for csv_file in csv_files:
        try:
            csv_file.unlink()
        except Exception as e:
            print(f"Warning: Could not delete {csv_file}: {e}")


def retry_failed_jobs(
    cfg: SubmissionConfig,
    log_dir: Path,
    runs_list: List[Tuple[int, RunParams]],
    test_epoch_dict: dict,
):
    failed_runs = []
    for er_file in log_dir.glob("*.ER"):
        if er_file.stat().st_size == 0:
            continue

        # Check if the error is due to out of memory error
        with open(er_file, "r") as f:
            er_contents = f.read()
            oom_error = "memory" in er_contents.lower()

        ou_file = er_file.with_suffix(".OU")

        # Find the run index from the .OU file
        if not ou_file.exists():
            print(f"Warning: .OU file not found for {er_file}, skipping")
            continue

        # Look for "  Job Array Index: <index>" line
        with open(ou_file, "r") as f:
            for line in f:
                if not "Job Array Index" in line:
                    continue
                parts = line.strip().split(":")
                run_index = int(parts[-1].strip())
                failed_runs.append(run_index)
                er_file.unlink()  # Remove .ER file
                ou_file.unlink()  # Remove .OU file
                break
    if len(failed_runs) > 0:
        print(f"Retrying {len(failed_runs)} failed runs: {failed_runs}")
        retry_runs = [(i, p) for i, p in runs_list if i in set(failed_runs)]
        for run_index, params in retry_runs:
            key = get_test_epoch_key(cfg, params)
            epoch_time = test_epoch_dict[key]
            if oom_error:
                # NOTE: not re-calculating epoch time, we're hoping smaller
                # batch size doesn't result in exceeding job walltime
                params.safety_margin -= 0.1
            submit_pbs_job(cfg, run_index, params, epoch_time, log_dir)


# Register the config schema with Hydra as a structured config
cs = ConfigStore.instance()

# Override schema to interpret "--config-name config" as a SubmissionConfig (not a BenchmarkConfig)
cs.store(name="config", node=SubmissionConfig)


@hydra.main(version_base="1.3", config_path="conf")
def main(cfg: SubmissionConfig):
    """Main entry point."""

    if cfg.batch_size is not None:
        print(
            "Warning: Overriding batch_size will affect test_epoch estimate memoization."
        )
    if cfg.safety_margin != BenchmarkConfig.safety_margin:
        print(
            f"Warning: Using non-default safety_margin ({cfg.safety_margin}) "
            "will affect test_epoch estimate memoization."
        )

    if cfg.verbose:
        print("Submission Configuration:")
        print(f"  Max queued jobs: {cfg.max_jobs}")
        print(f"  Queue Job Wait time: {cfg.wait_time} seconds")
        print(f"  Dry run: {cfg.dry_run}")
        print(f"  Verbose: {cfg.verbose}")
        if cfg.dataset:
            print(f"  Dataset filter: {cfg.dataset}")
        if cfg.hashmap_size:
            print(f"  Hashmap size filter: {cfg.hashmap_size}")
        if cfg.epochs:
            print(f"  Epochs override: {cfg.epochs}")
        print()

    # Ensure necessary directories exist
    cfg.memo_dir.mkdir(parents=True, exist_ok=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    cfg.test_epoch_dir.mkdir(parents=True, exist_ok=True)

    # Parse run parameters from config
    runs_list = parse_run_params(cfg)

    # Before filtering runs
    if cfg.verbose:
        before_len = len(runs_list)
        print(f"Total starting runs found: {before_len}")

    # 1. Filter by array_index_subset if specified
    if cfg.array_index_subset is not None:
        runs_list = [
            (i, r) for i, r in enumerate(runs_list) if i in cfg.array_index_subset
        ]
        if cfg.verbose:
            print(
                f"Filtered to {len(runs_list)} runs by array_index_subset {cfg.array_index_subset}"
            )
            before_len = len(runs_list)  # Update before_len for next filtering
    else:
        runs_list = list(enumerate(runs_list))

        cfg.aggregated_results_dir.mkdir(parents=True, exist_ok=True)

    # 2. Filter out runs already in the aggregate file
    params_name = (  # Base name of params file without extension
        cfg.params_file[: -len(".json")]
        if cfg.params_file.endswith(".json")
        else cfg.params_file
    )

    # Path to aggregated results file
    if cfg.aggregate_file is not None:
        aggregate_file = cfg.aggregated_results_dir / cfg.aggregate_file
        if not aggregate_file.exists():
            raise FileNotFoundError(
                f"Specified aggregate_file {aggregate_file} does not exist"
            )
    else:
        aggregate_file = cfg.aggregated_results_dir / (params_name + ".csv")

    if cfg.filter_from_aggregated:
        if not aggregate_file.exists():
            print(f"No aggregate file found at {aggregate_file}, skipping filtering.")
        else:
            if cfg.verbose:
                print(f"Filtering runs already in aggregate file {aggregate_file}")
            runs_list = filter_from_aggregate(cfg, aggregate_file, runs_list)
            if cfg.verbose:
                print(
                    f"Filtered out {before_len - len(runs_list)} jobs already in aggregate file"
                )
                before_len = len(runs_list)  # Update before_len for next filtering

    # 3. Filter out runs already running or queued
    if cfg.filter_running_or_queued:
        runs_list = filter_running_or_queued(runs_list)
        if cfg.verbose:
            print(
                f"Filtered out {before_len - len(runs_list)} jobs already running or queued"
            )

    print(f"Total runs to submit: {len(runs_list)}")
    if len(runs_list) == 0:
        print("No runs to submit, exiting.")
        return

    if cfg.verbose:
        datasets = set(r.dataset_name for _, r in runs_list)
        print(f"Datasets to process: {', '.join(sorted(datasets))}")
        print(
            f"Network types to process: {', '.join(sorted(set(r.network_type for _, r in runs_list)))}"
        )
        print()

    if cfg.use_memoization:
        test_epoch_dict = load_test_epoch_memoization(cfg)

    # Find all cached runs and runs needing computation
    cached_runs, runs_to_compute = [], []
    for run_index, params in runs_list:
        key = get_test_epoch_key(cfg, params)
        if key in test_epoch_dict:
            cached_runs.append((run_index, params))
        else:
            runs_to_compute.append((run_index, params))

    print(
        f"Test epoch status: {len(cached_runs)} cached, {len(runs_to_compute)} to compute"
    )

    if not cfg.dry_run:
        # Create unique output directory for this submission
        # (named after the params file)
        log_dir = cfg.log_dir / params_name
        if not log_dir.exists():
            log_dir /= "run_0"
            log_dir.mkdir(parents=True)
        else:
            # Find next available run id
            run_id = 1
            d = log_dir / f"run_{run_id}"
            while d.exists():
                run_id += 1
                d = log_dir / f"run_{run_id}"

            log_dir /= f"run_{run_id}"
            log_dir.mkdir(exist_ok=False)
    else:
        log_dir = None

    def _submit_cached_runs():
        """Submit all cached runs"""
        for run_index, params in cached_runs:
            key = get_test_epoch_key(cfg, params)
            epoch_time = test_epoch_dict[key]
            submit_pbs_job(cfg, run_index, params, epoch_time, log_dir)

    # If no runs need computation, just submit cached runs and exit
    if len(runs_to_compute) == 0:
        if cfg.verbose:
            print("All test epochs cached, submitting cached runs only.")
        _submit_cached_runs()
        return

    # Otherwise, launch workers to compute test epochs in parallel,
    # before submitting the cached runs
    from multiprocessing import Process, Lock, Value, Manager

    multiprocessing.set_start_method("spawn", force=True)

    # Create shared manager for test epoch results
    test_epoch_manager = Manager()
    test_epoch_lock = test_epoch_manager.Lock()
    test_epoch_dict = test_epoch_manager.dict(test_epoch_dict)
    print_lock = Lock()
    num_workers = min(cfg.num_workers, len(runs_to_compute))
    processes = []

    # Partition the work among available workers
    partition_size = math.ceil(len(runs_to_compute) / num_workers)
    for worker_id in range(num_workers):
        partition_low = worker_id * partition_size
        partition_high = min((partition_low + partition_size), len(runs_to_compute))
        jobs_to_run = runs_to_compute[partition_low:partition_high]
        if cfg.verbose:
            print(
                f"  Worker {worker_id} running {len(jobs_to_run)} jobs: {[i for i, _ in jobs_to_run]}"
            )
        if len(jobs_to_run) == 0:
            continue
        p = Process(
            target=compute_and_submit_job_worker,
            args=(
                cfg,
                worker_id,
                jobs_to_run,
                test_epoch_lock,
                test_epoch_dict,
                print_lock,
                log_dir,
            ),
        )
        p.start()
        processes.append(p)

    # Check that all workers have started their test epoch jobs before queuing cached runs
    if not cfg.dry_run:
        _, num_running = get_queued_jobs(return_num_running=True)
        while num_running < num_workers:
            time.sleep(1)
            _, num_running = get_queued_jobs(return_num_running=True)
        if len(cached_runs) > 0 and cfg.verbose:
            print("All workers are running, submitting cached runs...")
        _submit_cached_runs()

    # Wait for all workers to complete
    try:
        if cfg.verbose:
            print("\nWaiting for all workers to complete...")
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\nComputation interrupted by user, terminating workers...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.join()
        test_epoch_manager.shutdown()
        return

    if cfg.dry_run:
        print("\nDry run complete, exiting.")
        test_epoch_manager.shutdown()
        return

    try:
        # Wait for all jobs to complete
        print("\nWaiting for all jobs to complete.")
        while True:
            current_jobs = get_queued_jobs()
            if current_jobs == 0:
                break
            time.sleep(5 * 60)
            if cfg.verbose:
                print(f"  {current_jobs} jobs still in queue, waiting...")
        print("All jobs completed.")
    except KeyboardInterrupt:
        print("\nInterrupted by user, exiting.")
        test_epoch_manager.shutdown()
        return

    # Retry any failed jobs found in the log directory
    if cfg.retry_failed:
        if cfg.verbose:
            print("\nChecking for failed jobs to retry...")
        retry_failed_jobs(cfg, log_dir, runs_list, test_epoch_dict)

    # Finally, aggregate results into a single CSV file (the aggregate file)
    if cfg.aggregate_results:
        aggregate_results(cfg, aggregate_file)

    # Shutdown the multiprocessing manager at the very end
    test_epoch_manager.shutdown()


if __name__ == "__main__":
    main()
