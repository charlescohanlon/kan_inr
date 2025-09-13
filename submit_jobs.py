from contextlib import nullcontext
import multiprocessing
import os
import subprocess
import time
import pickle
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass, fields

import hydra
from hydra.core.config_store import ConfigStore

from benchmark import BenchmarkConfig, RunParams, parse_run_params


@dataclass
class SubmissionConfig(BenchmarkConfig):
    """Extended configuration that includes submission-specific parameters"""

    # Submission-specific parameters
    max_jobs: int = 20
    wait_time: int = 10 * 60
    dry_run: bool = False
    verbose: bool = False
    array_index_subset: Optional[List[int]] = None
    num_minutes_per_job: float = 3  # Expected minutes per job for test epoch
    compute_epoch_time_only: bool = False
    num_workers: int = 4  # Number of parallel workers test_epoch test epochs
    time_scaler: float = 0.9  # Scaling factor for estimated time

    # Directories for logs and memoization
    log_dir: Path = Path(BenchmarkConfig.home_dir) / "kan_inr" / "logs"
    memo_dir: Path = Path(BenchmarkConfig.home_dir) / "memo"
    test_epoch_dir: Path = log_dir / "test_epochs"

    test_epoch_memo_file: Path = memo_dir / "test_epoch_memo.pkl"


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


def submit_test_epoch_job(
    cfg: SubmissionConfig, run_indices: List[int], worker_id: int, output_file: Path
) -> str:
    """Create a bash script for testing single epochs"""
    # Estimate walltime for test epoch runs
    time_in_minutes = len(run_indices) * cfg.num_minutes_per_job
    walltime = f"{int(time_in_minutes // 60):02d}:{int(time_in_minutes % 60):02d}:00"
    job_name = f"te_w{worker_id}"

    override_args = get_override_args(cfg)
    override_args.append("test_epoch_run=True")
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
    # Submit script to PBS
    result = subprocess.run(
        ["qsub"],
        input=script,
        capture_output=True,
        text=True,
        check=True,
    )
    job_id = result.stdout.strip()

    return job_id


def get_pbs_walltime_and_gpus(
    cfg: SubmissionConfig, params: RunParams, epoch_time: float
) -> str:
    """
    Estimate walltime based on empirical test epoch results.
    """
    # Calculate total estimated time
    # NOTE: epoch_time includes other overhead which pads the estimate
    # +1 for reconstruction overhead
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
    print_lock=None,
) -> str:
    walltime, num_gpus = get_pbs_walltime_and_gpus(cfg, params, epoch_time)
    job_name = f"{params.network_type[:1]}_{params.dataset_name[:3]}_{run_index}"

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

    params_file = (
        cfg.params_file[: -len(".json")]
        if cfg.params_file.endswith(".json")
        else cfg.params_file
    )
    output_dir = cfg.log_dir / params_file
    if not output_dir.exists():
        output_dir /= "run_0"
        output_dir.mkdir(parents=True)
    else:
        # Find next available run id
        run_id = 1
        while (output_dir / f"run_{run_id}").exists():
            run_id += 1
        output_dir /= f"run_{run_id}"
        output_dir.mkdir()

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
        num_running = sum(1 for line in lines if "R" in line)
        num_jobs = max(0, len(lines) - 5)  # Skip header lines
        if return_num_running:
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
    success_counter,
    print_lock,
):
    test_epoch_job_id = None
    jobs_run = set()
    try:
        # Then submit the test epoch script for runs needing computation
        compute_indices = [run_index for run_index, _ in jobs_to_run]
        test_epoch_file = cfg.test_epoch_dir / f"test_epoch_w{worker_id}.txt"
        if not cfg.dry_run:
            test_epoch_job_id = submit_test_epoch_job(
                cfg, compute_indices, worker_id, test_epoch_file
            )
            while not test_epoch_file.exists():
                time.sleep(30)
        else:
            create_dry_run_test_epoch_file(test_epoch_file, compute_indices)

        while len(jobs_run) < len(compute_indices):
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

                if not cfg.dry_run:
                    successfully_saved = save_test_epoch_memoization(
                        cfg, test_epoch_dict, test_epoch_lock
                    )
                    if successfully_saved:
                        with success_counter.get_lock():
                            success_counter.value += 1

                if (run_index in jobs_run) or cfg.compute_epoch_time_only:
                    continue

                submit_pbs_job(cfg, run_index, params, time_elapsed, print_lock)
                jobs_run.add(run_index)

            if len(jobs_run) < len(compute_indices):
                time.sleep(30)  # Wait for file to update
    except KeyboardInterrupt:
        if test_epoch_job_id is not None:
            # Terminate test epoch job
            subprocess.run(["qdel", test_epoch_job_id])
    finally:
        if test_epoch_file.exists():
            test_epoch_file.unlink()


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
    total_runs = len(runs_list)
    print(f"Total runs to submit: {total_runs}")
    print(f"Repeats per run: {cfg.repeats}")

    if total_runs == 0:
        print("No runs found with specified filters")
        return

    # Filter by array_index_subset if specified
    if cfg.array_index_subset is not None:
        runs_list = [
            (i, r) for i, r in enumerate(runs_list) if i in cfg.array_index_subset
        ]
    else:
        runs_list = list(enumerate(runs_list))

    if cfg.num_workers > len(runs_list):
        print(
            f"Warning: adjusting num_workers to {cfg.num_workers} due to limited jobs"
        )
        cfg.num_workers = len(runs_list)

    # Summary of datasets to process
    if cfg.verbose:
        datasets = set(r.dataset_name for _, r in runs_list)
        print(f"Datasets to process: {', '.join(sorted(datasets))}")

    # Load existing test epoch memoization
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

    def _submit_cached_runs():
        """Submit all cached runs"""
        for run_index, params in cached_runs:
            key = get_test_epoch_key(cfg, params)
            epoch_time = test_epoch_dict[key]
            submit_pbs_job(cfg, run_index, params, epoch_time)

    def _summary(computed_completed=len(runs_to_compute)):
        """Print final submission summary"""
        print("\n=== SUBMISSION SUMMARY ===")
        print(f"Total runs: {len(cached_runs) + computed_completed}")
        print(f"Test epoch results from cached: {len(cached_runs)}")
        print(f"Test epoch results computed: {computed_completed}")

    # If no runs need computation, just submit cached runs and exit
    if len(runs_to_compute) == 0:
        _submit_cached_runs()
        _summary()
        return

    # Otherwise, launch workers to compute test epochs in parallel,
    # before submitting the cached runs
    from multiprocessing import Process, Lock, Value, Manager

    multiprocessing.set_start_method("spawn", force=True)

    # Create shared manager for test epoch results and success counter
    test_epoch_manager = Manager()
    test_epoch_lock = test_epoch_manager.Lock()
    test_epoch_dict = test_epoch_manager.dict(test_epoch_dict)
    success_counter = Value("i", 0)
    print_lock = Lock()

    processes = []
    partition_size = (len(runs_to_compute) + 1) // cfg.num_workers
    for worker_id in range(cfg.num_workers):
        partition_low = worker_id * partition_size
        partition_high = min((partition_low + partition_size), len(runs_to_compute))
        jobs_to_run = runs_to_compute[partition_low:partition_high]
        if cfg.verbose:
            print(
                f"  Worker {worker_id} running {len(jobs_to_run)} jobs: {[i for i, _ in jobs_to_run]}"
            )
        p = Process(
            target=compute_and_submit_job_worker,
            args=(
                cfg,
                worker_id,
                jobs_to_run,
                test_epoch_lock,
                test_epoch_dict,
                success_counter,
                print_lock,
            ),
        )
        p.start()
        processes.append(p)

    # Check that all workers have started their test epoch jobs before queuing cached runs
    _, num_running = get_queued_jobs(return_num_running=True)
    while num_running < cfg.num_workers:
        time.sleep(1)
        _, num_running = get_queued_jobs(return_num_running=True)
    if cfg.verbose:
        print("All workers have queued, submitting cached runs...")
    _submit_cached_runs()

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\nComputation interrupted by user, terminating workers...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.join()

    _summary(computed_completed=success_counter.value)


if __name__ == "__main__":
    main()
