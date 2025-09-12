import os
import sys
import subprocess
import time
import tempfile
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
    num_minutes_per_job: float = 3
    compute_epoch_time_only: bool = False


class JobSubmissionManager:
    def __init__(self, cfg: SubmissionConfig):
        self.cfg = cfg
        self.max_queued_jobs = cfg.max_jobs
        self.wait_time = cfg.wait_time
        self.dry_run = cfg.dry_run
        self.verbose = cfg.verbose
        self.home_dir = Path(cfg.home_dir)
        self.array_index_subset = cfg.array_index_subset
        self.num_minutes_per_job = cfg.num_minutes_per_job

        # Directories for logs and memoization
        self.log_dir = self.home_dir / "kan_inr" / "logs"
        self.memo_dir = self.home_dir / "memo"
        self.test_epoch_dir = self.log_dir / "test_epochs"
        self.memo_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.test_epoch_dir.mkdir(parents=True, exist_ok=True)

        # Only empirically compute epoch times (no submission)
        self.compute_epoch_time_only = cfg.compute_epoch_time_only

        # For storing test epoch timing results
        self.epoch_timing_results = {}

        # Memoization file for timing results
        self.timing_memo_file = self.memo_dir / "epoch_timing_memo.pkl"
        self.load_timing_memoization()

    def get_timing_results_key(self, params: RunParams) -> str:
        """Get a unique key for timing results based on RunParams, safety margin, and dataset name."""
        safety_margin = self.cfg.safety_margin
        dataset_name = params.dataset_name
        ssd_dir_provided = self.cfg.ssd_dir is not None
        return str(
            params.epoch_time_hash(safety_margin, dataset_name, ssd_dir_provided)
        )

    def load_timing_memoization(self):
        """Load existing timing results from memoization file"""
        if self.timing_memo_file.exists():
            try:
                with open(self.timing_memo_file, "rb") as f:
                    self.epoch_timing_results = pickle.load(f)
            except (pickle.PickleError, FileNotFoundError, EOFError) as e:
                print(f"Warning: Could not load timing memoization file: {e}")
                self.epoch_timing_results = {}
        else:
            self.epoch_timing_results = {}

    def save_timing_memoization(self):
        """Save timing results to memoization file"""
        try:
            with open(self.timing_memo_file, "wb") as f:
                pickle.dump(self.epoch_timing_results, f)
        except (pickle.PickleError, OSError) as e:
            print(f"Warning: Could not save timing memoization file: {e}")

    def has_cached_timing(self, params: RunParams) -> bool:
        """Check if timing results exist for given RunParams"""
        key = self.get_timing_results_key(params)
        return key in self.epoch_timing_results

    def create_test_epoch_script(self, run_indices: List[int]) -> Tuple[str, Path]:
        """Create a bash script for testing single epochs"""
        # Estimate walltime for test epoch runs
        time_in_minutes = len(run_indices) * self.num_minutes_per_job
        walltime = (
            f"{int(time_in_minutes // 60):02d}:{int(time_in_minutes % 60):02d}:00"
        )
        job_name = "test_epochs"

        # Convert SubmissionConfig superclass to BenchmarkConfig for argument parsing
        submission_fields = {f.name for f in fields(SubmissionConfig)}
        test_epoch_fields = {f.name for f in fields(BenchmarkConfig)}
        override_args = []
        for key in submission_fields:
            if key in test_epoch_fields and self.cfg[key] is not None:
                # Override repeats to 1 for test epoch
                if key == "repeats":
                    override_args.append("repeats=1")
                else:
                    override_args.append(f'{key}={str(self.cfg[key]).replace(" ", "")}')

        override_args.append("epochs=1")
        override_args.append("train_only=True")
        override_str = " ".join(override_args).replace("'", '"')

        timing_file = self.test_epoch_dir / "test_epoch_timings.txt"

        script = f"""#!/bin/bash
# ---------- PBS DIRECTIVES ----------
#PBS -A insitu
#PBS -q by-gpu
#PBS -l select=1:ncpus=8:gputype=A100:system=sophia
#PBS -l walltime={walltime}
#PBS -l filesystems=home:grand
#PBS -N {job_name}
#PBS -o {self.test_epoch_dir}
#PBS -e {self.test_epoch_dir}
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
    echo "run_index: $run_index, time_elapsed: $ELAPSED" >> {timing_file}
done
"""
        return script, timing_file

    def start_test_epoch_script(self, script_content: str) -> str:
        """Submit the test epoch script to PBS and return the job ID"""
        # Write script to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write(script_content)
            script_path = f.name

        # Make script executable
        os.chmod(script_path, 0o755)

        # Submit script to PBS
        result = subprocess.run(
            ["qsub", script_path],
            capture_output=True,
            text=True,
            check=True,
        )
        job_id = result.stdout.strip()

        if self.verbose:
            print(f"Submitted test epoch script as job {job_id}")

        return job_id

    def get_walltime_and_gpus(self, params: RunParams, epoch_time: float) -> str:
        """
        Estimate walltime based on empirical test epoch timing.
        """
        # Calculate total estimated time
        # NOTE: epoch_time includes other overhead which pads the estimate
        # +1 for reconstruction overhead
        total_time = (params.epochs + 1) * self.cfg.repeats * epoch_time

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

    def generate_pbs_script(
        self,
        run_index: int,
        params: RunParams,
        walltime: str,
        num_gpus: int = 1,
    ) -> str:
        """Generate a customized PBS script for this specific job"""
        # Generate a descriptive job name (PBS limits to 15 chars)
        job_name = f"{params.network_type[:1]}_{params.dataset_name[:3]}_{run_index}"

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
#PBS -l select={num_gpus}:ncpus={num_gpus*8}:gputype=A100:system=sophia
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
echo "Starting at: $(date)"

torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS \\
    benchmark.py -cn config {override_str}

echo "Completed at: $(date)"
"""
        return script

    def submit_job(
        self,
        run_index: int,
        params: RunParams,
        walltime: str,
        num_gpus: int = 1,
    ):
        """Submit a job to the PBS scheduler"""
        # Generate PBS script to submit job
        script_content = self.generate_pbs_script(run_index, params, walltime, num_gpus)

        # Submit via stdin to qsub, wait for success (will fail if queue is full)
        while True:
            try:
                self.wait_for_slot()
                result = subprocess.run(
                    ["qsub"],
                    input=script_content,
                    text=True,
                    capture_output=True,
                    check=True,  # Raise error on non-zero exit code
                )

                if self.verbose:
                    job_id = result.stdout.strip()
                    print(f"  Submitted: {job_id}")

                return
            except subprocess.CalledProcessError as e:
                self.wait_for_slot(must_wait=True)

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
                f"{current_jobs} are currently queued. "
                f"Waiting {self.wait_time / 60:.1f} minutes before retrying..."
            )
            time.sleep(self.wait_time)
            must_wait = False  # Only force wait once

    def run(self):
        """Main execution loop"""
        runs_list = parse_run_params(self.cfg)
        total_runs = len(runs_list)

        print(f"Total runs to submit: {total_runs}")
        print(f"Repeats per run: {self.cfg.repeats}")

        if total_runs == 0:
            print("No runs found with specified filters")
            return

        # Filter by array_index_subset if specified
        if self.array_index_subset is not None:
            runs_list = [
                (i, r) for i, r in enumerate(runs_list) if i in self.array_index_subset
            ]
        else:
            runs_list = list(enumerate(runs_list))

        # Summary of datasets to process
        if self.verbose:
            datasets = set(r.dataset_name for _, r in runs_list)
            print(f"Datasets to process: {', '.join(sorted(datasets))}")

        cached_runs, compute_runs = [], []
        for run_index, params in runs_list:
            if self.has_cached_timing(params):
                cached_runs.append((run_index, params))
            else:
                compute_runs.append((run_index, params))

        num_caches, num_compute = len(cached_runs), len(compute_runs)
        print(f"Timing status: {num_caches} cached, {num_compute} to compute")

        def _submit(i: int, p: RunParams):
            key = self.get_timing_results_key(p)
            time_elapsed = self.epoch_timing_results[key]
            walltime, num_gpus = self.get_walltime_and_gpus(p, time_elapsed)

            if self.verbose or self.dry_run:
                print(f"\nJob {i}:")
                print("  Run Parameters")
                for field in fields(RunParams):
                    value = getattr(p, field.name)
                    print(f"    {field.name}: {value}")
                print(f"  Epoch time: {time_elapsed:.2f} seconds")
                print(f"  Resources: {num_gpus} GPUs, {walltime} walltime")

            if not self.dry_run:
                self.submit_job(i, p, walltime, num_gpus)

        def _summary(computed_completed=num_compute):
            print("\n=== SUBMISSION SUMMARY ===")
            print(f"Total runs: {num_caches + computed_completed}")
            print(f"Timing results from cached: {num_caches}")
            print(f"Timing results computed: {computed_completed}")

        # First submit cached runs
        for run_index, params in cached_runs:
            if self.verbose:
                print("\nSubmitting cached run:")
            _submit(run_index, params)

        if num_compute == 0:
            _summary()
            return

        job_id = None
        jobs_run = set()
        try:
            # Then submit the test epoch script for runs needing timing
            compute_indices = [i for i, _ in compute_runs]
            script_content, timing_file = self.create_test_epoch_script(compute_indices)

            if not self.dry_run:
                job_id = self.start_test_epoch_script(script_content)

                # Timing file exists when first run completes
                while not timing_file.exists():
                    time.sleep(30)
            else:
                # In dry run mode, create a fake timing file
                create_dry_run_timing_file(timing_file, compute_indices)

            while len(jobs_run) < len(compute_indices):
                with open(timing_file, "r") as f:
                    lines = f.readlines()

                for line in lines:
                    if not "run_index" in line or not "time_elapsed" in line:
                        continue

                    parts = line.strip().split(",")
                    run_index = int(parts[0].split(":")[1].strip())
                    time_elapsed = float(parts[1].split(":")[1].strip())
                    # Find corresponding RunParams
                    params = next(p for i, p in compute_runs if i == run_index)

                    # Timing results now known, save them
                    key = self.get_timing_results_key(params)
                    self.epoch_timing_results[key] = time_elapsed
                    if not self.dry_run:
                        self.save_timing_memoization()

                    if (run_index in jobs_run) or self.compute_epoch_time_only:
                        continue

                    _submit(run_index, params)
                    jobs_run.add(run_index)

                if len(jobs_run) < len(compute_indices):
                    time.sleep(30)  # Wait for file to update

        except KeyboardInterrupt:
            print("\nComputing epoch test times interrupted by user.")
        finally:
            # Clean up
            if timing_file.exists():
                timing_file.unlink()

            if job_id is not None:
                # Terminate test epoch job
                subprocess.run(["qdel", job_id])

            # Final summary
            _summary(computed_completed=len(jobs_run))


def create_dry_run_timing_file(timing_file: Path, compute_indices: List[int]):
    """Create a fake timing file for dry run mode"""
    fake_time = -1.0
    with open(timing_file, "w") as f:
        for run_index in compute_indices:
            f.write(f"run_index: {run_index}, time_elapsed: {fake_time}\n")


# Register the config schema with Hydra as a structured config
cs = ConfigStore.instance()

# Override schema to interpret "--config-name config" as a SubmissionConfig (not a BenchmarkConfig)
cs.store(name="config", node=SubmissionConfig)


@hydra.main(version_base="1.3", config_path="conf")
def main(cfg: SubmissionConfig):
    """Main entry point with Hydra configuration"""

    if cfg.batch_size is not None:
        print("Warning: Overriding batch_size will affect timing estimate memoization.")
    if cfg.safety_margin != BenchmarkConfig.safety_margin:
        print(
            f"Warning: Using non-default safety_margin ({cfg.safety_margin}) "
            "will affect timing estimate memoization."
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

    # Create and run the submission manager
    manager = JobSubmissionManager(cfg)

    try:
        manager.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
