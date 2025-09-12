import os
import sys
import subprocess
import pexpect
import time
import tempfile
import threading
import queue
import pickle
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass, fields

import hydra
from hydra.core.config_store import ConfigStore

from benchmark import BenchmarkConfig, RunParams, parse_filename, parse_run_params


@dataclass
class SubmissionConfig(BenchmarkConfig):
    """Extended configuration that includes submission-specific parameters"""

    # Submission-specific parameters
    max_jobs: int = 20
    queue_job_wait_time: int = 10 * 60  # seconds
    dry_run: bool = False
    verbose: bool = False
    array_index_subset: Optional[List[int]] = None
    num_workers: int = 8  # Number of concurrent test_epoch+submission workers
    num_minutes_per_job: float = 3
    epoch_time_only: bool = False


class JobSubmissionManager:
    def __init__(self, cfg: SubmissionConfig):
        self.cfg = cfg
        self.max_queued_jobs = cfg.max_jobs
        self.wait_time = cfg.queue_job_wait_time
        self.dry_run = cfg.dry_run
        self.verbose = cfg.verbose
        self.home_dir = Path(cfg.home_dir)
        self.log_dir = self.home_dir / "kan_inr" / "logs"
        self.memo_dir = self.home_dir / "memo"
        self.memo_dir.mkdir(parents=True, exist_ok=True)
        if self.cfg.dataset is not None:
            self.log_dir /= self.cfg.dataset
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Only empirically compute epoch times (no submission)
        self.epoch_time_only = cfg.epoch_time_only

        # For storing test epoch timing results
        self.timing_results = {}
        self.interactive_session = None
        self.test_epoch_queue = queue.Queue()

        # Memoization file for timing results
        self.timing_memo_file = self.memo_dir / "epoch_timing_memo.pkl"
        self.load_timing_memoization()

        # For thread-safe access to the interactive session
        self.interactive_session_lock = threading.Lock()
        # For thread-safe access to timing results memoization
        self.timing_results_lock = threading.Lock()
        # For thread-safe printing (to stdout)
        self.print_lock = threading.Lock()

    def get_timing_results_key(self, params: RunParams) -> str:
        return str(hash(params))  # Use RunParams __hash__ method for unique key

    def load_timing_memoization(self):
        """Load existing timing results from memoization file"""
        if self.timing_memo_file.exists():
            try:
                with open(self.timing_memo_file, "rb") as f:
                    self.timing_results = pickle.load(f)
                if self.verbose:
                    print(
                        f"Loaded {len(self.timing_results)} timing results from memoization file"
                    )
            except (pickle.PickleError, FileNotFoundError, EOFError) as e:
                if self.verbose:
                    print(f"Warning: Could not load timing memoization file: {e}")
                self.timing_results = {}
        else:
            self.timing_results = {}

    def save_timing_memoization(self):
        """Save timing results to memoization file"""
        with self.timing_results_lock:
            try:
                with open(self.timing_memo_file, "wb") as f:
                    pickle.dump(self.timing_results, f)
                if self.verbose:
                    print(
                        f"Saved {len(self.timing_results)} timing results to memoization file"
                    )
            except (pickle.PickleError, OSError) as e:
                if self.verbose:
                    print(f"Warning: Could not save timing memoization file: {e}")

    def has_cached_timing(self, params: RunParams) -> bool:
        """Check if timing results exist for given RunParams"""
        key = self.get_timing_results_key(params)
        return key in self.timing_results

    def get_dataset_info(self, dataset_name: str) -> Tuple[Tuple[int, int, int], str]:
        """Get dataset shape and type from pre-parsed info or by finding the file"""

        # If not found, try to find the actual file
        data_dir = self.home_dir / self.cfg.data_path
        if data_dir.exists():
            for dataset_file in os.listdir(data_dir):
                if dataset_file.startswith(dataset_name):
                    data_path = data_dir / dataset_file
                    _, shape, dtype = parse_filename(data_path)
                    return shape, dtype

        raise ValueError(f"Unknown dataset: {dataset_name}")

    def create_test_epoch_script(
        self, run_index: int, params: RunParams
    ) -> Tuple[str, Path]:
        """Create a bash script for testing a single epoch"""
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

        # Add epochs=1 to ensure single epoch
        override_args.append("epochs=1")

        # Only train, no eval or save
        override_args.append("train_only=True")

        override_str = " ".join(override_args).replace("'", '"')

        # Create timing file path
        timing_file = self.log_dir / f"test_epoch_timing_{run_index}.txt"

        script = f"""#!/bin/bash
source /grand/insitu/cohanlon/miniconda3/etc/profile.d/conda.sh 
cd /grand/insitu/cohanlon/kan_inr
conda activate kan_inr

export PBS_ARRAY_INDEX={run_index}

START_TIME=$(date +%s.%N)

# Run single epoch benchmark
python benchmark.py -cn config {override_str}

END_TIME=$(date +%s.%N)
ELAPSED=$(echo "$END_TIME - $START_TIME" | bc)

echo "$ELAPSED" > {timing_file}
"""
        return script, timing_file

    def start_interactive_session(self, total_runs: int):
        """Start an interactive PBS session in the background"""
        # Calculate walltime for interactive session
        minutes = round(total_runs * self.cfg.num_minutes_per_job)
        hours = minutes // 60
        remaining_minutes = minutes % 60
        walltime = f"{hours:02d}:{remaining_minutes:02d}:00"

        if self.verbose:
            print(f"Waiting on interactive session with walltime {walltime}")

        if not self.dry_run:
            # Start interactive session with pexpect
            # (requires PTY which subprocess can't provide)
            args = [
                "-I",
                "-A",
                "insitu",
                "-q",
                "by-gpu",
                "-l",
                "select=1:ncpus=8:gputype=A100:system=sophia",
                "-l",
                "filesystems=home:grand",
                "-l",
                f"walltime={walltime}",
            ]
            self.interactive_session = pexpect.spawn(
                "qsub", args, encoding="utf-8", timeout=None
            )
            # Monitor until the interactive job starts (i.e., output contains "ready")
            exit_code = self.interactive_session.expect("ready", timeout=None)
            if exit_code != 0:
                raise RuntimeError("Failed to start interactive session")

            if self.verbose:
                print("Interactive session started")

    def get_test_epoch_time(self, run_index: int, params: RunParams) -> float:
        """Run a single epoch and return the time taken"""
        # Check if we have cached timing results for this RunParams
        if self.has_cached_timing(params):
            key = self.get_timing_results_key(params)
            cached_result = self.timing_results[key]
            num_gpus = cached_result["num_gpus"]
            epoch_time = cached_result["test_epoch_time"]
            if self.verbose:
                with self.print_lock:
                    print(
                        f"  Using cached timing: {epoch_time:.2f} seconds on {num_gpus} GPUs"
                    )
            return epoch_time

        if self.dry_run:
            return 1.0  # Dummy time for dry run

        # Create test_epoch script
        script_content, timing_file = self.create_test_epoch_script(run_index, params)

        # Write script to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write(script_content)
            script_path = f.name

        try:
            # Make script executable
            os.chmod(script_path, 0o755)

            # Run the test_epoch script in the interactive session
            if self.interactive_session:
                # Thread-safe access to interactive session
                with self.interactive_session_lock:
                    if self.verbose:
                        with self.print_lock:
                            print(f"  Running test epoch for job {run_index}...")

                    # Send the script to the interactive session
                    self.interactive_session.sendline(f"bash {script_path}\n")
                    self.interactive_session.expect("\n", timeout=None)

                    while not timing_file.exists():
                        time.sleep(5)

                    if timing_file.exists():
                        with open(timing_file, "r") as f:
                            elapsed_time = float(f.read().strip())

                        # Clean up timing file
                        timing_file.unlink()

                        if self.verbose:
                            with self.print_lock:
                                print(
                                    f"Test epoch for job {run_index} completed in {elapsed_time:.2f} seconds"
                                )
                    else:
                        raise RuntimeError(
                            f"Test epoch for job {run_index} did not complete in time"
                        )

                return elapsed_time

            else:
                raise RuntimeError("Interactive session is not active")
        finally:
            # Clean up script file
            if os.path.exists(script_path):
                os.unlink(script_path)

    def get_walltime(self, params: RunParams, epoch_time: float) -> str:
        """
        Estimate walltime based on empirical test epoch timing.
        """
        # Calculate total estimated time
        # NOTE: uses 0.8 b/c epoch_time includes other overhead so there is
        # which adds padding to the estimate
        total_time = 0.8 * params.epochs * self.cfg.repeats * epoch_time

        # Adjust for extra GPUs based on time thresholds
        num_gpus = 1
        seconds_per_hour = 3600
        if total_time > seconds_per_hour * 2:
            num_gpus = max(2, num_gpus)
        if total_time > seconds_per_hour * 4:
            num_gpus = max(4, num_gpus)
        if total_time > seconds_per_hour * 8:
            num_gpus = 8

        # Update memoization immediately when new results are added
        key = self.get_timing_results_key(params)
        with self.timing_results_lock:
            self.timing_results[key] = {
                "num_gpus": num_gpus,
                "test_epoch_time": epoch_time,
            }
        if not self.dry_run:
            self.save_timing_memoization()

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

        return walltime

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
echo "Starting at: $(date)"

torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS \\
    benchmark.py -cn config {override_str}

echo "Completed at: $(date)"
"""
        return script

    def submit_job_with_timing(
        self,
        run_index: int,
        params: RunParams,
        epoch_time: float,
    ) -> bool:
        """Submit a single job using empirical timing and return success status"""
        # Get walltime given test epoch time
        walltime = self.get_walltime(params, epoch_time)

        # At this point num_gpus and walltime are known
        if self.epoch_time_only:
            if self.verbose:
                with self.print_lock:
                    key = self.get_timing_results_key(params)
                    num_gpus = self.timing_results[key]["num_gpus"]
                    print(
                        f"Job {run_index}: Computed epoch time {epoch_time:.2f} seconds on {num_gpus} GPUs,"
                        f"would request {walltime} walltime"
                    )
            return True

        # Get GPU count from stored results
        key = self.get_timing_results_key(params)
        with self.timing_results_lock:
            num_gpus = self.timing_results.get(key, {}).get("num_gpus", 1)

        data_shape, data_type = self.get_dataset_info(params.dataset_name)

        if self.verbose or self.dry_run:
            with self.print_lock:
                print(f"\nJob {run_index}:")
                print(f"  Dataset shape: {data_shape} ({data_type})")
                print("  Run Parameters:")
                for field in fields(RunParams):
                    value = getattr(params, field.name)
                    print(f"    {field.name}: {value}")
                print(f"  Test Epoch time: {epoch_time:.2f} seconds")
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
                with self.print_lock:
                    print(f"  Submitted: {job_id}")

            return True

        except subprocess.CalledProcessError as e:
            with self.print_lock:
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

    def wait_for_slot(self, must_wait=False, should_print=False):
        """Wait for a queue slot to become available"""
        while True:
            current_jobs = self.get_queued_jobs()
            if current_jobs < self.max_queued_jobs and not must_wait:
                return

            if should_print:
                with self.print_lock:
                    print(
                        f"Queue full ({current_jobs + 1}/{self.max_queued_jobs}). "
                        f"Waiting {self.wait_time / 60:.1f} minutes before retrying..."
                    )
            time.sleep(self.wait_time)
            must_wait = False  # Only force wait once

    def test_epoch_and_submit_worker(self, work_queue: queue.Queue, worker_id: int = 0):
        """Worker thread to process test epoch and submission tasks"""
        while True:
            try:
                task = work_queue.get(timeout=1)
                if task is None:  # Poison pill to stop worker
                    break

                run_index, params = task

                try:
                    epoch_time = self.get_test_epoch_time(run_index, params)

                    # Only proceed if we got a valid epoch time
                    if epoch_time > 0:
                        # Submit job with empirical timing
                        if not self.dry_run:
                            self.wait_for_slot()

                        success = False
                        while not success:
                            success = self.submit_job_with_timing(
                                run_index, params, epoch_time
                            )
                            if not self.dry_run and not self.epoch_time_only:
                                # If last submission failed, must wait before trying again
                                must_wait = not success
                                # Only first worker prints
                                should_print = worker_id == 0
                                self.wait_for_slot(must_wait, should_print)

                        if not self.dry_run:
                            time.sleep(1)  # Small delay to avoid overwhelming scheduler
                    else:
                        raise ValueError("Invalid epoch time")

                except Exception as e:
                    print(f"Failed to process job {run_index}: {e}, skipping job")

                work_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in worker thread: {e}")

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
        if self.cfg.array_index_subset is not None:
            subset_runs = [
                (i, r)
                for i, r in enumerate(runs_list)
                if i in self.cfg.array_index_subset
            ]
        else:
            subset_runs = list(enumerate(runs_list))

        actual_runs = len(subset_runs)

        # Count how many runs have cached timing vs need computation
        cached_count = sum(
            1 for _, params in subset_runs if self.has_cached_timing(params)
        )
        new_count = actual_runs - cached_count

        print(f"Timing status: {cached_count} cached, {new_count} to compute")

        # Print summary of datasets if verbose
        if self.verbose:
            datasets = set(r.dataset_name for r in runs_list)
            print(f"Datasets to process: {', '.join(sorted(datasets))}")

        # Start interactive session
        if not self.dry_run:
            self.start_interactive_session(actual_runs)

        # Create work queue
        work_queue = queue.Queue()

        # Add all jobs to the queue
        for run_index, params in subset_runs:
            work_queue.put((run_index, params))

        # Start worker threads for concurrent test epoch and submission
        num_workers = min(self.cfg.num_workers, actual_runs)
        workers = []

        for worker_id in range(num_workers):
            worker = threading.Thread(
                target=self.test_epoch_and_submit_worker, args=(work_queue, worker_id)
            )
            worker.start()
            workers.append(worker)

        # Wait for all tasks to complete
        work_queue.join()

        # Stop workers
        for _ in range(num_workers):
            work_queue.put(None)

        for worker in workers:
            worker.join()

        # Close interactive session
        if self.interactive_session:
            try:
                self.interactive_session.sendline("exit")
                self.interactive_session.expect(pexpect.EOF, timeout=60)
            except Exception:
                self.interactive_session.close(force=True)
        # Final summary
        print("\n=== SUBMISSION SUMMARY ===")
        print(f"Total runs: {total_runs}")
        print(f"Processed: {actual_runs}")
        print(f"Timing results cached: {cached_count}")
        print(f"New timing computations: {new_count}")


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
        print(f"  Queue Job Wait time: {cfg.queue_job_wait_time} seconds")
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
