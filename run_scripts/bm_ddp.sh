#!/bin/bash
# ---------- PBS DIRECTIVES ----------
#PBS -A insitu
#PBS -q by-node
#PBS -l select=1:ncpus=64:gputype=A100:system=sophia
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:grand
#PBS -l place=scatter
#PBS -o /grand/insitu/cohanlon/alcf_kan_inr/logs/
#PBS -e /grand/insitu/cohanlon/alcf_kan_inr/logs/
# -----------------------------------

source /grand/insitu/cohanlon/miniconda3/etc/profile.d/conda.sh 
cd /grand/insitu/cohanlon/alcf_kan_inr
conda activate alcf_kan_inr

# Get the number of GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Number of GPUs detected: $NUM_GPUS"

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Running with DDP on $NUM_GPUS GPUs"
    # Use torchrun for DDP
    torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS \
        benchmark_ddp.py -cn config
else
    echo "Running on single GPU or CPU"
    # Run without DDP
    python benchmark.py -cn config
fi