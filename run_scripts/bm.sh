#!/bin/bash
# ---------- PBS DIRECTIVES ----------
#PBS -A insitu
#PBS -q by-gpu
#PBS -l select=1:ncpus=8:gputype=A100:system=sophia
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:grand
#PBS -o /grand/insitu/cohanlon/kan_inr/logs/
#PBS -e /grand/insitu/cohanlon/kan_inr/logs/
# -----------------------------------
# For running single jobs

source /grand/insitu/cohanlon/miniconda3/etc/profile.d/conda.sh 
cd /grand/insitu/cohanlon/kan_inr
conda activate kan_inr

# Get the number of GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Number of GPUs detected: $NUM_GPUS on host $(hostname)"

# Use the benchmark.py program to fit an INR, then decompress and save the result
torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS \
    benchmark.py -cn config params_file=params
