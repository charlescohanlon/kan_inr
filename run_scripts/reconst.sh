#!/bin/bash
# ---------- PBS DIRECTIVES ----------
#PBS -A insitu
#PBS -q by-gpu
#PBS -l select=8:ncpus=8:gputype=A100:system=sophia
#PBS -l walltime=08:00:00
#PBS -l filesystems=home:grand
#PBS -o /grand/insitu/cohanlon/alcf_kan_inr/logs/
#PBS -e /grand/insitu/cohanlon/alcf_kan_inr/logs/
#PBS -m b
#PBS -M charlescohanlon@gmail.com
# -----------------------------------

source /grand/insitu/cohanlon/miniconda3/etc/profile.d/conda.sh 
cd /grand/insitu/cohanlon/alcf_kan_inr
conda activate alcf_kan_inr

# Get the number of GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Number of GPUs detected: $NUM_GPUS on host $(hostname)"

# Use the benchmark.py program to fit an INR and run reconstruction for it
torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS \
    benchmark.py -cn config \
        repeats=1 \
        params_file="alcf_kan_inr/params.json" \
        dataset="beechnut" \
        hashmap_size=19 \
        epochs=5000 \
        checkpoint_freq=1000 \
        checkpoint_save=True \
        checkpoint_eval=True \