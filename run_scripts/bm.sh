#!/bin/bash
# ---------- PBS DIRECTIVES ----------
#PBS -A insitu
#PBS -q by-gpu
#PBS -l select=1:ncpus=8:gputype=A100:system=sophia
#PBS -l walltime=06:00:00
#PBS -l filesystems=home:grand
#PBS -l place=scatter
#PBS -o /grand/insitu/cohanlon/alcf_kan_inr/logs/
#PBS -e /grand/insitu/cohanlon/alcf_kan_inr/logs/
# -----------------------------------

source /grand/insitu/cohanlon/miniconda3/etc/profile.d/conda.sh 
cd /grand/insitu/cohanlon/alcf_kan_inr
conda activate alcf_kan_inr

python benchmark.py -cn config