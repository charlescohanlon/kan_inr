#!/bin/bash

HOME_DIR=/grand/insitu/cohanlon
source $HOME_DIR/miniconda3/etc/profile.d/conda.sh 
num_runs=$(conda run -n alcf_kan_inr python $HOME_DIR/alcf_kan_inr/benchmark.py -cn config only_count_runs=True)

# Iterate over the number of runs and execute qsub with a different index for each
for run_index in $(seq 0 $((num_runs - 1))); do
    echo "Submitting run index: $run_index"

    # PBS_ARRAY_INDEX mimics job arrays
    qsub -v PBS_ARRAY_INDEX=$run_index $HOME_DIR/alcf_kan_inr/run_scripts/bm.bash
done
