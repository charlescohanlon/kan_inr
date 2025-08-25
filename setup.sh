# A100 GPU compute capability
export TCNN_CUDA_ARCHITECTURES=80

# Location cuda toolkit is installed (the env root)
export CUDA_HOME=/lus/grand/projects/insitu/cohanlon/miniconda3/envs/kan_inr

export HOME_DIR=/grand/insitu/cohanlon # CHANGE ME

# Assumes miniconda installed
source $HOME_DIR/miniconda3/etc/profile.d/conda.sh 
conda create -n kan_inr python=3.10 -y
conda activate kan_inr

# Install CUDA Toolkit
conda install nvidia/label/cuda-12.1.0::cuda-toolkit -c nvidia/label/cuda-12.1.0 -y

# Install appropriate pytorch (with CUDA support)
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia -y

# A stub library path (necessary b/c this installation can't be done on compute node w/o network connection)
export STUBS_DIR=$CUDA_HOME/lib/stubs
export LIBRARY_PATH=$STUBS_DIR:$CUDA_HOME/targets/x86_64-linux/lib:${LIBRARY_PATH}

# Install TCNN python bindings
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# Install other dependencies
pip install -r requirements.txt

echo "At this point you might need to reload your shell."