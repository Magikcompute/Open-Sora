# create a virtual env
# conda create -n opensora python=3.10
# # activate virtual environment
# conda activate opensora

set -e
# install torch
# the command below is for CUDA 12.1, choose install commands from
# https://pytorch.org/get-started/locally/ based on your own CUDA version
pip install torch torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple

# install flash attention (optional)
# set enable_flashattn=False in config to avoid using flash attention
pip install packaging ninja -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install flash-attn --no-build-isolation -i https://pypi.tuna.tsinghua.edu.cn/simple

# install apex (optional)
# set enable_layernorm_kernel=False in config to avoid using apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" git+https://github.com/NVIDIA/apex.git 


# install xformers
pip install -U xformers --index-url https://download.pytorch.org/whl/cu121

# install this project
pip install -v -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
