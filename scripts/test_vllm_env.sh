#!/bin/bash

# Set CUDA environment variables
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

pip install qwen_vl_utils
pip install openai
pip install tenacity
pip install huggingface_hub
pip install flashinfer-python
pip install flash-attn --no-build-isolation
pip install vllm
pip install transformers
pip install pandas
pip install datasets
pip install -e .