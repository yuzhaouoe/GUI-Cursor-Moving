git clone https://github.com/yuzhaouoe/verl.git
pip install vllm
pip install flash-attn --no-build-isolation
pip install flashinfer-python
pip install qwen_vl_utils
pip install ipython
pip install debugpy
pip install -e .
pip install -e ./verl

cat >> ~/.bashrc << 'EOF'

# CUDA configuration
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
EOF

apt-get install byobu -y
apt-get install nvtop -y