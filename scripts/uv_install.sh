git clone https://github.com/yuzhaouoe/verl.git
uv pip install vllm==0.11.0
uv pip install flash-attn --no-build-isolation
uv pip install flashinfer-python
uv pip install qwen_vl_utils
uv pip install ipython
uv pip install debugpy
uv pip install -e .
uv pip install -e ./verl
uv pip install transformers==4.57.3
uv pip install flask
uv pip install 'nixl[cu12]'

cat >> ~/.bashrc << 'EOF'

# CUDA configuration
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
EOF

apt-get install byobu -y
apt-get install nvtop -y

git config --global user.email yuzhaouoe@gmail.com
git config --global user.name yuzhaouoe