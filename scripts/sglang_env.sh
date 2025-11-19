# conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
# conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
# conda create -n cursor python=3.11 -y
# CONDA_BASE=$(conda info --base)
# source "$CONDA_BASE/etc/profile.d/conda.sh"
# conda activate cursor

pip install qwen_vl_utils
pip install openai
pip install tenacity
pip install huggingface_hub
pip install "sglang[all]>=0.5.3rc0"
pip install flashinfer-python
pip install flash-attn --no-build-isolation
pip uninstall pynvml -y
pip install -e .