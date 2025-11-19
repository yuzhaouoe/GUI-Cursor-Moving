# # python cursor/move_loop.py --batch_size=16

# # python cursor/move_loop.py --batch_size=32

# python cursor/move_loop.py --batch_size=32 --tp_size=4

# # python cursor/move_loop.py --batch_size=64

export CUDA_HOME=/usr/local/cuda-12.8 && export PATH=$CUDA_HOME/bin:$PATH

python cursor/move_loop.py --batch_size 20 --tp_size 1 --use_vllm --exp_name sync-vllm-b20
python cursor/move_loop.py --batch_size 20 --tp_size 1 --use_vllm --exp_name sync-vllm-b64
python cursor/move_loop.py --batch_size 128 --tp_size 1 --use_vllm --exp_name sync-vllm-b128
python cursor/move_loop.py --batch_size 256 --tp_size 1 --use_vllm --exp_name sync-vllm-b256

python cursor/move_loop.py --batch_size 20 --tp_size 1 --use_async --use_vllm --exp_name async-vllm-b20
python cursor/move_loop.py --batch_size 20 --tp_size 1 --use_async --use_vllm --exp_name async-vllm-b64
python cursor/move_loop.py --batch_size 128 --tp_size 1 --use_async --use_vllm --exp_name async-vllm-b128
python cursor/move_loop.py --batch_size 256 --tp_size 1 --use_async --use_vllm --exp_name async-vllm-b256

