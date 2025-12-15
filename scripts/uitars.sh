# /mnt/ceph_rbd/models/ByteDance-Seed/UI-TARS-1.5-7B

EXP_NAME="uitars-from100-hint-to-move-to200steps"
MODEL_PATH="/mnt/ceph_rbd/models/new_trained/GUI-Cursor-uitars-from100-hint-to-move-to200steps"
GPU_IDX=4

CUDA_VISIBLE_DEVICES=${GPU_IDX} python cursor/move_loop.py \
  --dataset_name ScreenSpot-v2 \
  --base_model uitars \
  --batch_size 128 \
  --exp_name ${EXP_NAME} \
  --use_vllm \
  --use_async \
  --model_path ${MODEL_PATH} \
  --max_steps 4


CUDA_VISIBLE_DEVICES=${GPU_IDX} python cursor/move_loop.py \
  --dataset_name ScreenSpot-Pro \
  --base_model uitars \
  --batch_size 128 \
  --exp_name ${EXP_NAME} \
  --use_vllm \
  --use_async \
  --model_path ${MODEL_PATH} \
  --max_steps 4


CUDA_VISIBLE_DEVICES=${GPU_IDX} python cursor/move_loop.py \
  --dataset_name OSWorld-G_refined \
  --base_model uitars \
  --batch_size 128 \
  --exp_name ${EXP_NAME} \
  --use_vllm \
  --use_async \
  --model_path ${MODEL_PATH} \
  --max_steps 4


CUDA_VISIBLE_DEVICES=0 python cursor/move_loop.py \
  --dataset_name UI-Vision \
  --base_model uitars \
  --batch_size 128 \
  --exp_name ${EXP_NAME} \
  --use_vllm \
  --use_async \
  --model_path /mnt/ceph_rbd/models/new_trained/GUI-Cursor-uitars-from100-hint-to-move-to260steps \
  --max_steps 4


# CUDA_VISIBLE_DEVICES=4 python cursor/move_loop.py \
#   --dataset_name ScreenSpot-Pro \
#   --base_model uitars \
#   --batch_size 128 \
#   --exp_name GUI-Cursor-uitars-160steps_focus05 \
#   --cursor_focus_sizes 0.5 \
#   --use_vllm \
#   --use_async \
#   --model_path /mnt/ceph_rbd/models/new_trained/GUI-Cursor-uitars-160steps \
#   --max_steps 4

# CUDA_VISIBLE_DEVICES=4 python cursor/move_loop.py \
#   --dataset_name UI-Vision \
#   --base_model uitars \
#   --batch_size 128 \
#   --exp_name GUI-Cursor-uitars-160steps_focus05 \
#   --cursor_focus_sizes 0.5 \
#   --use_vllm \
#   --use_async \
#   --model_path /mnt/ceph_rbd/models/new_trained/GUI-Cursor-uitars-160steps \
#   --max_steps 4

# CUDA_VISIBLE_DEVICES=4 python cursor/move_loop.py \
#   --dataset_name ScreenSpot-v2 \
#   --base_model uitars \
#   --batch_size 128 \
#   --exp_name GUI-Cursor-uitars-160steps \
#   --use_vllm \
#   --use_async \
#   --model_path /mnt/ceph_rbd/models/new_trained/GUI-Cursor-uitars-160steps \
#   --max_steps 4

# CUDA_VISIBLE_DEVICES=4 python cursor/move_loop.py \
#   --dataset_name ScreenSpot-v2 \
#   --base_model uitars \
#   --batch_size 128 \
#   --exp_name GUI-Cursor-uitars-160steps_focus05 \
#   --cursor_focus_sizes 0.5 \
#   --use_vllm \
#   --use_async \
#   --model_path /mnt/ceph_rbd/models/new_trained/GUI-Cursor-uitars-160steps \
#   --max_steps 4


# CUDA_VISIBLE_DEVICES=4 python cursor/move_loop.py \
#   --dataset_name UI-Vision \
#   --base_model uitars \
#   --batch_size 128 \
#   --exp_name GUI-Cursor-uitars-160steps \
#   --use_vllm \
#   --use_async \
#   --model_path /mnt/ceph_rbd/models/new_trained/GUI-Cursor-uitars-160steps \
#   --max_steps 4


# CUDA_VISIBLE_DEVICES=4 python cursor/move_loop.py \
#   --dataset_name OSWorld-G_refined \
#   --base_model uitars \
#   --batch_size 128 \
#   --exp_name GUI-Cursor-uitars-180steps \
#   --use_vllm \
#   --use_async \
#   --model_path /mnt/ceph_rbd/models/new_trained/GUI-Cursor-uitars-180steps \
#   --max_steps 4
  
# -withoutccf

# CUDA_VISIBLE_DEVICES=0 python cursor/move_loop.py \
#   --dataset_name ScreenSpot-Pro \
#   --base_model uitars \
#   --batch_size 128 \
#   --exp_name GUI-Cursor-uitars-120steps \
#   --use_vllm \
#   --use_async \
#   --model_path /mnt/ceph_rbd/models/new_trained/GUI-Cursor-uitars-120steps \
#   --max_steps 4


# CUDA_VISIBLE_DEVICES=0 python cursor/move_loop.py \
#   --dataset_name Multimodal-Mind2Web \
#   --base_model qwen \
#   --batch_size 128 \
#   --exp_name GUI-Cursor_3step \
#   --use_vllm \
#   --use_async \
#   --model_path /mnt/ceph_rbd/models/GUI-Cursor \
#   --max_steps 3



# pkill -9 -f "VLLM::EngineCore"
# ray stop --force
# pkill -9 -f ray

# python cursor/move_loop.py \
#   --dataset_name OSWorld-G_refined \
#   --batch_size 128 \
#   --exp_name uitars-newsys \
#   --use_vllm \
#   --use_async \
#   --model_path /mnt/ceph_rbd/models/ByteDance-Seed/UI-TARS-1.5-7B \
#   --max_steps 1

## python cursor/move_loop.py \
#   --dataset_name OSWorld-G_refined \
#   --batch_size 128 \
#   --exp_name uitars-175-withccf-focus05 \
#   --use_vllm \
#   --use_async \
#   --model_path /mnt/ceph_rbd/models/backup/GUI-Cursor-UI-Tars-175step \
#   --max_steps 4

# python cursor/move_loop.py \
#   --dataset_name UI-Vision \
#   --batch_size 128 \
#   --exp_name uitars-175-withccf-focus05 \
#   --use_vllm \
#   --use_async \
#   --model_path /mnt/ceph_rbd/models/backup/GUI-Cursor-UI-Tars-175step \
#   --max_steps 4


# python cursor/move_loop.py \
#   --dataset_name ScreenSpot \
#   --batch_size 128 \
#   --exp_name uitars-175-withccf \
#   --use_vllm \
#   --use_async \
#   --model_path /mnt/ceph_rbd/models/backup/GUI-Cursor-UI-Tars-175step \
#   --max_steps 4


# python cursor/move_loop.py \
#   --dataset_name ScreenSpot-Pro \
#   --batch_size 128 \
#   --exp_name uitars-175-withccf \
#   --use_vllm \
#   --use_async \
#   --model_path /mnt/ceph_rbd/models/backup/GUI-Cursor-UI-Tars-175step \
#   --max_steps 4


