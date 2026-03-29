# MODEL_PATH=/mnt/ceph_rbd/models/new_trained/GUI-Cursor-ablation1step-Qwen3-VL-2B-Thinking
# EXP_NAME="Qwen3-VL-2B-Thinking-1210-180steps-ablation1step"

# MODEL_PATH=/mnt/ceph_rbd/models/new_trained/GUI-Cursor-uitars-160steps
# EXP_NAME=GUI-Cursor-uitars-1step-speed
# BASE_MODEL="uitars"

MODEL_PATH=/mnt/ceph_rbd/models/GUI-Cursor
EXP_NAME=GUI-Cursor-icml-3step-speed
BASE_MODEL=qwen

MAX_STEPS=1
GPU_IDX=0

dataset_name=("ScreenSpot-Pro" "ScreenSpot-v2") #  "OSWorld-G_refined" "UI-Vision"

for ds in "${dataset_name[@]}"; do
  CUDA_VISIBLE_DEVICES=${GPU_IDX} python cursor/move_loop.py \
    --dataset_name ${ds} \
    --base_model ${BASE_MODEL} \
    --batch_size 128 \
    --exp_name ${EXP_NAME} \
    --use_vllm \
    --use_async \
    --model_path ${MODEL_PATH} \
    --max_steps ${MAX_STEPS}
done
    # --cursor_reshape_factor 0.7 \

# 1 step: disable ccf + max=1step
# 2 step: max=1step
# 3 step: max=2step