MODEL_PATH=/mnt/ceph_rbd/models/new_trained/GUI-Cursor-ablation1step-Qwen3-VL-2B-Thinking
EXP_NAME="Qwen3-VL-2B-Thinking-1210-180steps-ablation1step"

GPU_IDX=4
BASE_MODEL="qwen3"

# CUDA_VISIBLE_DEVICES=${GPU_IDX} python cursor/move_loop.py \
#     --dataset_name "ScreenSpot-Pro" \
#     --base_model ${BASE_MODEL} \
#     --batch_size 128 \
#     --exp_name debug \
#     --use_vllm \
#     --use_async \
#     --model_path ${MODEL_PATH} \
#     --max_steps 1

dataset_name=("ScreenSpot-v2" "ScreenSpot-Pro" "OSWorld-G_refined" "UI-Vision")

# EXP_NAME="Qwen3-VL-2B-Thinking-1210-180steps-factor07"

for ds in "${dataset_name[@]}"; do
  CUDA_VISIBLE_DEVICES=${GPU_IDX} python cursor/move_loop.py \
    --dataset_name ${ds} \
    --base_model ${BASE_MODEL} \
    --batch_size 128 \
    --exp_name ${EXP_NAME} \
    --use_vllm \
    --use_async \
    --model_path ${MODEL_PATH} \
    --max_steps 1
done
    # --cursor_reshape_factor 0.7 \