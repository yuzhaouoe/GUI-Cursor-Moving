# python -m verl.model_merger merge \
#     --backend fsdp \
#     --local_dir checkpoints/verl_fsdp_gsm8k_examples/qwen2_5_0b5_fsdp_saveload/global_step_1/actor \
#     --target_dir /path/to/merged_hf_model


# python -m verl.model_merger merge \
#     --backend fsdp \
#     --local_dir /mnt/ceph_rbd/GUI-Cursor-Moving/checkpoints/GUI-Cursor/qwen3-bs32n16-1210-1step/global_step_180/actor \
#     --target_dir /mnt/ceph_rbd/models/new_trained/GUI-Cursor-ablation1step-Qwen3-VL-2B-Thinking


python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /mnt/ceph_rbd/GUI-Cursor-Moving/checkpoints/GUI-Cursor/qwen3-bs32n16-1210/global_step_200/actor \
    --target_dir /mnt/ceph_rbd/model/new_trained/GUI-Cursor-Qwen3-VL-2B-Thinking-1210
