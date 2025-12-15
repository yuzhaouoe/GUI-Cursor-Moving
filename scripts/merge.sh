# python -m verl.model_merger merge \
#     --backend fsdp \
#     --local_dir checkpoints/verl_fsdp_gsm8k_examples/qwen2_5_0b5_fsdp_saveload/global_step_1/actor \
#     --target_dir /path/to/merged_hf_model


python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /mnt/ceph_rbd/GUI-Cursor-Moving/checkpoints/GUI-Cursor/uitars-bs32n12-hint-to-move/global_step_200/actor \
    --target_dir /mnt/ceph_rbd/models/new_trained/GUI-Cursor-uitars-from100-hint-to-move-to200steps

