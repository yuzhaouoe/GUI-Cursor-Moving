gpu_num=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected GPU count: $gpu_num"

export VLLM_USE_V1=1

PYTHONUNBUFFERED=1 python -m cursor.rl.entry \
--config-path=/mnt/ceph_rbd/GUI-Cursor-Moving/cursor/rl/config/ \
--config-name=default_config \
 custom_reward_function.path=./cursor/rl/reward_funcs.py \
 custom_reward_function.name=get_traj_reward \
 reward_model.enable=False \
 algorithm.adv_estimator=grpo \
 algorithm.use_kl_in_reward=false \
 data.reward_fn_key=overall \
 data.custom_cls.path=./cursor/rl/cursor_dataset.py \
 data.custom_cls.name=CursorStateDataset \
 data.train_files=/mnt/ceph_rbd/data/ScreenSpot-v2/screenspot_v2_processed_full.parquet \
 data.val_files=/mnt/ceph_rbd/data/ScreenSpot-v2/screenspot_v2_processed_full.parquet \
 data.train_batch_size=6 \
 data.max_prompt_length=16384 \
 data.max_response_length=256 \
 actor_rollout_ref.model.path=/mnt/ceph_rbd/models/Qwen/Qwen2.5-VL-7B-Instruct_resaved \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=1 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
 actor_rollout_ref.rollout.mode=async \
 actor_rollout_ref.rollout.max_model_len=32768 \
 actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
 actor_rollout_ref.rollout.n=8 \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=48 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
 trainer.logger=console \
 trainer.val_before_train=False \
 trainer.n_gpus_per_node=$gpu_num \
 trainer.nnodes=1 \
 trainer.save_freq=10 \
 trainer.test_freq=10 \
 trainer.total_epochs=15 2>&1 | tee verl_demo.log

#  actor_rollout_ref.model.path=/mnt/ceph_rbd/models/Qwen/Qwen2.5-VL-7B-Instruct_resaved \
# actor_rollout_ref.model.path=/mnt/ceph_rbd/models/GUI-Cursor_resaved