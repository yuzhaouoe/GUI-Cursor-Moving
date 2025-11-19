NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected GPU count: $NUM_GPUS"

export VLLM_USE_V1=1

# export RAY_DEBUG_POST_MORTEM=1
# export YUZHAO_DEBUG=1

MAX_TURNS=1
TRAIN_BATCH_SIZE=32
ROLLOUT_N=8
AGENT_NUM_WORKERS=8
# PPO_MINI_BATCH_SIZE=$((TRAIN_BATCH_SIZE * ROLLOUT_N))
# ACTOR_DTYPE=bfloat16
ACTOR_DTYPE=fp32
EXP_NAME=onestep-bs32
PROJECT_NAME=GUI-Cursor
DATALOADER_NUM_WORKERS=5

PYTHONUNBUFFERED=1 python -m cursor.rl.entry \
--config-path=/mnt/ceph_rbd/GUI-Cursor-Moving/cursor/rl/config/ \
--config-name=default_config \
 online_filtering=true \
 max_times_to_make_a_batch=2 \
 custom_reward_function.path=./cursor/rl/reward_funcs.py \
 custom_reward_function.name=calculate_rewards \
 reward_model.enable=False \
 algorithm.adv_estimator=grpo \
 algorithm.use_kl_in_reward=false \
 data.val_batch_size=64 \
 data.reward_fn_key=overall \
 data.custom_cls.path=./cursor/rl/cursor_dataset.py \
 data.custom_cls.name=CursorStateDataset \
 data.train_files=/mnt/ceph_rbd/data/grounding_train_0714-clean/grounding_train_0714-clean_processed_full.parquet \
 data.val_files=/mnt/ceph_rbd/data/ScreenSpot-v2/screenspot_v2_processed_full.parquet \
 data.train_batch_size=$TRAIN_BATCH_SIZE \
 data.max_prompt_length=3072 \
 data.max_response_length=4096 \
 data.return_raw_chat=true \
 data.dataloader_num_workers=$DATALOADER_NUM_WORKERS \
 actor_rollout_ref.actor.fsdp_config.model_dtype=$ACTOR_DTYPE \
 actor_rollout_ref.model.path=/mnt/ceph_rbd/models/Qwen/Qwen2.5-VL-7B-Instruct_resaved \
 actor_rollout_ref.actor.use_kl_loss=false \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
 actor_rollout_ref.rollout.mode=async \
 actor_rollout_ref.rollout.max_model_len=32768 \
 actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
 actor_rollout_ref.rollout.n=$ROLLOUT_N \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
 actor_rollout_ref.rollout.agent.num_workers=$AGENT_NUM_WORKERS \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
 actor_rollout_ref.rollout.agent.default_agent_loop=cursor_agent_loop \
 actor_rollout_ref.rollout.multi_turn.enable=true \
 actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$MAX_TURNS \
 actor_rollout_ref.rollout.multi_turn.max_user_turns=$MAX_TURNS \
 cursor.max_steps=$MAX_TURNS \
 trainer.logger='["console","wandb"]' \
 trainer.val_before_train=false \
 trainer.n_gpus_per_node=$NUM_GPUS \
 trainer.nnodes=1 \
 trainer.project_name=$PROJECT_NAME \
 trainer.experiment_name=$EXP_NAME \
 trainer.save_freq=50 \
 trainer.test_freq=20 \
 trainer.total_epochs=250 2>&1 | tee verl_demo.log


# pkill -9 -f "VLLM::EngineCore"
# ray stop --force
# pkill -9 -f ray

#  trainer.validation_data_dir=./validation_outputs/${PROJECT_NAME}/${EXP_NAME} \
#  actor_rollout_ref.rollout.free_cache_engine=false \
# screenspot_v2_processed_50.parquet
#  actor_rollout_ref.model.path=/mnt/ceph_rbd/models/Qwen/Qwen2.5-VL-7B-Instruct_resaved \
# actor_rollout_ref.model.path=/mnt/ceph_rbd/models/GUI-Cursor_resaved
# actor_rollout_ref.model.path=/mnt/ceph_rbd/models/Qwen/Qwen3-VL-2B-Instruct
# /mnt/ceph_rbd/models/Qwen/Qwen2.5-VL-3B-Instruct_resaved