CUDA_VISIBLE_DEVICES=1 python cursor/move_loop.py \
  --model_path /mnt/ceph_rbd/models/GUI-Cursor \
  --dataset_name ScreenSpot-v2 \
  --max_steps 3 \
  --max_examples 512 \
  --move_until_max_steps \
  --disable_ccf \
  --batch_size 128 \
  --use_vllm \
  --use_async \
  --exp_name speed_test_move3step


CUDA_VISIBLE_DEVICES=1 python cursor/move_loop.py \
  --model_path /mnt/ceph_rbd/models/GUI-Cursor \
  --dataset_name ScreenSpot-v2 \
  --max_steps 2 \
  --max_examples 512 \
  --move_until_max_steps \
  --disable_ccf \
  --batch_size 128 \
  --use_vllm \
  --use_async \
  --exp_name speed_test_move2step


CUDA_VISIBLE_DEVICES=1 python cursor/move_loop.py \
  --model_path /mnt/ceph_rbd/models/GUI-Cursor \
  --dataset_name ScreenSpot-v2 \
  --max_steps 1 \
  --max_examples 512 \
  --move_until_max_steps \
  --disable_ccf \
  --batch_size 128 \
  --use_vllm \
  --use_async \
  --exp_name speed_test_move1step

