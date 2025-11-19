# /mnt/ceph_rbd/models/ByteDance-Seed/UI-TARS-1.5-7B
python cursor/move_loop.py \
  --dataset_name OSWorld-G_refined \
  --batch_size 128 \
  --exp_name uitars-newsys \
  --use_vllm \
  --use_async \
  --model_path /mnt/ceph_rbd/models/ByteDance-Seed/UI-TARS-1.5-7B \
  --max_steps 1

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


