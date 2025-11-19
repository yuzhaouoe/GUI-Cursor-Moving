from transformers import AutoProcessor, GenerationConfig
from transformers import Qwen2_5_VLForConditionalGeneration

model_path = "/mnt/ceph_rbd/models/Qwen/Qwen2.5-VL-3B-Instruct"
save_path = "/mnt/ceph_rbd/models/Qwen/Qwen2.5-VL-3B-Instruct_resaved"

# model_path = "/mnt/ceph_rbd/models/GUI-Cursor"
# save_path = "/mnt/ceph_rbd/models/GUI-Cursor_resaved"


model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path)
model.save_pretrained(save_path)

processor = AutoProcessor.from_pretrained(model_path)
processor.save_pretrained(save_path)

# generation_config = GenerationConfig.from_pretrained(model_path)
# generation_config.save_pretrained(save_path)