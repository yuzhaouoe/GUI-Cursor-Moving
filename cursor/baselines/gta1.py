import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from vllm import LLM
from vllm import SamplingParams
from qwen_vl_utils import smart_resize
import re
import json
from pathlib import Path
from transformers import AutoProcessor
import math
from cursor.prepare_data import load_iterable_dataset

SYSTEM_PROMPT = """
You are an expert UI element locator. Given a GUI image and a user's element description, provide the coordinates of the specified element as a single (x,y) point. The image resolution is height {height} and width {width}. For elements with area, return the center point.

Output the coordinate pair exactly:
(x,y)
"""
SYSTEM_PROMPT = SYSTEM_PROMPT.strip()


def is_within_bbox(x, y, bbox_coords):
    top_left_x, top_left_y, bottom_right_x, bottom_right_y = bbox_coords
    return top_left_x <= x <= bottom_right_x and top_left_y <= y <= bottom_right_y


# Function to extract coordinates from model output
def extract_coordinates(raw_string):
    try:
        matches = re.findall(r"\((-?\d*\.?\d+),\s*(-?\d*\.?\d+)\)", raw_string)
        return [tuple(map(int, match)) for match in matches][0]
    except:
        return 0, 0


def prepare_input(sample, processor):
    image = sample["image"]
    instruction = sample["query"]
    bbox_proportions = sample["bbox_proportions"]

    width, height = image.width, image.height
    resized_height, resized_width = smart_resize(
        image.height,
        image.width,
        factor=processor.image_processor.patch_size * processor.image_processor.merge_size,
        min_pixels=processor.image_processor.min_pixels,
        max_pixels=processor.image_processor.max_pixels,
    )
    resized_image = image.resize((resized_width, resized_height))
    bbox_coords = [
        int(bbox_proportions[0] * resized_width),
        int(bbox_proportions[1] * resized_height),
        int(bbox_proportions[2] * resized_width),
        int(bbox_proportions[3] * resized_height),
    ]

    # bbox_coords = bbox_coords_from_proportions(resized_width, resized_height, bbox_proportions)

    # Prepare system and user messages
    system_message = {"role": "system", "content": SYSTEM_PROMPT.format(height=image.height, width=image.width)}

    user_message = {
        "role": "user",
        "content": [{"type": "image", "image": image}, {"type": "text", "text": instruction}],
    }

    message_inputs = [system_message, user_message]

    prompt = processor.apply_chat_template(message_inputs, tokenize=False, add_generation_prompt=True)
    inputs = {
        "prompt": prompt,
        "multi_modal_data": {"image": [image]},
    }
    return {
        "inputs": inputs,
        "bbox_coords": bbox_coords,
        "item_idx": sample["item_idx"],
        "data_source": sample["data_source"],
        "image": image,
        "query": instruction,
    }


# def crop_from_results(image, pred_coord, bbox_proportions, processor):
# resized_height, resized_width = smart_resize(
#     image.height,
#     image.width,
#     factor=processor.image_processor.patch_size * processor.image_processor.merge_size,
#     min_pixels=processor.image_processor.min_pixels,
#     max_pixels=processor.image_processor.max_pixels,
# )
# resized_image = image.resize((resized_width, resized_height))

# cursor_x_ratio = pred_coord[0] / resized_width
# cursor_y_ratio = pred_coord[1] / resized_height

# org_width, org_height = image.size
# cursor_org_x = int(org_width * cursor_x_ratio)
# cursor_org_y = int(org_height * cursor_y_ratio)

# cropped_image_pixels = 1920 * 1080  # 2560 * 1440
# crop_width = int(math.sqrt((cropped_image_pixels * org_width) / org_height))
# crop_height = int(math.sqrt((cropped_image_pixels * org_height) / org_width))
# # if -5 < (crop_width - 2560) < 5 and -5 < (crop_height - 1440) < 5:
#     crop_width, crop_height = 2560, 1440

# if crop_width >= org_width or crop_height >= org_height:
#     crop_width = org_width
#     crop_height = org_height
#     cropped_image = image
#     cropped_bbox_proportions = bbox_proportions
# else:
#     cropped_image, left, top = crop_image_at_coordinate(
#         image, cursor_org_x, cursor_org_y, crop_width, crop_height
#     )
#     resized_height, resized_width = smart_resize(
#         cropped_image.height,
#         cropped_image.width,
#         factor=processor.image_processor.patch_size * processor.image_processor.merge_size,
#         min_pixels=processor.image_processor.min_pixels,
#         max_pixels=processor.image_processor.max_pixels,
#     )
#     cropped_image.resize((resized_width, resized_height))
#     cropped_bbox_coords = [org_bbox_coords[0] - left, org_bbox_coords[1] - top,
#                            org_bbox_coords[2] - left, org_bbox_coords[3] - top]
#     cropped_bbox_proportions = [cropped_bbox_coords[0] / crop_width,
#                                 cropped_bbox_coords[1] / crop_height,
#                                 cropped_bbox_coords[2] / crop_width,
#                                 cropped_bbox_coords[3] / crop_height]
# return cropped_image, cropped_bbox_proportions


def inference():
    max_new_tokens = 32
    # dataset_name = "UI-Vision"
    dataset_name = "OSWorld-G_refined"

    exp_name = "GTA1-7B"

    model_path = "/mnt/ceph_rbd/models/GTA1-7B"
    batch_size = 128

    model = LLM(
        model=model_path,
        tensor_parallel_size=1,
        max_model_len=32768,
        limit_mm_per_prompt={"image": 1, "video": 0, "audio": 0},
        max_num_seqs=batch_size,
        disable_log_stats=False,
        mm_processor_cache_gb=0,
    )

    sampling_kwargs = {
        "detokenize": True,
        "temperature": 0.0,
        "max_tokens": max_new_tokens,
    }
    sampling_params = SamplingParams(**sampling_kwargs)

    processor = AutoProcessor.from_pretrained(model_path, min_pixels=3136, max_pixels=4096 * 2160)

    dataset = load_iterable_dataset(dataset_name)
    batch = []
    predictions = []
    correctness = []

    output_dir = Path("./outputs") / dataset_name / exp_name
    os.makedirs(output_dir, exist_ok=True)
    save_file = "predictions.jsonl"

    while True:
        while len(batch) < batch_size:
            sample = next(dataset, None)
            if sample is None:
                break
            batch.append(prepare_input(sample, processor))
        if len(batch) == 0:
            break

        batch_input = [item["inputs"] for item in batch]
        batch_target = [item["bbox_coords"] for item in batch]
        batch_item_idx = [item["item_idx"] for item in batch]
        batch_data_source = [item["data_source"] for item in batch]
        outputs = model.generate(batch_input, sampling_params=sampling_params)
        llm_responses = [cur_output.outputs[0].text for cur_output in outputs]

        cur_preds = [extract_coordinates(t) for t in llm_responses]
        cur_correct = [is_within_bbox(c[0], c[1], bcs) for c, bcs in zip(cur_preds, batch_target)]
        correctness.extend(cur_correct)
        predictions.extend(cur_preds)

        print(f"total: {len(correctness)}, avg acc: {sum(correctness) / len(correctness):.3%}")
        with open(output_dir / save_file, "a") as f:
            for item_idx, p, c, s in zip(batch_item_idx, cur_preds, cur_correct, batch_data_source):
                f.write(
                    json.dumps({"item_idx": item_idx, "prediction": p, "within_bbox_history": [c], "data_source": s}) + "\n"
                )

        batch = []


if __name__ == "__main__":
    inference()
