from vllm import LLM
from vllm import SamplingParams
from transformers import AutoProcessor
import re
from qwen_vl_utils import smart_resize

import json
import time

from regex import F
from cursor.prepare_data import load_iterable_dataset
from tqdm import tqdm
import math
from collections import Counter, defaultdict

import os
from PIL import ImageDraw, Image

from cursor.utils.image import (
    bbox_coords_from_proportions,
    smart_resize_min_max_tokens_per_img,
    crop_image_at_coordinate,
)


def load_results():
    path = "outputs/ScreenSpot-Pro/GUI-Cursor-icml-3step-speed/predictions.jsonl"
    results = dict()
    with open(path, "r") as f:
        for line in f:
            data = json.loads(line)
            results[data["item_idx"]] = data
    return results


def crop_logic(
    bbox_proportions,
    original_screenshot,
    cursor_x,
    cursor_y,
    cursor_focus_sizes=1920 * 1080,
):
    if cursor_focus_sizes is None:
        return False

    org_width, org_height = original_screenshot.size

    screenshot = smart_resize_min_max_tokens_per_img(
        image=original_screenshot, min_tokens_per_img=2048, max_tokens_per_img=10240
    )

    if org_width * org_height <= 2073600:  # 2073600 1920*1080, do not apply ccf when processing low resolution
        return False

    if org_height > org_width:  # do not applied for mobile images
        return False

    width, height = screenshot.size
    cursor_x_ratio = cursor_x / width
    cursor_y_ratio = cursor_y / height
    org_bbox_coords = bbox_coords_from_proportions(org_width, org_height, bbox_proportions)

    # focus to one monitor
    if (
        (org_width == 3840 and org_height == 1080) or (org_width == 5120 and org_height == 1440)
    ) and cursor_x_ratio != 0.5:
        if cursor_x_ratio < 0.5:
            top, left = 0, 0
            print(f"focus to left monitor")
        else:
            top, left = 0, org_width // 2
            print(f"focus to right monitor")
        # (left, top, right, bottom)
        cropped_image = original_screenshot.crop((left, top, left + org_width // 2, top + org_height))
        crop_width, crop_height = cropped_image.size
    else:
        cursor_org_x = int(org_width * cursor_x_ratio)
        cursor_org_y = int(org_height * cursor_y_ratio)
        if cursor_focus_sizes > 1.0:
            cropped_image_pixels = cursor_focus_sizes
            crop_width = int(math.sqrt((cropped_image_pixels * org_width) / org_height))
            crop_height = int(math.sqrt((cropped_image_pixels * org_height) / org_width))
        else:
            cropped_factor = cursor_focus_sizes
            crop_width = int(org_width * cropped_factor)
            crop_height = int(org_height * cropped_factor)

        if crop_width >= org_width or crop_height >= org_height:
            return False

        cropped_image, left, top = crop_image_at_coordinate(
            original_screenshot, cursor_org_x, cursor_org_y, crop_width, crop_height
        )
    # print(f"env_size: {self.environment.screenshot.size}")
    # print(f"cursor_x, cursor_y: {self.cursor_x}, {self.cursor_y}")
    # print(f"org_size: {self.environment.original_screenshot.size}, cursor_org_x_y = {cursor_org_x}, {cursor_org_y}")
    print(f"Focus, original: {org_width}x{org_height}, left-top: ({left}, {top}), crop to {crop_width}x{crop_height}")

    cropped_bbox_coords = [
        org_bbox_coords[0] - left,
        org_bbox_coords[1] - top,
        org_bbox_coords[2] - left,
        org_bbox_coords[3] - top,
    ]
    cropped_bbox_proportions = [
        cropped_bbox_coords[0] / crop_width,
        cropped_bbox_coords[1] / crop_height,
        cropped_bbox_coords[2] / crop_width,
        cropped_bbox_coords[3] / crop_height,
    ]

    if (
        cropped_bbox_coords[0] < 0
        or cropped_bbox_coords[1] < 0
        or cropped_bbox_coords[2] > crop_width
        or cropped_bbox_coords[3] > crop_height
    ):
        target_in_cropped_image = False
    else:
        target_in_cropped_image = True

    return {
        "cropped_image": cropped_image,
        "cropped_bbox_proportions": cropped_bbox_proportions,
        "input_screenshot": screenshot,
        "crop_coordinates": (left, top, left + crop_width, top + crop_height),
        "target_in_cropped_image": target_in_cropped_image,
    }


def save_crop_image_to_debug(save_name, screenshot, cropped_image, crop_coordinates, bbox_proportions):
    save_dir = "judge_taget_is_in_image/debug"
    os.makedirs(save_dir, exist_ok=True)
    draw = ImageDraw.Draw(screenshot)
    bbox_coordinates = bbox_coords_from_proportions(screenshot.size[0], screenshot.size[1], bbox_proportions)
    draw.rectangle(crop_coordinates, outline="red", width=5)
    draw.rectangle(bbox_coordinates, outline="blue", width=5)
    screenshot.save(os.path.join(save_dir, f"{save_name}.png"))
    cropped_image.save(os.path.join(save_dir, f"{save_name}_cropped.png"))


def create_judge_dataset():

    results = load_results()

    dataset_name = "ScreenSpot-Pro"
    dataset = load_iterable_dataset(dataset_name)

    save_image_dir = "judge_taget_is_in_image/dataset/images"
    os.makedirs(save_image_dir, exist_ok=True)
    save_data_file = "judge_taget_is_in_image/dataset/judge_dataset.jsonl"

    judge_dataset = []

    for data in tqdm(dataset):
        item_idx = data["item_idx"]
        cur_pred = results[item_idx]
        crop_left_top_history = cur_pred["focus_left_top_history"]

        bbox_proportions = data["bbox_proportions"]
        original_screenshot = data["image"]
        if len(cur_pred["position_history_global"]) == 0:
            cursor_x, cursor_y = cur_pred["position_history"][1]
        else:
            cursor_x, cursor_y = cur_pred["position_history_global"][0][1]

        crop_info = crop_logic(
            bbox_proportions,
            original_screenshot,
            cursor_x,
            cursor_y,
            cursor_focus_sizes=1920 * 1080,
        )

        if crop_info is False:
            assert len(crop_left_top_history) == 0
            continue

        crop_left_top = crop_left_top_history[0]
        assert crop_left_top[0] == crop_info["crop_coordinates"][0]
        assert crop_left_top[1] == crop_info["crop_coordinates"][1]

        input_screenshot = crop_info["input_screenshot"]
        # save_crop_image_to_debug(
        #     f"item_{item_idx}", input_screenshot, crop_info["cropped_image"], crop_info["crop_coordinates"], data["bbox_proportions"]
        # )
        # if item_idx > 15:
        #     break

        cur_judge_item = {
            "item_idx": item_idx,
            "query": data["query"],
            "target_in_cropped_image": crop_info["target_in_cropped_image"],
        }

        cropped_image = crop_info["cropped_image"]

        cropped_image.save(os.path.join(save_image_dir, f"item_{item_idx}_cropped.png"))

        with open(save_data_file, "a") as f:
            f.write(json.dumps(cur_judge_item) + "\n")


JUDGE_SYSTEM_PROMPT_TEMPLATE = r"""
Given a GUI screenshot and a user instruction, determine whether the target element described in the instruction is within the screenshot.

Answer with "Yes" if the target element is within the screenshot, and "No" if it is not.
""".strip()

QUERY_TEMPLATE = r"""
Instruction: {query}

Is the target element described in the instruction within the screenshot? Answer with "Yes" or "No".
""".strip()


def load_iterative_judge_data():

    data_path = "judge_taget_is_in_image/dataset/judge_dataset.jsonl"
    image_dir = "judge_taget_is_in_image/dataset/images"

    data = []
    with open(data_path, "r") as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
    print("Loaded judge dataset with {} samples.".format(len(data)))

    def iter_data():
        for item in data:

            item_idx = item["item_idx"]
            query = item["query"]
            target_in_cropped_image = item["target_in_cropped_image"]

            image_path = os.path.join(image_dir, f"item_{item_idx}_cropped.png")
            image = Image.open(image_path).convert("RGB")

            instruction = QUERY_TEMPLATE.format(query=query)

            yield {
                "item_idx": item_idx,
                "target_in_cropped_image": target_in_cropped_image,
                "image": image,
                "instruction": instruction,
            }

    return iter_data()


def prepare_input(batch_judge_data, processor):
    processed_input_batch = []
    labels = []
    item_ids = []
    for item in batch_judge_data:
        image = item["image"]
        instruction = item["instruction"]
        target_in_cropped_image = item["target_in_cropped_image"]
        labels.append(target_in_cropped_image)
        system_message = {"role": "system", "content": JUDGE_SYSTEM_PROMPT_TEMPLATE}
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
        processed_input_batch.append(inputs)
        item_ids.append(item["item_idx"])

    return processed_input_batch, labels, item_ids


def extract_answer(raw_output):
    raw_output = raw_output.strip().lower()
    if "yes" in raw_output and "no" in raw_output:
        return None
    elif "yes" not in raw_output and "no" not in raw_output:
        return None

    if "yes" in raw_output:
        return True
    elif "no" in raw_output:
        return False

    return None


def f1_score(y_true, y_pred):
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def run_judge():

    # load vllm model

    # run judge, binary classification

    model_path = "/mnt/ceph_rbd/models/GUI-Cursor"
    log_dir = "./judge_taget_is_in_image/results"
    log_path = os.path.join(log_dir, model_path.split("/")[-1] + "_judge_results.jsonl")

    if os.path.exists(log_path):
        print(f"Log file {log_path} already exists. Please remove it before running the judge.")
        exit(-1)

    model = LLM(
        model=model_path,
        tensor_parallel_size=1,
        max_model_len=32768,
        limit_mm_per_prompt={"image": 1, "video": 0, "audio": 0},
        max_num_seqs=128,
        disable_log_stats=False,
        mm_processor_cache_gb=0,
    )

    sampling_kwargs = {
        "detokenize": True,
        "temperature": 0.0,
        "max_tokens": 512,
    }

    sampling_params = SamplingParams(**sampling_kwargs)
    processor = AutoProcessor.from_pretrained(model_path, min_pixels=2048 * 28 * 28, max_pixels=10240 * 28 * 28)

    iter_dataset = load_iterative_judge_data()

    predictions = []
    correctness = []

    counter_preds = {"yes": 0, "no": 0, "none": 0}

    all_labels = []

    batch = []
    while True:

        while len(batch) < 128:
            sample = next(iter_dataset, None)
            if sample is None:
                break
            batch.append(sample)
        if len(batch) == 0:
            break

        batch_input, batch_labels, item_ids = prepare_input(batch, processor)
        batch = []
        outputs = model.generate(batch_input, sampling_params=sampling_params)

        for output, label, item_idx in zip(outputs, batch_labels, item_ids):
            raw_output = output.outputs[0].text
            answer = extract_answer(raw_output)

            if answer is None:
                print(f"Uncertain prediction for output: {raw_output}")

            predictions.append(answer)
            correctness.append(answer == label)
            all_labels.append(label)
            if answer is True:
                counter_preds["yes"] += 1
            elif answer is False:
                counter_preds["no"] += 1
            else:
                counter_preds["none"] += 1

            with open(log_path, "a") as f:
                cur_save_log = {
                    "item_idx": item_idx,
                    "raw_output": raw_output,
                    "extracted_answer": answer,
                    "label": label,
                    "correct": answer == label,
                }
                f.write(json.dumps(cur_save_log) + "\n")

        print(f"Processed {len(correctness)} samples. Current accuracy: {sum(correctness) / len(correctness):.3%}")
        print(f"Prediction counts so far: {counter_preds}")

    print(f"Final accuracy: {sum(correctness) / len(correctness):.3%}")
    print(f"Final prediction counts: {counter_preds}")

    # F1 score
    y_true = [1 if c else 0 for c in correctness]
    y_pred = [1 if p else 0 for p in predictions]
    f1 = f1_score(y_true, y_pred)
    print(f"F1 score: {f1:.3%}")

    # label counter
    label_counts = Counter(all_labels)
    print(f"Label distribution: {label_counts}")


def analyse_results():
    log_path = "judge_taget_is_in_image/results/GUI-Cursor_judge_results.jsonl"
    label_pred_counts = defaultdict(lambda: {"yes": 0, "no": 0, "none": 0})

    # how many "target not in image" samples are predicted as "yes", how many "target in image" samples are predicted as "no"

    target_not_in_image_pred_yes = 0
    target_in_image_pred_no = 0
    total_not_in_image = 0
    total_in_image = 0
    data = []
    with open(log_path, "r") as f:
        for line in f:
            item = json.loads(line)
            data.append(item)

    for item in data:
        label = item["label"]
        pred = item["extracted_answer"]
        if label is True:
            total_in_image += 1
            if pred is False:
                target_in_image_pred_no += 1
        elif label is False:
            total_not_in_image += 1
            if pred is True:
                target_not_in_image_pred_yes += 1
        if pred is True:
            label_pred_counts[label]["yes"] += 1
        elif pred is False:
            label_pred_counts[label]["no"] += 1
        else:
            label_pred_counts[label]["none"] += 1

    print(f"Total 'target in image' samples: {total_in_image}, 'target not in image' samples: {total_not_in_image}")
    print(
        f"'Target in image' predicted as 'no': {target_in_image_pred_no} ({target_in_image_pred_no / total_in_image:.3%})"
    )
    print(
        f"'Target not in image' predicted as 'yes': {target_not_in_image_pred_yes} ({target_not_in_image_pred_yes / total_not_in_image:.3%})"
    )
    print(f"Label-prediction counts: {label_pred_counts}")


if __name__ == "__main__":
    # create_judge_dataset()

    # run_judge()

    analyse_results()
