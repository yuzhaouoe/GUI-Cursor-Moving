from email.mime import image
import json
import collections
from networkx import draw, predecessor
import numpy as np
import os
from PIL import Image
import string
import argparse
import re


def calculate_f1(pred, label):
    pred = set(pred.lower().strip().split())
    label = set(label.lower().strip().split())
    # remove punctuation
    pred = set([x for x in pred if x not in string.punctuation])
    label = set([x for x in label if x not in string.punctuation])
    if len(pred) == 0 and len(label) == 0:
        return 1
    if len(pred) == 0 or len(label) == 0:
        return 0

    tp = len(pred & label)
    fp = len(pred - label)
    fn = len(label - pred)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision == 0 or recall == 0:
        return 0
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def is_output_inside_bbox(bboxes, output, scale):
    output_x, output_y = output
    output_x /= scale
    output_y /= scale

    for bbox in bboxes:
        bbox_x, bbox_y, bbox_width, bbox_height = bbox
        if bbox_x <= output_x <= bbox_x + bbox_width and bbox_y <= output_y <= bbox_y + bbox_height:
            return True, (output_x, output_y)
    return False, (output_x, output_y)


def extract_coordinates(operation, image_path):
    # extract for cogagent output
    tap_match = re.search(r"tap\s*\[\[(\d+),(\d+)\]\]", operation, re.IGNORECASE)
    box_match = re.search(r"\[\[(\d+),(\d+),(\d+),(\d+)\]\]", operation)

    image = Image.open(image_path)
    width, height = image.size

    if tap_match:
        x, y = map(int, tap_match.groups())
        x = int(width * (x / 1000))
        y = int(height * (y / 1000))
        return (x, y)
    elif box_match:
        x1, y1, x2, y2 = map(int, box_match.groups())
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        center_x = int(width * (center_x / 1000))
        center_y = int(height * (center_y / 1000))
        return (center_x, center_y)
    else:
        raise ValueError("Operation format not recognized", operation)

def draw_debug(query, image_path, bboxes, output_coords, save_path):
    from PIL import ImageDraw, ImageFont
    

    image  = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for bbox in bboxes:
        bbox_x, bbox_y, bbox_width, bbox_height = bbox
        draw.rectangle([bbox_x, bbox_y, bbox_x + bbox_width, bbox_y + bbox_height], outline="red", width=2)
    output_x, output_y = output_coords
    draw.ellipse([output_x - 10, output_y - 10, output_x + 10, output_y + 10], fill="blue")
    # draw the query text # increase the font size
    font_size = 20
    try:
        default_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except:
        try:
            # Try to use load_default with size parameter (Pillow >= 10.0.0)
            default_font = ImageFont.load_default(size=font_size)
        except TypeError:
            # Fall back to old load_default() for older Pillow versions
            default_font = ImageFont.load_default()
    draw.text((10, 10), query, fill="red", font=default_font)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image.save(save_path)

def get_metrics_with_prediction(sample_data, plan_data, ans_data):
    all_element_acc = []
    all_operation_f1 = []
    sample_to_website = {}
    all_step_acc = []

    for sample in sample_data:
        annotation_id = sample["annotation_id"]
        action_uid = sample["action_uid"]
        sample_id = f"{annotation_id}_{action_uid}"

        sample_to_website[annotation_id] = sample["website"]

        # Get planner data
        planner_entry = next(
            (item for item in plan_data if item["annotation_id"] == annotation_id and item["action_uid"] == action_uid),
            None,
        )
        if planner_entry:
            gpt_action = planner_entry["gpt_action"].lower()
            gpt_value = planner_entry["gpt_value"].lower()
            if gpt_value == "none":
                pred_action = gpt_action
            else:
                pred_action = f"{gpt_action} {gpt_value}"
        else:
            pred_action = ""

        # Get ans data
        if ans_data[0].get("id", ""):
            ans_entry = next((item for item in ans_data if item["id"] == sample_id), None)
        else:
            ans_entry = planner_entry
        if ans_entry:
            output = ans_entry.get("output", "")
            ans_block = ans_entry.get("ans_block", "")
            if output and isinstance(output, str):
                output = output.strip("()").split(", ")
                output = (float(output[0]), float(output[1]))
            elif output and isinstance(output, (list, tuple)):
                output = (float(output[0]), float(output[1]))
            # For CogAgent
            elif not output and ans_block:
                blocks_path = planner_entry["blocks_path"]
                image_path = os.path.join(block_base_dir, blocks_path, f"{ans_entry['ans_block']}.png")
                output = extract_coordinates(ans_entry["pred_operation"], image_path)
            else:
                output = (0, 0)

            bboxes = ans_entry.get("bbox", [])
            scale = 1.0

            correct, coords = is_output_inside_bbox(bboxes, output, scale)
            all_element_acc.append([1 if correct else 0, annotation_id])

            # draw for debugging
            # if not correct:
            #     draw_debug(
            #         ans_entry["query"],
            #         ans_entry["image_path"],
            #         bboxes,
            #         coords,
            #         f"./debug/{sample_id}_{'correct' if correct else 'wrong'}.png",
            #     )

        else:
            all_element_acc.append([0, annotation_id])

        current_action = (sample["operation"], sample["value"])
        f1_score = calculate_f1(pred_action, current_action[0] + " " + current_action[1])
        all_operation_f1.append([f1_score, annotation_id])
        all_step_acc.append([1 if (all_operation_f1[-1][0] == 1 and all_element_acc[-1][0] == 1) else 0, annotation_id])

    total_steps = {sample["annotation_id"]: sample["total_steps"] for sample in sample_data}
    current_steps = collections.defaultdict(int)
    for _, annotation_id in all_element_acc:
        current_steps[annotation_id] += 1
    for annotation_id, steps in total_steps.items():
        while current_steps[annotation_id] < steps:
            all_element_acc.append([0, annotation_id])
            all_operation_f1.append([0, annotation_id])
            all_step_acc.append([0, annotation_id])
            current_steps[annotation_id] += 1

    macro_element_acc = collections.defaultdict(list)
    macro_operation_f1 = collections.defaultdict(list)
    macro_step_acc = collections.defaultdict(list)
    for x in all_element_acc:
        macro_element_acc[x[1]].append(x[0])
    for x in all_operation_f1:
        macro_operation_f1[x[1]].append(x[0])
    for x in all_step_acc:
        macro_step_acc[x[1]].append(x[0])

    error_ratio = collections.defaultdict(int)
    acc_per_website = collections.defaultdict(list)
    for annotation_id, x in macro_step_acc.items():
        acc_per_website[sample_to_website[annotation_id]].append(np.mean(x))
        error_count = len([y for y in x if y == 0])
        if error_count <= 3:
            error_ratio[error_count] += 1
        else:
            error_ratio[">3"] += 1

    acc_per_website = {k: (np.mean(v), len(v)) for k, v in acc_per_website.items()}
    error_ratio = {k: v / len(macro_element_acc) for k, v in error_ratio.items()}
    macro_element_acc = np.mean([np.mean(x) for x in macro_element_acc.values()])
    macro_operation_f1 = np.mean([np.mean(x) for x in macro_operation_f1.values()])
    macro_step_acc = np.mean([np.mean(x) for x in macro_step_acc.values()])

    return {
        "element_acc": np.mean([x[0] for x in all_element_acc]),
        "operation_f1": np.mean([x[0] for x in all_operation_f1]),
        "step_acc": np.mean([x[0] for x in all_step_acc]),
        "macro_element_acc": macro_element_acc,
        "macro_operation_f1": macro_operation_f1,
        "macro_step_acc": macro_step_acc,
        "error_ratio": error_ratio,
        "acc_per_website": acc_per_website,
    }


# python eval.py --sample_file /mnt/ceph_rbd/UGround/offline_evaluation/Multimodal-Mind2Web/data/samples/cross_website_blocks.jsonl --plan_file /mnt/ceph_rbd/UGround/offline_evaluation/Multimodal-Mind2Web/data/gpt-4o_results/cross_task_plan.jsonl --ans_file /mnt/ceph_rbd/UGround/grounding/outputs/uground --blocks /mnt/ceph_rbd/data/UGround-Offline-Evaluation/release_images/cross_website
# Load data


def main():
    from cursor.prepare_data import load_iterable_dataset

    # parser = argparse.ArgumentParser(description="Calculate metrics for Mind2Web data")
    # parser.add_argument("--sample_file", type=str, required=True, help="Path to sample (blocks) JSONL file")
    # parser.add_argument("--plan_file", type=str, required=True, help="Path to plan JSONL file")
    # parser.add_argument("--ans_file", type=str, required=True, help="Path to answer JSONL file")
    # parser.add_argument("--blocks", type=str, required=True, help="Base directory for block images")

    # args = parser.parse_args()

    # exp_name = "GUI-Cursor_3step"
    # exp_name = "GTA1-7B"
    exp_name = "GUI-Cursor-uitars-160steps"
    # exp_name = "GUI-G2-7B"
    dataset_name = "Multimodal-Mind2Web"
    data_sources = ["domain", "task", "website"]

    def load_jsonl(file_path):
        with open(file_path, "r") as f:
            return [json.loads(line) for line in f]

    all_ans_data = load_jsonl(f"./outputs/{dataset_name}/{exp_name}/predictions.jsonl")
    item_idx_to_image_size = dict()
    item_idx_to_bbox_list = dict()
    item_idx_to_image_path = dict()
    item_idx_to_query = dict()
    if os.path.exists(f"./outputs/{dataset_name}/item_idx_to_image_size.json"):
        item_idx_to_image_size = json.load(open(f"./outputs/{dataset_name}/item_idx_to_image_size.json"))
        item_idx_to_bbox_list = json.load(open(f"./outputs/{dataset_name}/item_idx_to_bbox_list.json"))
    else:
        from tqdm import tqdm

        for item in tqdm(load_iterable_dataset(dataset_name), desc="load size"):
            item_idx_to_image_size[item["item_idx"]] = item["image"].size
            item_idx_to_bbox_list[item["item_idx"]] = item["bbox_list"]
        json.dump(item_idx_to_image_size, open(f"./outputs/{dataset_name}/item_idx_to_image_size.json", "w"))
        json.dump(item_idx_to_bbox_list, open(f"./outputs/{dataset_name}/item_idx_to_bbox_list.json", "w"))

    for item in load_iterable_dataset(dataset_name, load_image=False):
        item_idx_to_image_path[item["item_idx"]] = item["img_filename"]
        item_idx_to_query[item["item_idx"]] = item["query"]

    for idx in range(len(all_ans_data)):
        cur_data_source, cur_id = all_ans_data[idx]["item_idx"].split("-yuzhaouoe-")
        all_ans_data[idx]["data_source"] = cur_data_source
        all_ans_data[idx]["id"] = cur_id

        image_size = item_idx_to_image_size[all_ans_data[idx]["item_idx"]]
        
        if "prediction" in all_ans_data[idx]:
            pred_x, pred_y = all_ans_data[idx]["prediction"]
        else:
            pred_proportion = all_ans_data[idx]["position_prop_history"][-1]
            pred_x, pred_y = float(pred_proportion[0] * image_size[0]), float(pred_proportion[1] * image_size[1])
    
        all_ans_data[idx]["output"] = (pred_x, pred_y)
        all_ans_data[idx]["bbox"] = item_idx_to_bbox_list[all_ans_data[idx]["item_idx"]]
        all_ans_data[idx]["image_path"] = item_idx_to_image_path[all_ans_data[idx]["item_idx"]]
        all_ans_data[idx]["query"] = item_idx_to_query[all_ans_data[idx]["item_idx"]]
    
    avg_element_acc = []
    for data_source in data_sources:
        sample_data = load_jsonl(
            f"/mnt/ceph_rbd/UGround/offline_evaluation/Multimodal-Mind2Web/data/samples/cross_{data_source}_blocks.jsonl"
        )
        plan_data = load_jsonl(
            f"/mnt/ceph_rbd/UGround/offline_evaluation/Multimodal-Mind2Web/data/gpt-4o_results/cross_{data_source}_plan.jsonl"
        )

        ans_data = [x for x in all_ans_data if x["data_source"] == data_source]

        # Load data
        # sample_data = load_jsonl(f"/mnt/ceph_rbd/UGround/offline_evaluation/Multimodal-Mind2Web/data/samples/cross_{data_source}_blocks.jsonl")
        # plan_data = load_jsonl(f"/mnt/ceph_rbd/UGround/offline_evaluation/Multimodal-Mind2Web/data/gpt-4o_results/cross_{data_source}_plan.jsonl")

        # Calculate metrics
        metrics = get_metrics_with_prediction(sample_data, plan_data, ans_data)

        # Print results
        print("=" * 20)
        print(f"{data_source} Metrics:")
        for key, value in metrics.items():
            if not isinstance(value, dict):
                print(f"{key}: {value*100:.2f}%")
            if key == "element_acc":
                avg_element_acc.append(value)
    print(np.mean(avg_element_acc) * 100)

if __name__ == "__main__":
    main()
