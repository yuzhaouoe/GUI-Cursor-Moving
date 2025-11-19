from pdb import run
import vllm

import re
import os
import time
from pathlib import Path
from io import BytesIO
from PIL import Image
from typing import List
import json

from cursor.utils.log import get_logger


logger = get_logger(__name__)


def get_logit_bias(processor, processor_name="Qwen2_5_VLProcessor"):
    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    bias_dict = {image_token_id: -100}

    if processor_name == "Qwen2_5_VLProcessor":
        bad_token = processor.tokenizer.encode(" addCriterion")[0]
        bias_dict[bad_token] = -100

    return bias_dict


class VLLMModel:
    def __init__(self, model_path, tp_size=1, batch_size=None):
        from vllm import LLM, SamplingParams
        from transformers import AutoProcessor
        import torch

        self.model = LLM(
            model=model_path,
            tensor_parallel_size=tp_size,
            max_model_len=32768,
            limit_mm_per_prompt={"image": 4, "video": 0, "audio": 0},
            max_num_seqs=batch_size,
            disable_log_stats=False,
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        logits_bias = get_logit_bias(self.processor)
        self.sampling_params = SamplingParams(**{"temperature": 0.0, "max_tokens": 512, "logit_bias": logits_bias})

    def batch_predict(self, vllm_inputs) -> dict:
        vllm_batch_responses = self.model.generate(vllm_inputs, sampling_params=self.sampling_params)
        batch_responses = [output.outputs[0].text for output in vllm_batch_responses]
        return batch_responses


def download_image():
    path = "/mnt/ceph_rbd/data/SpatialMQA/Dataset/dataset/test.jsonl"
    data = []
    for line in open(path, "r"):
        data.append(json.loads(line))

    # save images
    import requests
    from tqdm import tqdm

    IMAGE_DIR = "http://images.cocodataset.org/test2017"
    image_dir = "/mnt/ceph_rbd/data/SpatialMQA/download_image"
    for item in tqdm(data):
        image_id = item["image"]
        image_url = f"{IMAGE_DIR}/{image_id}"
        response = requests.get(image_url)
        image_content = BytesIO(response.content)
        image = Image.open(image_content)
        image.save(os.path.join(image_dir, image_id))


def load_spatial_dataset(dataset_name):
    if dataset_name == "SpatialMQA":
        path = "/mnt/ceph_rbd/data/SpatialMQA/Dataset/dataset/test.jsonl"
        data = []
        for line in open(path, "r"):
            data.append(json.loads(line))

        image_dir = "/mnt/ceph_rbd/data/SpatialMQA/download_image"

        def iter_data():
            for idx, item in enumerate(data):
                image = Image.open(os.path.join(image_dir, item["image"]))
                yield {
                    "item_idx": idx,
                    "question": item["question"],
                    "options": item["options"],
                    "answer": item["answer"],
                    "image": image,
                }

        return iter_data()

    elif dataset_name == "SPHERE":
        single_skill_json = [
            "single_skill/size_only",
            "single_skill/distance_only",
            "single_skill/position_only",
            "single_skill/counting_only-paired-distance_and_counting",
            "single_skill/counting_only-paired-position_and_counting",
        ]
        combine_2_skill_json = [
            "combine_2_skill/distance_and_size",
            "combine_2_skill/distance_and_counting",
            "combine_2_skill/position_and_counting",
        ]
        reasoning_json = [
            "reasoning/object_manipulation",
            "reasoning/object_occlusion",
            "reasoning/object_manipulation_w_intermediate",
            "reasoning/object_occlusion_w_intermediate",
        ]
        jdir = "/mnt/ceph_rbd/data/SPHERE-VLM/eval_datasets/coco_test2017_annotations"
        all_annotation_json = single_skill_json + combine_2_skill_json + reasoning_json

        def iter_data():
            img_dir = "/mnt/ceph_rbd/data/SPHERE-VLM/eval_datasets/coco_test2017"
            data = []
            for ann_json in all_annotation_json:
                json_path = f"{jdir}/{ann_json}.json"
                cur_data = json.load(open(json_path, "r"))
                for idx in range(len(cur_data)):
                    cur_data[idx]["source"] = ann_json
                data.extend(cur_data)
            for idx, ann in enumerate(data):
                source_img_id = ann["metadata"]["source_img_id"]
                full_source_img_id = (12 - len(source_img_id)) * "0" + source_img_id
                img_path = f"{img_dir}/{full_source_img_id}.jpg"
                image = Image.open(img_path).convert("RGB")
                yield {
                    "item_idx": idx,
                    "question": ann["question"],
                    "options": ann["option"],
                    "answer": ann["answer"],
                    "image": image,
                    "source": ann["source"],
                }

        return iter_data()

    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented.")


ZEROSHOT_SYSTEM_PROMPT = (
    f"You are currently a senior expert in spatial relation reasoning. "
    f"\nGiven an Image, a Question and Options, your task is to answer the correct spatial relation. "
    f"Note that you only need to choose one option from the all options without explaining any reason."
)

THINKING_SYSTEM_PROMPT = (
    f"You are currently a senior expert in spatial relation reasoning."
    f"\nGiven an Image, a Question and Options, your task is to answer the correct spatial relation."
    f"\nYour response must contain a thinking process before the answer. The thinking process is enclosed in <think> tag, and the answer is enclosed in <answer> tag."
    f"\nThe thinking process should identify the locations of the objects, and then reason the spatial relationship between them. The answer should only contain the option label."
)


def prepare_vllm_input(item, processor, thinking, system_prompt):

    question = item["question"]
    options = item["options"]
    answer = item["answer"]
    # A, B, C, D, ...
    option_labels = [chr(ord("A") + i) for i in range(len(options))]
    answer_option = option_labels[options.index(answer)]
    option_strs = [f"{label}. {option}" for label, option in zip(option_labels, options)]
    option_text = "\n".join(option_strs)
    user_prompt = f"Question: {question}\nOptions: {option_text}\nPlease select the correct option."
    messages = []
    if system_prompt is not None and system_prompt != "":
        messages.append({"role": "system", "content": system_prompt})

    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image", "image": item["image"]},
                {"type": "text", "text": user_prompt},
            ],
        }
    )
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    multi_modal_data = {"image": [item["image"]]}
    vllm_inputs = {
        "prompt": prompt,
        "multi_modal_data": multi_modal_data,
    }
    return {"vllm_inputs": vllm_inputs, "messages": messages, "answer_option": answer_option}


def prepare_batch(iter_dataset, batch_size, processor, thinking, system_prompt):
    cur_batch = []
    while len(cur_batch) < batch_size:
        item = next(iter_dataset, None)
        if item is None:
            break

        vllm_input = prepare_vllm_input(item, processor, thinking, system_prompt)
        item.update(vllm_input)
        cur_batch.append(item)

    return cur_batch


def save_item(save_path, input_item, response, pred, acc):
    with open(save_path, "a") as f:
        save_item = {
            "item_idx": input_item["item_idx"],
            "question": input_item["question"],
            "options": input_item["options"],
            "answer": input_item["answer"],
            "target_option": input_item["answer_option"],
            "response": response,
            "prediction": pred,
            "acc": acc,
        }
        if "source" in input_item:
            save_item["source"] = input_item["source"]
        json.dump(save_item, f)
        f.write("\n")


def extract_answer(response, thinking):
    if not thinking:
        # extract answer like "A", "B", etc.
        match = re.search(r"\b([A-Z])\b", response)
        if match:
            return match.group(1)
        else:
            return None
    else:
        # extract answer enclosed in <answer> tag
        match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        if match:
            answer_text = match.group(1).strip()
            # further extract option like "A", "B", etc. from answer_text
            option_match = re.search(r"\b([A-Z])\b", answer_text)
            if option_match:
                return option_match.group(1)
            else:
                return None
        else:
            return None


def run_eval(dataset_name, model_path, exp_name, thinking, system_prompt=None):
    if system_prompt is None:
        if thinking:
            system_prompt = THINKING_SYSTEM_PROMPT
        else:
            system_prompt = ZEROSHOT_SYSTEM_PROMPT

    batch_size = 128
    iter_dataset = load_spatial_dataset(dataset_name)

    model = VLLMModel(model_path=model_path, tp_size=1, batch_size=batch_size)

    output_dir = Path("outputs") / dataset_name / exp_name
    os.makedirs(output_dir, exist_ok=True)

    correctness: List[bool] = []
    completed_ids: List[int] = []
    is_valid: List[bool] = []

    eval_start_time = time.perf_counter()
    while True:
        cur_batch = prepare_batch(iter_dataset, batch_size, model.processor, thinking, system_prompt)

        if len(cur_batch) == 0:
            break

        vllm_inputs = [item["vllm_inputs"] for item in cur_batch]
        batch_outputs = model.batch_predict(vllm_inputs)

        cur_batch_size = len(cur_batch)
        if cur_batch_size == 0:
            logger.info("No more items to process.")
            break

        for sidx in range(len(cur_batch)):
            input_item = cur_batch[sidx]
            response = batch_outputs[sidx]
            pred = extract_answer(response, thinking)
            acc = pred == input_item["answer_option"]
            save_item(output_dir / "predictions.jsonl", input_item, response, pred, acc)

            correctness.append(acc)
            if pred is None:
                is_valid.append(False)
            else:
                is_valid.append(True)

        if len(correctness) > 0:
            logger.info(
                f"num: {len(correctness)}, avg acc: {sum(correctness) / len(correctness):.3%}, invalid num: {is_valid.count(False)}"
            )

    eval_end_time = time.perf_counter()
    logger.info(
        f"Evaluation completed in {eval_end_time - eval_start_time:.2f} seconds, "
        f"total completed items: {len(completed_ids)}, "
    )

    # save acc and time
    if len(correctness) > 0:
        acc = sum(correctness) / len(correctness)
    else:
        acc = 0.0

    with open(output_dir / "summary.txt", "w") as f:
        json.dump(
            {
                "total_completed_items": len(completed_ids),
                "accuracy": acc,
                "total_evaluation_time": eval_end_time - eval_start_time,
                "async_mode": False,
            },
            f,
        )

    postprocess_invalid(path=str(output_dir / "predictions.jsonl"))


def extract_answer_from_back(response):
    # find the last occurrence of answer like "A", "B", etc.
    matches = re.findall(r"\b([A-Z])\b", response)
    if matches:
        return matches[-1]
    else:
        return None


def postprocess_invalid(path=None):
    # path = "/mnt/ceph_rbd/GUI-Cursor-Moving/outputs/SpatialMQA/guicursor_thinking/predictions.jsonl"
    # path = "/mnt/ceph_rbd/GUI-Cursor-Moving/outputs/SpatialMQA/qwen_thinking/predictions.jsonl"
    if path is None:
        path = "/mnt/ceph_rbd/GUI-Cursor-Moving/outputs/SpatialMQA/guicursor_thinking_general2/predictions.jsonl"

    data = []
    for line in open(path, "r"):
        data.append(json.loads(line))

    postprocessed_acc = []
    remain_invalid = 0
    for item_idx, item in enumerate(data):
        if item["prediction"] is None:
            response = item["response"]
            # try to extract answer again
            pred = extract_answer_from_back(response)
            item["prediction"] = pred
            item["acc"] = pred == item["target_option"]
            data[item_idx]["prediction"] = pred
            data[item_idx]["acc"] = item["acc"]
            postprocessed_acc.append(item["acc"])
            if pred is None:
                remain_invalid += 1
    print(f"Postprocessed {len(postprocessed_acc)} invalid predictions.")
    print(f"Remaining invalid predictions: {remain_invalid}")
    if len(postprocessed_acc) > 0:
        print(f"Postprocessed Mean Acc: {sum(postprocessed_acc) / len(postprocessed_acc):.3%}")

    print("Overall Mean Acc after Postprocessing:")
    all_acc = [item["acc"] for item in data]
    overall_mean_acc = sum(all_acc) / len(all_acc)
    print(f"Overall Mean Acc: {overall_mean_acc:.1%}, Total Samples: {len(all_acc)}")

    # save back

    with open(path + "post.jsonl", "w") as f:
        for item in data:
            json.dump(item, f)
            f.write("\n")

    summarised_path = path + "post_summary.txt"
    with open(summarised_path, "w") as f:
        json.dump(
            {
                "overall_accuracy_after_postprocessing": overall_mean_acc,
                "total_samples": len(all_acc),
                "invalid_predictions_after_postprocessing": remain_invalid,
            },
            f,
        )


def debug():
    from transformers import AutoProcessor

    iter_dataset = load_spatial_dataset()
    processor = AutoProcessor.from_pretrained("/mnt/ceph_rbd/models/Qwen/Qwen2.5-VL-7B-Instruct_resaved")
    # prepare_batch(iter_dataset, batch_size=2, processor=processor)

    # print(extract_answer("The answer is A. B C"))
    all_image_size = []
    for item in iter_dataset:

        image = item["image"]
        image_size = image.size
        all_image_size.append(image_size)

    from collections import Counter

    print(Counter(all_image_size))


def main():
    # dataset_name = "SpatialMQA"
    # # dataset_name = "SPHERE"
    # # model_path = "/mnt/ceph_rbd/models/Qwen/Qwen2.5-VL-7B-Instruct_resaved"
    # # model_path = "/mnt/ceph_rbd/models/GUI-Cursor"
    # # exp_name = "guicursor"
    # model_path = "Salesforce/GTA1-7B-2507"
    # exp_name = "gta1"
    # # model_path = "inclusionAI/GUI-G2-7B"
    # # exp_name = "guig2"
    # thinking = False
    # # system_prompt = THINKING_SYSTEM_PROMPT
    # system_prompt = None

    run_eval(
        dataset_name="SpatialMQA",
        model_path="Salesforce/GTA1-7B-2507",
        exp_name="gta1",
        thinking=True,
        # system_prompt=ZEROSHOT_SYSTEM_PROMPT,
    )

    # run_eval(
    #     dataset_name="SpatialMQA",
    #     model_path="inclusionAI/GUI-G2-7B",
    #     exp_name="guig2",
    #     thinking=True,
    #     # system_prompt=ZEROSHOT_SYSTEM_PROMPT,
    # )

if __name__ == "__main__":
    main()
    # debug()
    # postprocess_invalid()
