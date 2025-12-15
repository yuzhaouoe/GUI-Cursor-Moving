import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from vllm import LLM
from vllm import SamplingParams

from qwen_vl_utils import smart_resize
import re
import json
from pathlib import Path
import math
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

from cursor.prepare_data import load_iterable_dataset


def prepare_input(sample, processor):
    image = sample["image"]
    instruction = sample["query"]
    bbox_proportions = sample["bbox_proportions"]
    bbox_coords = sample["bbox_coords"]
    # width, height = image.width, image.height
    width, height = image.width, image.height

    if bbox_proportions is not None:
        bbox_coords = [
            int(bbox_proportions[0] * width),
            int(bbox_proportions[1] * height),
            int(bbox_proportions[2] * width),
            int(bbox_proportions[3] * height),
        ]
        bbox_coords_list = None
    else:
        bbox_coords_list = sample["bbox_list"] 
        bbox_coords = None
    # resized_height, resized_width = smart_resize(
    #     image.height,
    #     image.width,
    #     factor=processor.image_processor.patch_size * processor.image_processor.merge_size,
    #     min_pixels=processor.image_processor.min_pixels,
    #     max_pixels=processor.image_processor.max_pixels,
    # )
    # resized_image = image.resize((resized_width, resized_height))
    # scale_x, scale_y = width / resized_width, height / resized_height

    # bbox_coords = bbox_coords_from_proportions(resized_width, resized_height, bbox_proportions)

    prompt_origin = (
        "Outline the position corresponding to the instruction: {}. The output should be only [x1,y1,x2,y2]."
    )
    full_prompt = prompt_origin.format(instruction)

    message_inputs = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": full_prompt},
            ],
        }
    ]

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
        "bbox_coords_list": bbox_coords_list,
    }

def is_output_inside_bbox_list(bboxes, output):
    output_x, output_y = output
    for bbox in bboxes:
        bbox_x, bbox_y, bbox_width, bbox_height = bbox
        if bbox_x <= output_x <= bbox_x + bbox_width and bbox_y <= output_y <= bbox_y + bbox_height:
            return True
    return False
# def load_model():
#     model_path = download_to_temp_dir("/mnt/yuzhao/tmp_model_dir", "hf/inclusionAI/GUI-G2-7B")

#     # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     #     model_path,
#     #     device_map="auto",
#     #     trust_remote_code=True,
#     #     torch_dtype=torch.bfloat16,
#     #     attn_implementation="flash_attention_2",
#     # ).eval().cuda()

#     tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
#     processor = AutoProcessor.from_pretrained(model_path)
#     generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True).to_dict()
#     # generation_config.update({"max_length": 2048, "do_sample": False, "temperature": 0.0})
#     # model.generation_config = GenerationConfig(**generation_config)

#     return model, tokenizer, processor


def infer(model, processor, instruction, image):

    prompt_origin = (
        "Outline the position corresponding to the instruction: {}. The output should be only [x1,y1,x2,y2]."
    )
    full_prompt = prompt_origin.format(instruction)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": full_prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    print(output_text)
    input_height = inputs["image_grid_thw"][0][1] * 14
    input_width = inputs["image_grid_thw"][0][2] * 14

    try:
        box = eval(output_text[0])
        abs_y1 = float(box[1] / input_height)
        abs_x1 = float(box[0] / input_width)
        abs_y2 = float(box[3] / input_height)
        abs_x2 = float(box[2] / input_width)
        box = [abs_x1, abs_y1, abs_x2, abs_y2]
    except:
        box = [0, 0, 0, 0]

    point = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]

    result_dict = {"result": "positive", "format": "x1y1x2y2", "raw_response": output_text, "bbox": box, "point": point}

    return result_dict


def extract_coordinates(output_text):
    box = eval(output_text)
    point = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
    return point


def is_within_bbox(x, y, bbox_coords):
    top_left_x, top_left_y, bottom_right_x, bottom_right_y = bbox_coords
    return top_left_x <= x <= bottom_right_x and top_left_y <= y <= bottom_right_y


def main():
    # model, tokenizer, processor = load_model()
    # model_path = download_to_temp_dir("/mnt/yuzhao/tmp_model_dir", "hf/inclusionAI/GUI-G2-7B")
    # dataset_name = "UI-Vision"
    # dataset_name = "OSWorld-G_refined"
    # dataset_name = "UI-Vision"
    dataset_name = "Multimodal-Mind2Web"

    model_path = "/mnt/ceph_rbd/models/GUI-G2-7B"
    exp_name = "GUI-G2-7B"
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
        "max_tokens": 128,
    }
    sampling_params = SamplingParams(**sampling_kwargs)
    processor = AutoProcessor.from_pretrained(model_path)

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
        if batch_target[0] is None:
            batch_target_list  = [item["bbox_coords_list"] for item in batch]
            cur_correct = []
            for c, bcs in zip(cur_preds, batch_target_list):
                is_correct = is_output_inside_bbox_list(bcs, c)
                cur_correct.append(is_correct)
        else:
            cur_correct = [is_within_bbox(c[0], c[1], bcs) for c, bcs in zip(cur_preds, batch_target)]
        correctness.extend(cur_correct)
        predictions.extend(cur_preds)

        print(f"total: {len(correctness)}, avg acc: {sum(correctness) / len(correctness):.3%}")

        with open(output_dir / save_file, "a") as f:

            for item_idx, p, c, s in zip(batch_item_idx, cur_preds, cur_correct, batch_data_source):
                f.write(
                    json.dumps({"item_idx": item_idx, "prediction": p, "within_bbox_history": [c], "data_source": s})
                    + "\n"
                )

        batch = []


if __name__ == "__main__":
    main()
