import base64
import io
from typing import Union, List
from PIL import Image
from tenacity import retry, wait_random_exponential, stop_after_attempt
from openai import AzureOpenAI, AsyncAzureOpenAI
# from azure.identity import (
#     DefaultAzureCredential,
#     ChainedTokenCredential,
#     ManagedIdentityCredential,
#     AzureCliCredential,
#     get_bearer_token_provider,
# )
import json
import logging
import os
from pathlib import Path
from huggingface_hub import snapshot_download
import math
import shutil

logging.basicConfig(
    format="%(asctime)s - %(levelname)s %(name)s %(lineno)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def load_description(desc_path="./data/re_gpt-4o.jsonl"):
    descriptions = []
    with open(desc_path, "r") as f:
        for line in f:
            line = json.loads(line)
            descriptions.append(line["description"])
    return descriptions


# def create_client():
#     credential = ChainedTokenCredential(
#         AzureCliCredential(),
#         ManagedIdentityCredential(client_id="88ddd80a-a87c-4d11-8ada-05792ca52780"),
#     )

#     token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")
#     azure_endpoint = "https://aidaihub2294673195.cognitiveservices.azure.com"
#     client = AzureOpenAI(
#         azure_endpoint=azure_endpoint,
#         azure_ad_token_provider=token_provider,
#         api_version="2025-03-01-preview",
#     )
#     return client


def encode_image(image: Union[str, Image.Image], format) -> str:
    if isinstance(image, str):
        with open(image, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    elif isinstance(image, Image.Image):
        image = image.convert("RGB")
        buffer = io.BytesIO()
        if format == "JPEG":
            image.save(buffer, format="JPEG")
        elif format == "PNG":
            image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


def get_url_payload(url: str) -> dict:
    return {"type": "input_image", "image_url": url}


def get_base64_payload(base64_image: str, format) -> dict:
    if format == "JPEG":
        return {
            "type": "input_image",
            "image_url": f"data:image/jpeg;base64,{base64_image}",
        }
    elif format == "PNG":
        return {
            "type": "input_image",
            "image_url": f"data:image/png;base64,{base64_image}",
        }


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(20))
def get_reply(client, message_history, max_tokens, temperature=0):
    response = client.responses.create(
        model="gpt-4o",
        previous_response_id=None,
        input=message_history,
        max_output_tokens=max_tokens,
        temperature=temperature,
        top_p=1.0,
    )
    return response


def process_images(
    client: AzureOpenAI,
    system_prompt: str,
    query: str,
    image: Union[str, Image.Image, List[Union[str, Image.Image]]],
    max_tokens=2048,
    temperature=0,
    format="PNG",
    message_history=None,
) -> str:
    content = []
    if isinstance(image, str) and image.startswith("http"):
        content.append(get_url_payload(image))
    else:
        base64_image = encode_image(image, format=format)
        content.append(get_base64_payload(base64_image, format=format))

    if query is not None and query != "":
        content.append({"type": "input_text", "text": query})

    assert len(content) > 0, "No content provided for the model."

    if message_history is None:
        if system_prompt is None:
            system_prompt = "You are a helpful assistant."
        message_history = [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {"role": "user", "content": content},
        ]
    else:
        message_history.append({"role": "user", "content": content})

    try:
        response = get_reply(
            client=client,
            message_history=message_history,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception as e:
        # This is a general catch-all for other potential exceptions
        logger.error(f"An unexpected error occurred: {e}")
        # if "Too many images in request." in str(e):
        #     json.dump(message_history, open("error_log.json", "w"))
        #     exit()
        raise e

    # message_history.append(
    #     {"role": "assistant", "content": response.output[0].content[0].text}
    # )
    return {
        "output_text": response.output[0].content[0].text,
        "message_history": message_history,
    }


# def load_vllm_model(model_path, dtype, max_num_images, max_num_seqs, max_model_len):
#     from vllm import LLM
#     import torch

#     engine_args = {
#         "model": model_path,
#         "max_model_len": max_model_len,
#         "max_num_seqs": max_num_seqs,
#         "limit_mm_per_prompt": {"image": max_num_images, "video": 0, "audio": 0},
#         "dtype": dtype,
#         "gpu_memory_utilization": 0.90,
#         "tensor_parallel_size": torch.cuda.device_count(),
#         "mm_processor_kwargs": {"use_fast": True},
#     }
#     logger.info(f"Loading VLLM model with args: {engine_args}")
#     llm = LLM(**engine_args)
#     logger.info("VLLM model loaded successfully.")
#     return llm


def load_vllm_model(model_path, dtype, max_num_images, max_num_seqs, max_model_len):
    from vllm import LLM
    from transformers import AutoProcessor
    import torch

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True,
    )
    engine_kwargs = {}
    engine_kwargs["disable_mm_preprocessor_cache"] = True
    engine_kwargs["limit_mm_per_prompt"] = {"image": max_num_images, "video": 0, "audio": 0}

    inference_engine = LLM(
        model=model_path,
        skip_tokenizer_init=False,
        trust_remote_code=True,
        # load_format="dummy",
        dtype=dtype,
        seed=1,
        max_model_len=max_model_len,
        # distributed_executor_backend="external_launcher",
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.9,
        max_num_batched_tokens=32768,
        disable_log_stats=False,
        enforce_eager=False,
        disable_custom_all_reduce=True,
        enable_chunked_prefill=False,
        # enable_sleep_mode=True,
        enable_prefix_caching=True,
        **engine_kwargs,
    )
    return inference_engine


def get_iterable_data(
    dataset_name="ScreenSpot", skip_ids=None, first_n=None, selected_ids=None, from_cropped_exp_name=None
):
    from datasets import load_dataset

    if from_cropped_exp_name is not None:
        dataset_name = f"cropped##{dataset_name}##{from_cropped_exp_name}"

    if "@" in dataset_name:
        dataset_name, split = dataset_name.split("@")
        if ":" in split:
            start_idx, end_idx = split.split(":")
            start_idx = start_idx.split("-")[1]
            start_idx, end_idx = int(start_idx), int(end_idx)
            print(f"{dataset_name=}, {split=}, {start_idx=}, {end_idx=}")
    else:
        split = "all"
        start_idx, end_idx = None, None
    logger.info(f"Loading dataset: {dataset_name}")
    if dataset_name == "ScreenSpot":
        data_path = "rootsautomation/ScreenSpot"
        dataset = load_dataset(data_path, split="test")

        # dataset = load_from_disk("./data10")
        descriptions = load_description()
        for item_idx, item in enumerate(dataset):
            if skip_ids is not None and item_idx in skip_ids:
                logger.info(f"Item {item_idx} already processed, skipping.")
                continue
            query = descriptions[item_idx]
            item["item_idx"] = item_idx
            yield {"item": item, "query": query}

    elif dataset_name == "ScreenSpot-v2":
        dataset = load_dataset("yuzhaouoe/ScreenSpot-v2", split="test")

        for item_idx, item in enumerate(dataset):
            if skip_ids is not None and item_idx in skip_ids:
                logger.info(f"Item {item_idx} already processed, skipping.")
                continue
            query = item["instruction"]
            item["item_idx"] = item_idx
            if first_n is not None and item_idx >= first_n:
                break
            if split == "all":
                yield {"item": item, "query": query}
            else:
                if start_idx <= item_idx < end_idx:
                    yield {"item": item, "query": query}
                if item_idx >= end_idx:
                    break

    elif dataset_name == "UGround-V1-Data-Box":
        dataset = load_dataset(
            "osunlp/UGround-V1-Data-Box",
            split="train",
            streaming=True,
            token=os.getenv("HF_TOKEN"),
        )
        cur_idx = 0
        yield_nums = 0
        if selected_ids is not None:
            print(f"selected_ids: {selected_ids[:5]} ... {selected_ids[-5:]}")
        for item in dataset:
            conversation = item["conversations"]
            conversation = json.loads(conversation)
            assert len(conversation) % 2 == 0
            for idx in range(0, len(conversation), 2):
                image = Image.open(io.BytesIO(item["image"]))
                assert conversation[idx]["from"] == "human"
                assert conversation[idx + 1]["from"] == "gpt"
                bbox = eval(conversation[idx + 1]["value"])
                bbox = tuple(t / 1000 for t in bbox)
                query = conversation[idx]["value"]
                new_item = {"image": image, "bbox": bbox, "item_idx": cur_idx}
                if selected_ids is None:
                    yield {"item": new_item, "query": query}
                elif cur_idx in selected_ids:
                    yield {"item": new_item, "query": query}
                    yield_nums += 1
                    if yield_nums >= len(selected_ids):
                        logger.info("Data iteration complete.")
                        # raise StopIteration
                        return
                cur_idx += 1
                if first_n is not None and cur_idx >= first_n:
                    break
    elif "grounding_train" in dataset_name or "azureml" in dataset_name:
        print(f"load from {dataset_name}")
        data = json.load(open(dataset_name, "r"))
        for item_idx, item in enumerate(data):
            yield {
                "item": {
                    "item_idx": item_idx,
                    "image": os.path.join("image", item["image"]),
                    "bbox": [xy / 1000 for xy in item["bbox"]],
                },
                "query": item["conversations"][0]["value"].removeprefix("<image>"),
            }
    elif dataset_name == "ScreenSpot-Pro":
        data_dir = Path(os.path.expanduser("~")) / "ScreenSpot-Pro"
        snapshot_download(repo_id="likaixin/ScreenSpot-Pro", repo_type="dataset", local_dir=data_dir)
        files = [
            "vscode_macos.json",
            "pycharm_macos.json",
            "premiere_windows.json",
            "illustrator_windows.json",
            "matlab_macos.json",
            "solidworks_windows.json",
            "powerpoint_windows.json",
            "inventor_windows.json",
            "eviews_windows.json",
            "vmware_macos.json",
            "unreal_engine_windows.json",
            "windows_common_windows.json",
            "autocad_windows.json",
            "android_studio_macos.json",
            "blender_windows.json",
            "origin_windows.json",
            "macos_common_macos.json",
            "photoshop_windows.json",
            "excel_macos.json",
            "fruitloops_windows.json",
            "stata_windows.json",
            "word_macos.json",
            "linux_common_linux.json",
            "davinci_macos.json",
            "vivado_windows.json",
            "quartus_windows.json",
        ]
        examples = []
        for file in files:
            dataset_name = data_dir / "annotations" / file
            examples.extend(json.load(open(dataset_name, "r")))

        image_dir = data_dir / "images"

        for item_idx, item in enumerate(examples):
            # [bbox[0], bbox[1], bbox[2], bbox[3]]  # x1, y1, x2, y2
            # xmin, xmax, ymin, ymax = item["bbox"]
            xmin, ymin, xmax, ymax = item["bbox"]
            width, height = item["img_size"]
            # bbox_proportions = [xmin / width, xmax / width, ymin / height, ymax / height]
            bbox_proportions = [xmin / width, ymin / height, xmax / width, ymax / height]
            yield {
                "item": {
                    "item_idx": item_idx,
                    "image": Image.open(image_dir / item["img_filename"]),
                    "bbox": bbox_proportions,
                },
                "query": item["instruction"],
            }
    elif dataset_name.startswith("cropped##"):
        # cropped##ScreenSpot-Pro##expname
        print("Load Cropped Image Based on the First Step Prediction")
        print(f"Load Cropped Image Based on the First Step Prediction {from_cropped_exp_name}")
        from cursor.utils.image import (
            smart_resize_min_max_tokens_per_img,
            bbox_coords_from_proportions,
            crop_image_at_coordinate,
            crop_image_with_padding,
        )

        _, org_dataset_name, exp_name = dataset_name.split("##")
        predictions = [
            json.loads(line)
            for line in open(
                f"/mnt/ceph_rbd/cursor-moving/outputs/{org_dataset_name}/{exp_name}/predictions.jsonl", "r"
            ).readlines()
        ]
        all_cursor_positions = [item["position_history"][-1] for item in predictions]
        for example, cursor_coords in zip(get_iterable_data(org_dataset_name), all_cursor_positions):
            item = example["item"]
            item_idx = item["item_idx"]
            query = example["query"]
            cursor_x, cursor_y = cursor_coords
            org_image = item["image"]
            screenshot = smart_resize_min_max_tokens_per_img(
                image=item["image"],
                min_tokens_per_img=2048,
                max_tokens_per_img=10240,
            )
            width, height = screenshot.size
            cursor_x_ratio = cursor_x / width
            cursor_y_ratio = cursor_y / height

            org_width, org_height = org_image.size
            org_bbox_coords = bbox_coords_from_proportions(org_width, org_height, item["bbox"])

            cursor_org_x = int(org_width * cursor_x_ratio)
            cursor_org_y = int(org_height * cursor_y_ratio)

            # crop_width = int(0.5 * org_width)
            # crop_height = int(0.5 * org_height)
            org_image_pixels = org_width * org_height

            cropped_image_pixels = 1920 * 1080  # 2560 * 1440
            crop_width = int(math.sqrt((cropped_image_pixels * org_width) / org_height))
            crop_height = int(math.sqrt((cropped_image_pixels * org_height) / org_width))
            # if -5 < (crop_width - 2560) < 5 and -5 < (crop_height - 1440) < 5:
            #     crop_width, crop_height = 2560, 1440

            if crop_width >= org_width or crop_height >= org_height:
                crop_width = org_width
                crop_height = org_height
                cropped_image = org_image
                cropped_bbox_proportions = item["bbox"]
            else:
                cropped_image, left, top = crop_image_at_coordinate(
                    org_image, cursor_org_x, cursor_org_y, crop_width, crop_height
                )
                # cropped_image, left, top = crop_image_with_padding(
                #     org_image, cursor_org_x, cursor_org_y, crop_width, crop_height
                # )
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

            yield {
                "item": {
                    "item_idx": item_idx,
                    "image": cropped_image,
                    "bbox": cropped_bbox_proportions,
                },
                "query": query,
            }

    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")

    logger.info("Data iteration complete.")


def reorder_predictions(path):
    with_global_steps = False
    with open(path, "r") as f:
        first_line = json.loads(f.readline())
        if "global_steps" in first_line:
            with_global_steps = True
    if with_global_steps:
        idx2item = dict()
        with open(path, "r") as f:
            for line in f:
                item = json.loads(line)
                if item["item_idx"] not in idx2item:
                    idx2item[item["item_idx"]] = dict()
                idx2item[item["item_idx"]][item["global_steps"]] = item
        sorted_preds = sorted(idx2item.items(), key=lambda x: x[0])
        sorted_preds = [item[1] for item in sorted_preds]
        for idx in range(len(sorted_preds)):
            global2item = sorted_preds[idx]
            sorted_global2item = sorted(global2item.items(), key=lambda x: x[0])
            sorted_preds[idx] = [item[1] for item in sorted_global2item]
        all_sorted_items = []
        for preds in sorted_preds:
            for item in preds:
                assert item["item_idx"] == preds[0]["item_idx"]
                assert item["global_steps"] >= preds[0]["global_steps"]
                all_sorted_items.append(item)
        with open(path, "w") as f:
            for item in all_sorted_items:
                f.write(json.dumps(item) + "\n")
    else:
        idx2item = dict()
        with open(path, "r") as f:
            for line in f:
                item = json.loads(line)
                idx2item[item["item_idx"]] = item
        sorted_preds = sorted(idx2item.items(), key=lambda x: x[0])
        sorted_preds = [item[1] for item in sorted_preds]
        with open(path, "w") as f:
            for item_idx, item in enumerate(sorted_preds):
                assert item_idx == item["item_idx"]
                f.write(json.dumps(item) + "\n")


def load_existing_pred_ids(prediction_save_path):
    with open(prediction_save_path, "r") as f:
        already_preds = [json.loads(line) for line in f]
    if "global_steps" not in already_preds[0]:

        # filter out error items
        filtered_preds = []
        for item in already_preds:
            if item["stop_reason"] in [
                "get_reply_failed",
                "extract_action_failed",
                "error",
            ]:
                continue
            filtered_preds.append(item)
        existed_ids = set([item["item_idx"] for item in filtered_preds])
        logger.info(f"Resuming from {prediction_save_path}, {len(existed_ids)} items")
        return existed_ids

    else:
        existed_ids = [item["item_idx"] for item in already_preds]
        filter_ids = []
        for item in already_preds:
            if item["stop_reason"] in [
                "get_reply_failed",
                "extract_action_failed",
                "error",
                "max_steps_reached_but_continue",
            ]:
                filter_ids.append(item["item_idx"])
        existed_ids = list(set(existed_ids) - set(filter_ids))
        logger.info(f"Resuming from {prediction_save_path}, {len(existed_ids)} items")
        return existed_ids

