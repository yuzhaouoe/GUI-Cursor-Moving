from cursor import DATA_DIR
from huggingface_hub import snapshot_download
import zipfile
import shutil
import os
import json
from PIL import Image
import pandas as pd
import io
import base64
from datasets import Dataset, load_dataset
from tqdm import tqdm
from pathlib import Path
from PIL import ImageDraw


def load_iterable_dataset(dataset_name: str, load_image=True, max_examples=None, data_source=None):

    if dataset_name == "ScreenSpot-v2":
        dataset = load_screenspot_v2(load_image=load_image)
    elif dataset_name == "OSWorld-G":
        dataset = load_osworld_g(refined=False, load_image=load_image)
    elif dataset_name == "OSWorld-G_refined":
        dataset = load_osworld_g(refined=True, load_image=load_image)
    elif dataset_name == "UI-Vision":
        dataset = load_ui_vision(load_image=load_image)
    elif dataset_name.startswith("grounding_train"):
        dataset = load_train_data(dataset_name)
    elif dataset_name == "ScreenSpot-Pro":
        dataset = load_screenspot_pro(load_image=load_image)
    elif dataset_name == "Multimodal-Mind2Web":
        dataset = load_multimodal_mind2web_uground_plan(load_image=load_image)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    if max_examples is not None:

        def wrap_iterable_with_limit():
            count = 0
            for item in dataset:
                if count >= max_examples:
                    break
                yield item
                count += 1

        return wrap_iterable_with_limit()

    return dataset


def load_train_data(dataset_name, load_image=False):
    data_path = DATA_DIR / f"{dataset_name}.json"
    # image_dir = DATA_DIR / "grounding_data" / "image.zip"
    data = json.load(open(data_path, "r"))
    for item_idx, item in enumerate(data):
        query = item["conversations"][0]["value"].removeprefix("<image>")
        bbox_proportions = [xy / 1000 for xy in item["bbox"]]
        # image_path = image_dir / item["image"]
        yield {
            "item_idx": item_idx,
            "img_filename": item["image"],
            "bbox_proportions": bbox_proportions,
            "query": query,
        }


def load_screenspot_pro(load_image=True):
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
    data_dir = DATA_DIR / "ScreenSpot-Pro"
    examples = []
    for file in files:
        dataset_name = data_dir / "annotations" / file
        examples.extend(json.load(open(dataset_name, "r")))

    image_dir = data_dir / "images"

    def iter_data():
        for item_idx, item in enumerate(examples):
            xmin, ymin, xmax, ymax = item["bbox"]
            if load_image:
                image = Image.open(image_dir / item["img_filename"]).convert("RGB")
                width, height = item["img_size"]
                # bbox_proportions = [xmin / width, xmax / width, ymin / height, ymax / height]
                bbox_proportions = [xmin / width, ymin / height, xmax / width, ymax / height]
                bbox_coords = [xmin, ymin, xmax, ymax]
            else:
                image = None
                bbox_proportions = None
                bbox_coords = None
            yield {
                "item_idx": item_idx,
                "bbox_coords": bbox_coords,
                "bbox_proportions": bbox_proportions,
                "query": item["instruction"],
                # "data_type": item["data_type"],
                # "data_source": item["data_source"],
                "img_filename": item["img_filename"],
                "image": image,
            }

    return iter_data()


def load_multimodal_mind2web_uground_plan(load_image=True, select_data_source=None):
    data_sources = ["website", "task", "domain"]
    question_files = [
        f"/mnt/ceph_rbd/UGround/offline_evaluation/Multimodal-Mind2Web/data/gpt-4o_results/cross_{data_source}_query.jsonl"
        for data_source in data_sources
    ]
    image_folders = [
        f"/mnt/ceph_rbd/data/UGround-Offline-Evaluation/release_images/cross_{data_source}"
        for data_source in data_sources
    ]
    image_key = "image"

    item_idx = 0
    for idx in range(len(data_sources)):
        data_source = data_sources[idx]
        question_file = question_files[idx]
        image_folder = image_folders[idx]
        questions = [json.loads(line) for line in open(question_file, "r")]
        for item in questions:
            if select_data_source is not None and data_source != select_data_source:
                item_idx += 1
                continue
            else:
                image_base_dir = os.path.expanduser(image_folder)
                image_path = os.path.join(image_base_dir, item[image_key])
                if load_image:
                    image = Image.open(image_path).convert("RGB")
                else:
                    image = None
                query = item["description"]

                yield {
                    "item_idx": f"{data_source}-yuzhaouoe-{item['id']}",
                    "image": image,
                    "query": query,
                    "bbox_coords": None,
                    "bbox_proportions": None,
                    "data_source": data_source,
                    "img_filename": image_path,
                    "bbox_list": item["bbox"],
                }
                item_idx += 1


def load_ui_vision(load_image=True):
    # download ServiceNow/ui-vision
    hf_path = "ServiceNow/ui-vision"
    # if path not exists, download
    if not (DATA_DIR / "UI-Vision").exists():
        snapshot_download(repo_id=hf_path, repo_type="dataset", local_dir=DATA_DIR / "UI-Vision")

    file_dir = DATA_DIR / "UI-Vision" / "annotations/element_grounding"
    files = ["element_grounding_basic.json", "element_grounding_functional.json", "element_grounding_spatial.json"]

    all_data = []
    for file in files:
        data = json.load(open(file_dir / file, "r"))
        data_type = file.replace("element_grounding_", "").replace(".json", "")
        for item in data:
            item["data_source"] = data_type
        all_data.extend(data)

    def iter_data():
        image_dir = image_path = DATA_DIR / "UI-Vision" / "images"
        for item_idx, item in enumerate(all_data):
            bbox = item["bbox"]
            bbox_coords = bbox
            if load_image:
                image_path = image_dir / item["image_path"]
                image = Image.open(image_path).convert("RGB")
                bbox_proportions = [
                    bbox_coords[0] / image.width,
                    bbox_coords[1] / image.height,
                    bbox_coords[2] / image.width,
                    bbox_coords[3] / image.height,
                ]
            else:
                image = None
                bbox_proportions = None

            yield {
                "item_idx": item_idx,
                "bbox_coords": bbox_coords,
                "bbox_proportions": bbox_proportions,
                "query": item["prompt_to_evaluate"],
                "data_type": item["element_type"],
                "data_source": item["data_source"],
                "img_filename": item["image_path"],
                "image": image,
            }

    return iter_data()


def load_osworld_g(refined, load_image=True):
    if refined:
        path = "/mnt/ceph_rbd/data/OSWorld-G/OSWorld-G/benchmark/OSWorld-G_refined.json"
    else:
        path = "/mnt/ceph_rbd/data/OSWorld-G/OSWorld-G/benchmark/OSWorld-G.json"
    dataset = json.load(open(path, "r"))
    image_dir = Path("/mnt/ceph_rbd/data/OSWorld-G/OSWorld-G/benchmark/images")

    #     "id": "0FOB4CLBT2-0",
    # "image_path": "0FOB4CLBT2.png",
    # "image_size": [
    #   1920,
    #   1080
    # ],
    # "instruction": "Click the button that including an icon of funnel on the right of the \"search settings\" bar",
    # "box_type": "bbox",
    # "box_coordinates": [
    #   1422.9,
    #   326.4,
    #   26.679999999999836,
    #   28.400000000000034
    # ],
    # "GUI_types": [
    #   "Label",
    #   "Button",
    #   "Icon"
    # ]

    def iter_data():
        for item_idx, item in enumerate(dataset):
            image_path = image_dir / item["image_path"]
            bbox = item["box_coordinates"]
            bbox_coords = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            if load_image:
                image = Image.open(image_path).convert("RGB")
                bbox_proportions = [
                    bbox_coords[0] / image.width,
                    bbox_coords[1] / image.height,
                    bbox_coords[2] / image.width,
                    bbox_coords[3] / image.height,
                ]
            else:
                image = None
                bbox_proportions = None
            yield {
                "item_idx": item["id"],  # item_idx,
                "bbox_coords": bbox_coords,
                "bbox_proportions": bbox_proportions,
                "query": item["instruction"],
                "data_type": item["GUI_types"],
                "data_source": None,
                "img_filename": item["image_path"],
                "image": image,
                # "id": item["id"],
            }

    return iter_data()


def load_screenspot_v2(load_image=False):
    if not (DATA_DIR / "ScreenSpot-v2").exists():
        hf_path = "OS-Copilot/ScreenSpot-v2"
        snapshot_download(repo_id=hf_path, repo_type="dataset", local_dir=DATA_DIR / "ScreenSpot-v2")
        with zipfile.ZipFile(DATA_DIR / "ScreenSpot-v2" / "screenspotv2_image.zip", "r") as zip_ref:
            zip_ref.extractall(DATA_DIR / "ScreenSpot-v2")
        os.remove(DATA_DIR / "ScreenSpot-v2" / "screenspotv2_image.zip")

    data = []
    for files in ["screenspot_desktop_v2.json", "screenspot_mobile_v2.json", "screenspot_web_v2.json"]:
        data.extend(json.load(open(DATA_DIR / "ScreenSpot-v2" / files, "r")))

    image_dir = DATA_DIR / "ScreenSpot-v2" / "screenspotv2_image"

    def iter_data():
        for item_idx, item in enumerate(data):
            image_path: Path = image_dir / item["img_filename"]
            bbox = item["bbox"]
            bbox_coords = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            if load_image:
                image = Image.open(image_path).convert("RGB")
                bbox_proportions = [
                    bbox_coords[0] / image.width,
                    bbox_coords[1] / image.height,
                    bbox_coords[2] / image.width,
                    bbox_coords[3] / image.height,
                ]
            else:
                image = None
                bbox_proportions = None
            yield {
                "item_idx": item_idx,
                "bbox_coords": bbox_coords,
                "bbox_proportions": bbox_proportions,
                "query": item["instruction"],
                "data_type": item["data_type"],
                "data_source": item["data_source"],
                "img_filename": item["img_filename"],
                "image": image,
            }

    return iter_data()


def prepare_verl_parquet_data_files(dataset_name, max_samples=None):
    """
    Prepare dataset files and convert to parquet format for VERL training.

    Args:
        dataset_name: Name of the dataset to prepare
        max_samples: Maximum number of samples to process (None for all)

    Returns:
        str: Path to the prepared parquet file
    """

    data_iter = load_iterable_dataset(dataset_name, load_image=False)
    # data_iter = load_screenspot_v2()

    # Convert iterator to list of dictionaries
    data_list = []
    print(f"Processing ScreenSpot-v2 data (max_samples: {max_samples})...")

    for i, item in tqdm(enumerate(data_iter)):
        if max_samples and i >= max_samples:
            print(f"Stopping at {i} samples (reached max_samples={max_samples})")
            break

        data_dict = {
            "item_idx": item["item_idx"],
            "bbox_proportions": item["bbox_proportions"],
            "query": item["query"],
            "img_filename": item["img_filename"],
            # "image_width": item["image"].width,
            # "image_height": item["image"].height,
        }
        #             "bbox_coords": item["bbox_coords"],
        # "data_type": item["data_type"],
        # "data_source": item["data_source"],
        if "bbox_coords" in item:
            data_dict["bbox_coords"] = item["bbox_coords"]
        if "data_type" in item:
            data_dict["data_type"] = item["data_type"]
        if "data_source" in item:
            data_dict["data_source"] = item["data_source"]
        data_list.append(data_dict)

    # Create dataset and save as parquet
    print("Creating dataset and saving to parquet...")
    dataset = Dataset.from_list(data_list)

    # Use different filename based on whether it's the full dataset or subset
    if max_samples:
        parquet_path = DATA_DIR / dataset_name / f"{dataset_name}_processed_{max_samples}.parquet"
    else:
        parquet_path = DATA_DIR / dataset_name / f"{dataset_name}_processed_full.parquet"
    dataset.to_parquet(str(parquet_path))
    print(f"Saved {len(data_list)} samples to {parquet_path}")
    return str(parquet_path)


def load_parquet_with_images(parquet_path: str, image_dir: str):
    dataset = Dataset.from_parquet(str(parquet_path))
    for idx, item in enumerate(dataset):
        if ".zip" in str(image_dir):
            with zipfile.ZipFile(image_dir, "r") as zip_ref:
                with zip_ref.open(os.path.join("image", item["img_filename"])) as image_file:
                    image = Image.open(image_file).convert("RGB")
        else:
            image_path = image_dir / item["img_filename"]
            image = Image.open(image_path).convert("RGB")

        yield {
            "item_idx": item["item_idx"],
            "bbox_coords": item.get("bbox_coords", None),
            "bbox_proportions": item["bbox_proportions"],
            "query": item["query"],
            "data_type": item.get("data_type", None),
            "data_source": item.get("data_source", None),
            "img_filename": item["img_filename"],
            "image": image,
        }

        if idx > 10:
            break


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--dataset_name", type=str, default="ScreenSpot-v2")
    args = parser.parse_args()

    print(f"Arguments: dataset_name={args.dataset_name}, max_samples={args.max_samples}")

    # Test the parquet preparation function
    print(
        f"Preparing {args.dataset_name} parquet file {f'({args.max_samples} samples)' if args.max_samples else '(full)'}"
    )
    ret_parquet_path = prepare_verl_parquet_data_files(args.dataset_name, max_samples=args.max_samples)

    # if args.max_samples:
    #     parquet_path = DATA_DIR / "ScreenSpot-v2" / f"screenspot_v2_processed_{args.max_samples}.parquet"
    # else:
    #     parquet_path = DATA_DIR / "ScreenSpot-v2" / "screenspot_v2_processed_full.parquet"

    if args.dataset_name == "ScreenSpot-v2":
        image_dir = DATA_DIR / "ScreenSpot-v2" / "screenspotv2_image"
    elif args.dataset_name.startswith("grounding_train"):
        image_dir = DATA_DIR / "grounding_data" / "image.zip"
    else:
        raise NotImplementedError(f"Dataset {args.dataset_name} not supported yet.")

    print(f"Testing loading from parquet: {ret_parquet_path}")
    # Test loading from parquet
    loaded_data_iter = load_parquet_with_images(ret_parquet_path, image_dir)

    save_test_num = 0
    while save_test_num < 3:
        item = next(loaded_data_iter)
        image = item["image"]
        if item.get("bbox_coords", None) is None:
            width, height = image.width, image.height
            bbox_proportions = item["bbox_proportions"]
            bbox_coords = [
                bbox_proportions[0] * width,
                bbox_proportions[1] * height,
                bbox_proportions[2] * width,
                bbox_proportions[3] * height,
            ]
        else:
            bbox_coords = item["bbox_coords"]
        draw = ImageDraw.Draw(image)
        draw.rectangle(bbox_coords, outline="red", width=3)
        item_idx = item["item_idx"]
        save_path = f"./tmp/{args.dataset_name}_{item_idx}.png"
        image.save(save_path)
        print(f"Save to {save_path}")
        print(f"Query: {item['query']}")
        print(f"Image size: {image.width}x{image.height}")
        print(f"Bbox coords: {bbox_coords}")
        save_test_num += 1


def draw_and_save():
    # Test loading from parquet
    from PIL import ImageDraw

    parquet_path = DATA_DIR / "ScreenSpot-v2" / "screenspot_v2_processed_full.parquet"
    image_dir = DATA_DIR / "ScreenSpot-v2" / "screenspotv2_image"
    loaded_data = load_parquet_with_images(parquet_path=parquet_path, image_dir=image_dir)
    save_dir = Path("./tmp/screenspot_v2/")
    os.makedirs(save_dir, exist_ok=True)
    for item in loaded_data:
        image = item["image"]
        bbox_coords = item["bbox_coords"]
        draw = ImageDraw.Draw(image)
        draw.rectangle(bbox_coords, outline="red", width=3)
        image.save(save_dir / f"{item['item_idx']}.png")
        print(f"Sample item: {item['query']}")
        print(f"Image size: {image.width}x{image.height}")
        print(f"Bbox coords: {bbox_coords}")


def unzip_images_and_save():
    data_path = DATA_DIR / "grounding_train_1225_filter30.json"
    image_zip_path = DATA_DIR / "grounding_data" / "image.zip"
    save_image_dir = DATA_DIR / "unzip_images"

    data = json.load(open(data_path, "r"))
    with zipfile.ZipFile(image_zip_path, "r") as zip_ref:
        for item in data:
            image_name = item["image"]
            if os.path.exists(save_image_dir / image_name):
                continue
            with zip_ref.open(os.path.join("image", image_name)) as image_file:
                image = Image.open(image_file).convert("RGB")
                image.save(save_image_dir / image_name)
                print(f"Saved image: {image_name}")


if __name__ == "__main__":
    # main()
    unzip_images_and_save()
    # draw_and_save()
    # dataset = load_ui_vision()
    # from tqdm import tqdm
    # for item in tqdm(dataset):
    #     pass
