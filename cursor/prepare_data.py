from cursor import DATA_DIR
from huggingface_hub import snapshot_download
import zipfile
import shutil
import os
import json
from PIL import Image


def load_iterable_dataset(dataset_name: str):

    if dataset_name == "ScreenSpot-v2":
        dataset = load_screenspot_v2()
    elif dataset_name == "ScreenSpot-Pro":
        raise NotImplementedError
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")
    return dataset


def load_screenspot_v2():
    if not (DATA_DIR / "ScreenSpot-v2").exists():
        hf_path = "OS-Copilot/ScreenSpot-v2"
        snapshot_download(repo_id=hf_path, repo_type="dataset", local_dir=DATA_DIR / "ScreenSpot-v2")
        with zipfile.ZipFile(DATA_DIR / "ScreenSpot-v2" / "screenspotv2_image.zip", "r") as zip_ref:
            zip_ref.extractall(DATA_DIR / "ScreenSpot-v2")
        os.remove(DATA_DIR / "ScreenSpot-v2" / "screenspotv2_image.zip")

    data = []
    for files in ["screenspot_desktop_v2.json", "screenspot_mobile_v2.json", "screenspot_web_v2.json"]:
        data.extend(json.load(open(DATA_DIR / "ScreenSpot-v2" / files, "r")))

    def iter_data():
        for item_idx, item in enumerate(data):
            image_path = DATA_DIR / "ScreenSpot-v2" / "screenspotv2_image" / item["img_filename"]
            image = Image.open(image_path).convert("RGB")
            yield {
                "item_idx": item_idx,
                "bbox_coords": item["bbox"],
                "instruction": item["instruction"],
                "data_type": item["data_type"],
                "data_source": item["data_source"],
                "img_filename": item["img_filename"],
                "image": image,
            }

    return iter_data()


def main():
    iterable_data = load_screenspot_v2()

    for item in iterable_data:
        print(item)
        break


if __name__ == "__main__":
    main()
