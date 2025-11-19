from cursor.prepare_data import load_iterable_dataset
import json
import numpy as np
from collections import defaultdict


def load_jsonl(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    # sort by item_idx
    # data.sort(key=lambda x: x["item_idx"])

    item_idx_to_data = dict() # deduplicate
    for item in data:
        item_idx_to_data[item["item_idx"]] = item
    data = list(item_idx_to_data.values())
    return data


def print_results(path):

    
    data = load_jsonl(path)
    # msg_path = "/mnt/ceph_rbd/GUI-Cursor-Moving/outputs/OSWorld-G_refined/refine3/message_history.jsonl"
    # messages = load_jsonl(msg_path)
    category_info = json.load(open("/mnt/ceph_rbd/data/OSWorld-G/OSWorld-G/benchmark/classification_result.json", "r"))
    category_info = category_info["classified"]

    category_acc = defaultdict(list)
    id_to_category = dict()
    for category in category_info.keys():
        for item in category_info[category]:
            id_to_category[item["id"]] = category

    for pred in data:
        cur_id = pred["item_idx"]
        category = id_to_category[cur_id]
        cur_acc = pred["within_bbox_history"][-1]
        category_acc[category].append(cur_acc)

        # if category == "refusal":
        #     print(f"Refusal Message: {msg}")

    for category in category_acc.keys():
        acc_list = category_acc[category]
        mean_acc = np.mean(acc_list)
        print(f"Category: {category}, Mean Acc: {mean_acc:.1%}, Num Samples: {len(acc_list)}")

    num_samples = len(data)
    all_acc = [pred["within_bbox_history"][-1] for pred in data]
    overall_mean_acc = np.mean(all_acc)
    print(f"Overall Mean Acc: {overall_mean_acc:.1%}, Total Samples: {num_samples}")



def main():
    paths = [
        # "/mnt/ceph_rbd/GUI-Cursor-Moving/outputs/OSWorld-G_refined/refine3/predictions.jsonl",
        "/mnt/ceph_rbd/GUI-Cursor-Moving/outputs/OSWorld-G_refined/focus05/predictions.jsonl",
        "/mnt/ceph_rbd/GUI-Cursor-Moving/outputs/OSWorld-G_refined/uitars-175-withccf/predictions.jsonl",
        "/mnt/ceph_rbd/GUI-Cursor-Moving/outputs/OSWorld-G_refined/GTA1-7B/predictions.jsonl",
    ]
    for path in paths:
        print(f"Results for {path}:")
        print_results(path)
        print("\n")

if __name__ == "__main__":

    main()