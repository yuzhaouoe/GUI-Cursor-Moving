from cursor.prepare_data import load_iterable_dataset
import json
import numpy as np
from collections import defaultdict


def load_jsonl(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    item_idx_to_data = dict()  # deduplicate
    for item in data:
        item_idx_to_data[item["item_idx"]] = item
    data = list(item_idx_to_data.values())
    return data


def main():

    path = "/mnt/ceph_rbd/GUI-Cursor-Moving/outputs/UI-Vision/focus05/predictions.jsonl"
    data = load_jsonl(path)
    iter_data = load_iterable_dataset("UI-Vision", load_image=False)
    category_acc = defaultdict(list)
    num_items = 0
    for pred, item in zip(data, iter_data):
        category = item["data_source"]
        cur_acc = pred["within_bbox_history"][-1]
        category_acc[category].append(cur_acc)
        num_items += 1
        if num_items >= 832:
            break

    
    for category in category_acc.keys():
        acc_list = category_acc[category]
        mean_acc = np.mean(acc_list)
        print(f"Category: {category}, Mean Acc: {mean_acc:.1%}, Num Samples: {len(acc_list)}")

    num_samples = len(data)
    all_acc = [pred["within_bbox_history"][-1] for pred in data]
    overall_mean_acc = np.mean(all_acc)
    print(f"Overall Mean Acc: {overall_mean_acc:.1%}, Total Samples: {num_samples}")



if __name__ == "__main__":
    main()
