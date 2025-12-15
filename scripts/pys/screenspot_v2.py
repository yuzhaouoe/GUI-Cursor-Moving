from cursor.prepare_data import load_iterable_dataset
from pathlib import Path
import json
from collections import defaultdict


def load_jsonl(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    data = sorted(data, key=lambda x: x["item_idx"])
    return data

def main():

    exp_name = "GUI-Cursor-uitars-160steps"
    path = Path("./outputs") / "ScreenSpot-v2" / exp_name / "predictions.jsonl"
    predictions = load_jsonl(path)
    iter_data = load_iterable_dataset("ScreenSpot-v2", load_image=False)

    category_acc = defaultdict(list)
    for pred, item in zip(predictions, iter_data):
        assert pred["item_idx"] == item["item_idx"]
        source = item["data_source"]
        if item["data_source"] in ["windows", "macos"]:
            category = "Desktop"
        elif item["data_source"] in ["android", "ios"]:
            category = "Mobile"
        else:
            category = "Web"
        category = f"{category}_{item['data_type']}"
        correct = pred["within_bbox_history"][-1]
        category_acc[category].append(correct)
    
    for category in category_acc.keys():
        acc_list = category_acc[category]
        mean_acc = sum(acc_list) / len(acc_list)
        correct_count = sum(1 for x in acc_list if x)
        print(f"Category: {category}, Mean Acc: {mean_acc:.1%}, Correct: {correct_count} Num Samples: {len(acc_list)}")
    # overall acc
    num_samples = len(predictions)
    all_acc = [pred["within_bbox_history"][-1] for pred in predictions]
    overall_mean_acc = sum(all_acc) / len(all_acc)
    print(f"Overall Mean Acc: {overall_mean_acc:.1%}, Total Samples: {num_samples}")

if __name__ == "__main__":

    main()

