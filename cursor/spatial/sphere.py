from cursor.spatial.eval import VLLMModel
import json, os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
import numpy as np
from collections import defaultdict
import re
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from copy import deepcopy


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def word2num(predicted_answer):
    word_to_number = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16,
        "seventeen": 17,
        "eighteen": 18,
        "nineteen": 19,
        "twenty": 20,
        "no": 0,
    }
    pattern = re.compile(r"\b(" + "|".join(word_to_number.keys()) + r")\b")
    predicted_answer_pattern = re.sub(pattern, lambda x: str(word_to_number[x.group().lower()]), predicted_answer)
    return predicted_answer_pattern


def process_format_num(predicted_answer):
    if predicted_answer.isdigit():
        return predicted_answer
    else:
        predicted_answer_pattern = word2num(predicted_answer)
        numbers = re.findall(r"\b\d+\.?\d*\b", predicted_answer_pattern)
        if numbers:
            return numbers[0]
        else:
            return "-"


def clean_string(input_str):
    # Create a translation table to remove punctuation
    translator = str.maketrans("", "", string.punctuation)

    # Remove punctuation and trailing spaces
    cleaned_str = input_str.translate(translator).strip()

    return cleaned_str.lower()


def num2word(predicted_answer):
    number_to_word = {
        0: "zero",
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
        10: "ten",
        11: "eleven",
        12: "twelve",
        13: "thirteen",
        14: "fourteen",
        15: "fifteen",
        16: "sixteen",
        17: "seventeen",
        18: "eighteen",
        19: "nineteen",
        20: "twenty",
    }
    number_to_word = {str(k): v for k, v in number_to_word.items()}
    pattern = re.compile(r"\b(" + "|".join(number_to_word.keys()) + r")\b")
    predicted_answer_pattern = re.sub(pattern, lambda x: str(number_to_word[x.group().lower()]), predicted_answer)
    return predicted_answer_pattern


def find_unique_nouns(str1, str2):
    str1 = lemmatize_words(str1)
    str2 = lemmatize_words(str2)

    # Tokenize and find unique tokens in str1
    tokens_str1 = word_tokenize(str1)
    tokens_str2 = word_tokenize(str2)

    # POS tag
    pos_tags_str1 = pos_tag(tokens_str1)
    unique_pos_tags_str1 = [tag for tag in pos_tags_str1 if tag[0] not in tokens_str2]

    # Extract nouns (NN, NNS, NNP, NNPS)
    unique_nouns_str1 = {word.lower() for word, pos in unique_pos_tags_str1 if pos.startswith("NN")}

    # Remove common terms labeled as nouns
    common_terms = set(["one"])
    unique_nouns_str1 = {word for word in unique_nouns_str1 if word not in common_terms}

    return unique_nouns_str1


def remove_stopwords(input_str):
    stop_words = set(["a", "the"])
    return " ".join([word.lower() for word in input_str.split() if word.lower() not in stop_words])


def lemmatize_words(text):
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(lemmatized_words)


def longest_matching_subsequence(str1, str2):

    str1 = lemmatize_words(str1)
    str2 = lemmatize_words(str2)

    # Split the strings into words
    words1 = str1.split()
    words2 = str2.split()

    # Lengths of the word lists
    m, n = len(words1), len(words2)

    # Create a DP table initialized with empty lists
    dp = [[([], None, None, None, None) for _ in range(n + 1)] for _ in range(m + 1)]

    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if words1[i - 1] == words2[j - 1]:
                subseq, start1, start2, _, _ = dp[i - 1][j - 1]
                new_start1 = start1 if start1 is not None else i - 1
                new_start2 = start2 if start2 is not None else j - 1
                dp[i][j] = (subseq + [words1[i - 1]], new_start1, new_start2, i - 1, j - 1)
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1], key=lambda x: len(x[0]))

    # The longest subsequence and positions are at the bottom-right corner of the table
    longest_subseq, start1, start2, end1, end2 = dp[m][n]
    if len(longest_subseq) == 0:
        start1 = end1 = m
        start2 = end2 = n

    return longest_subseq, start1, start2, end1, end2


def process_format_mcq(predicted_answer, option_list, text):

    # process only first sentence
    predicted_answer = re.split(r"[.!?]", predicted_answer, maxsplit=1)[0].strip()

    # process predicted answer
    predicted_answer = num2word(clean_string(predicted_answer))

    # process text and detect predictions that mention nouns not in text
    text = num2word(clean_string(text))
    unique_nouns = list(find_unique_nouns(predicted_answer, text))
    option_list_expanded = option_list + unique_nouns

    predicted_answer = remove_stopwords(predicted_answer)
    text = remove_stopwords(text)

    # process option and find match
    len_subseq = {}
    start_subseq = {}
    end_subseq = {}
    for option in option_list_expanded:
        option_pattern = num2word(clean_string(remove_stopwords(option)))

        longest_subseq, start1, start2, end1, end2 = longest_matching_subsequence(predicted_answer, option_pattern)
        len_subseq[option] = len(longest_subseq) / len(option_pattern.split())
        start_subseq[option] = start1
        end_subseq[option] = end1

    max_len_subseq = max(len_subseq.values())
    if max_len_subseq > 0:
        count_max_len_subseq = len([k for k, v in len_subseq.items() if v == max_len_subseq])
        # return longest match
        if count_max_len_subseq == 1:
            selected_option = max(len_subseq, key=len_subseq.get)
            if selected_option not in unique_nouns:
                return selected_option

        # return earliest match
        min_start_subseq = min(start_subseq.values())
        min_end_subseq = min(end_subseq.values())
        count_min_start_subseq = len([k for k, v in start_subseq.items() if v == min_start_subseq])
        count_min_end_subseq = len([k for k, v in end_subseq.items() if v == min_end_subseq])
        if count_min_start_subseq == 1:
            selected_option = min(start_subseq, key=start_subseq.get)
            if selected_option not in unique_nouns:
                return selected_option
        if count_min_end_subseq == 1:
            selected_option = min(end_subseq, key=end_subseq.get)
            if selected_option not in unique_nouns:
                return selected_option

    return "-"


def compute_accuracy(correct, total):
    return correct / total


def compute_validity(valid, total):
    return valid / total


THINKING_SYSTEM_PROMPT = (
    f"You are currently a senior expert in spatial reasoning."
    f"\nGiven an Image, a Question and Options, your task is to answer the question based on your spatial understanding of the image."
    f"\nYour response must contain a thinking process before the answer. The thinking process is enclosed in <think> tag, and the answer is enclosed in <answer> tag."
)


def extract_answer_part(response):
    # try to extract <answer> </answer>, if not <answer> </answer>, try to remove <think> </think>
    # if there is no <think> </think>, return the last line by /n

    # Try to extract content within <answer> tags
    answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()

    # If no <answer> tags, try to remove <think> tags
    think_pattern = r"<think>.*?</think>"
    response_without_think = re.sub(think_pattern, "", response, flags=re.DOTALL).strip()
    if response_without_think != response.strip():
        # <think> tags were found and removed
        return response_without_think

    # If no <think> tags, return the last line
    lines = response.strip().split("\n")
    return lines[-1].strip() if lines else ""


class COCODataset(Dataset):
    def __init__(self, annotations_json, img_dir, processor, thinking):
        self.annotations_dir = "/mnt/ceph_rbd/data/SPHERE-VLM/eval_datasets/coco_test2017_annotations"
        self.annotations_json = annotations_json
        self.img_dir = img_dir
        self.processor = processor
        self.thinking = thinking
        with open(os.path.join(self.annotations_dir, annotations_json + ".json"), "r") as f:
            self.coco_annotations = json.load(f)

    def __len__(self):
        return len(self.coco_annotations)

    def __getitem__(self, idx):
        ann = self.coco_annotations[idx]
        source_img_id = ann["metadata"]["source_img_id"]
        full_source_img_id = (12 - len(source_img_id)) * "0" + source_img_id
        img_path = f"{self.img_dir}/{full_source_img_id}.jpg"
        image = Image.open(img_path).convert("RGB")

        return idx, image, img_path, ann

    def get_dataloader(self, batch_size):

        def collate_fn(batch):
            prepared_batch = []
            for idx, image, img_path, ann in batch:
                if "raw_prediction" not in ann:
                    ann["raw_prediction"] = {}
                    ann["prediction"] = {}
                    ann["valid"] = {}
                    ann["correct"] = {}

                for seed in range(5):
                    set_seed(seed)
                    if ann["metadata"]["format"] == "num":
                        query = "Give the answer in number format."
                    else:
                        # shuffle options
                        options = ann["option"]
                        query = " or ".join(options).capitalize() + "?"
                    text = " ".join([ann["question"], query])

                    if self.thinking:
                        if ann["metadata"]["format"] == "num":
                            system_prompt = THINKING_SYSTEM_PROMPT + "\nMake sure answer the number within <answer> </answer> tags."
                        else:
                            system_prompt = THINKING_SYSTEM_PROMPT + "\nMake sure answer the option within <answer> </answer> tags."
                        messages = [
                            {
                                "role": "system",
                                "content": system_prompt,
                            },
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": text},
                                    {"type": "image"},
                                ],
                            },
                        ]
                    else:
                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": f"{text} Answer the question directly."},
                                    {"type": "image"},
                                ],
                            },
                        ]
                    prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    multi_modal_data = {"image": [image]}
                    vllm_inputs = {
                        "prompt": prompt,
                        "multi_modal_data": multi_modal_data,
                    }
                    prepared_batch.append(
                        {
                            "vllm_inputs": vllm_inputs,
                            "input": text,
                            "ann": ann,
                            "idx": idx,
                            "seed": seed,
                        }
                    )
                    # raw_prediction = self.model.predict(
                    #     image=image,
                    #     image_path=image_path,
                    #     text=text
                    # )
                    # raw_prediction_list.append(raw_prediction)

            return prepared_batch

        return DataLoader(self, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


def run_single_task(exp_name, model: VLLMModel, processor, annotations_json, batch_size, thinking):
    dataset_name = "SPHERE"
    save_suffix = "w_predictions"
    annotations_dir = "/mnt/ceph_rbd/data/SPHERE-VLM/eval_datasets/coco_test2017_annotations"
    img_dir = "/mnt/ceph_rbd/data/SPHERE-VLM/eval_datasets/coco_test2017"

    # Load dataset
    dataset = COCODataset(annotations_json, img_dir, processor, thinking)

    dataloader = dataset.get_dataloader(batch_size=batch_size)

    save_path = os.path.join(annotations_dir, f"{annotations_json}_{save_suffix}.json")
    dataset_w_predictions = dataset.coco_annotations
    correct = 0
    valid = 0
    total = 0
    for batch in dataloader:
        vllm_inputs = [item["vllm_inputs"] for item in batch]
        # breakpoint()
        responses = model.batch_predict(vllm_inputs)

        # group the prediction by idx, difference is seed
        id_to_predictions = defaultdict(dict)
        for bid in range(len(batch)):
            idx = batch[bid]["idx"]
            id_to_predictions[idx][batch[bid]["seed"]] = {
                "item": batch[bid],
                "seed": batch[bid]["seed"],
                "response": responses[bid],
            }

        for idx in id_to_predictions.keys():
            prediction_list = []
            raw_prediction_list = []
            for seed in range(5):
                item = id_to_predictions[idx][seed]
                ann = item["item"]["ann"]

                if ann["metadata"]["format"] == "num":
                    query = "Give the answer in number format."
                else:
                    # shuffle options
                    options = ann["option"]
                    query = " or ".join(options).capitalize() + "?"
                text = " ".join([ann["question"], query])

                raw_prediction = item["response"]
                raw_prediction_list.append(raw_prediction)
                # processed prediction
                if thinking:
                    raw_prediction = extract_answer_part(raw_prediction)
                if ann["metadata"]["format"] == "num":
                    prediction = process_format_num(clean_string(raw_prediction))
                else:
                    prediction = process_format_mcq(raw_prediction, ann["option"], text)

                prediction_list.append(prediction)

            answer = clean_string(ann["answer"])
            valid_i = np.mean([p != "-" for p in prediction_list])
            correct_i = np.mean([answer == p for p in prediction_list])  # invalid answers are counted as wrong

            correct += correct_i
            valid += valid_i
            total += 1

            ann["raw_prediction"][exp_name] = raw_prediction_list
            ann["prediction"][exp_name] = prediction_list
            ann["valid"][exp_name] = valid_i
            ann["correct"][exp_name] = correct_i
            dataset_w_predictions[idx] = deepcopy(ann)

        with open(save_path, "w") as f:
            json.dump(dataset_w_predictions, f, indent=4)

        accuracy = compute_accuracy(correct, total)
        validity = compute_validity(valid, total)
        # return accuracy, validity

        # id_to_predictions[idx]["prediction"] = prediction_list

    # Save results
    save_path = os.path.join(annotations_dir, "result.json")
    if os.path.exists(save_path):
        with open(save_path, "r") as json_file:
            results = json.load(json_file)
    else:
        results = {}

    eval_qn_type = "all"

    annotations_json = annotations_json if eval_qn_type == "all" else f"{annotations_json}-{eval_qn_type}"
    if annotations_json not in results:
        results[annotations_json] = {"accuracy": {}, "validity": {}}

    results[annotations_json]["accuracy"][exp_name] = accuracy
    results[annotations_json]["validity"][exp_name] = validity

    with open(save_path, "w") as f:
        json.dump(results, f, indent=4, sort_keys=True)

    print(f"Accuracy Score: {accuracy}, Validity Score: {validity}")


def run_all_tasks():

    # Single skill annotations
    single_skill_json = [
        "single_skill/size_only",
        "single_skill/distance_only",
        "single_skill/position_only",
        "single_skill/counting_only-paired-distance_and_counting",
        "single_skill/counting_only-paired-position_and_counting",
    ]

    # Combined 2-skill annotations
    combine_2_skill_json = [
        "combine_2_skill/distance_and_size",
        "combine_2_skill/distance_and_counting",
        "combine_2_skill/position_and_counting",
    ]

    # Reasoning annotations
    reasoning_json = [
        "reasoning/object_manipulation",
        "reasoning/object_occlusion",
        "reasoning/object_manipulation_w_intermediate",
        "reasoning/object_occlusion_w_intermediate",
    ]

    # All annotation JSONs combined
    all_annotation_json = single_skill_json + combine_2_skill_json + reasoning_json

    # model_path = "Salesforce/GTA1-7B-2507"
    # model_path = "inclusionAI/GUI-G2-7B"
    # exp_name = "guig2"
    model_path = "microsoft/GUI-Actor-7B-Qwen2.5-VL"
    exp_name = "guiactor"
    # model_path = "/mnt/ceph_rbd/models/Qwen/Qwen2.5-VL-7B-Instruct_resaved"
    # exp_name = "qwen25vl7b"

    # model_path = "/mnt/ceph_rbd/models/GUI-Cursor"
    # exp_name = "gta1"
    thinking = False
    batch_size = 32
    model = VLLMModel(model_path=model_path, tp_size=1, batch_size=batch_size)
    processor = model.processor
    # model = None
    # from transformers import AutoProcessor
    # processor = AutoProcessor.from_pretrained(model_path)
    for annotations_json in all_annotation_json:
        print(f"Running evaluation for {annotations_json}...")
        run_single_task(exp_name, model, processor, annotations_json, batch_size, thinking)


if __name__ == "__main__":
    run_all_tasks()

# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_name", type=str, default="instruct_blip")
#     parser.add_argument("--annotations_dir", type=str, default="eval_datasets/coco_test2017_annotations")
#     parser.add_argument("--annotations_json", type=str, default="single_skill/size_only")
#     parser.add_argument("--img_dir", type=str, default="eval_datasets/coco_test2017")
#     parser.add_argument("--save_predictions", action="store_true")
#     parser.add_argument("--save_suffix", type=str, default="w_predictions")
#     parser.add_argument("--results_filename", type=str, default="results")
#     parser.add_argument("--num_seeds", type=int, default=5)
#     parser.add_argument("--eval_saved_predictions", action="store_true")
#     parser.add_argument(
#         "--eval_qn_type", type=str, default="all", choices=["all", "intermediate", "final", "allo", "ego"]
#     )
#     args = parser.parse_args()

#     main(args)
