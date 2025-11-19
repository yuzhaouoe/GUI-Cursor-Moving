# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import logging
import os
import re
from collections import defaultdict
from typing import Optional
from pathlib import Path
from PIL import Image

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

from cursor.move_loop import init_environment_and_state, observe_get_message_input
from cursor.move_loop import ACTION_EXTRACTOR_FN, MESSAGE_HISTORY_UPDATE_FN
from qwen_vl_utils import smart_resize
import zipfile

logger = logging.getLogger(__name__)


def collate_fn(data_list: list[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, \*dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}


class CursorStateDataset(Dataset):
    """
    Load and preprocess RLHF data from Parquet files.

    - Caches files locally.
    - Reads into a HuggingFace Dataset and tokenizes prompts.
    - Optionally handles images/videos via a ProcessorMixin.
    - Filters prompts over a max length.
    - Supports resuming from checkpoints.

    Args:
        data_files (str or list): Path(s) to Parquet file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        cursor_config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: Optional[int] = None,
    ):
        if not isinstance(data_files, list | ListConfig):
            data_files = [data_files]

        file_name_split = data_files[0].split("/")
        if "ScreenSpot-v2" in file_name_split:
            self.dataset_name = "ScreenSpot-v2"
            self.image_dir = Path("/mnt/ceph_rbd/data/ScreenSpot-v2/screenspotv2_image")
        elif "grounding_train" in file_name_split[-1]:
            self.dataset_name = "grounding_train"
            self.image_dir = "/mnt/ceph_rbd/data/grounding_data/image.zip"
        else:
            raise NotImplementedError(f"dataset_name not detected from {data_files[0]}")

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())
        self.use_shm = config.get("use_shm", False)
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False
        self.return_multi_modal_inputs = config.get("return_multi_modal_inputs", True)
        self.system_prompt_format = open(cursor_config.system_prompt_format_path, "r").read()
        self.first_turn_prompt_format = open(cursor_config.first_turn_prompt_format_path, "r").read()
        self.reply_prompt_format = open(cursor_config.reply_prompt_format_path, "r").read()
        self.action_extractor_fn = ACTION_EXTRACTOR_FN[cursor_config.action_extractor_fn_name]
        self.message_history_update_fn = MESSAGE_HISTORY_UPDATE_FN[cursor_config.message_history_update_fn_name]
        self.cursor_config = cursor_config

        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}")

    #     self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)

    # def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset = None):
    #     # filter out too long prompts
    #     if self.filter_overlong_prompts:
    #         print(f"Filtering prompts longer than {self.max_prompt_length} tokens...")
    #         tokenizer = self.tokenizer
    #         processor = self.processor
    #         prompt_key = self.prompt_key
    #         image_key = self.image_key
    #         video_key = self.video_key

    #         if processor is not None:
    #             from verl.utils.dataset.vision_utils import process_image, process_video

    #             def doc2len(doc) -> int:
    #                 messages = self._build_messages(doc)
    #                 raw_prompt = self.processor.apply_chat_template(
    #                     messages, add_generation_prompt=True, tokenize=False
    #                 )
    #                 images = [process_image(image) for image in doc[image_key]] if image_key in doc else None
    #                 videos = [process_video(video) for video in doc[video_key]] if video_key in doc else None

    #                 return len(processor(text=[raw_prompt], images=images, videos=videos)["input_ids"][0])

    #         else:

    #             def doc2len(doc) -> int:
    #                 return len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True))

    #         dataframe = dataframe.filter(
    #             lambda doc: doc2len(doc) <= self.max_prompt_length,
    #             num_proc=self.num_workers,
    #             desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
    #         )

    #         print(f"filter dataset len: {len(dataframe)}")
    #     return dataframe

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __len__(self):
        return len(self.dataframe)

    # def _build_messages(self, example: dict):
    #     query = example["query"]
    #     img_filename = example["img_filename"]
    #     image_path = self.image_dir / img_filename
    #     image = Image.open(image_path).convert("RGB")
    #     # todo: we don't need this, let's init the env and state to build message
    #     # messages: list = example.pop(self.prompt_key)

    #     # if self.image_key in example or self.video_key in example:
    #     #     for message in messages:
    #     #         content = message["content"]
    #     #         content_list = []
    #     #         segments = re.split("(<image>|<video>)", content)
    #     #         segments = [item for item in segments if item != ""]
    #     #         for segment in segments:
    #     #             if segment == "<image>":
    #     #                 content_list.append({"type": "image"})
    #     #             elif segment == "<video>":
    #     #                 content_list.append({"type": "video"})
    #     #             else:
    #     #                 content_list.append({"type": "text", "text": segment})

    #     #         message["content"] = content_list

    #     return messages
    def __getitem__(self, item):
        row_dict: dict = self.dataframe[item]
        img_filename = row_dict["img_filename"]
        tensor_item_idx = torch.tensor(row_dict["item_idx"], dtype=torch.long)
        if ".zip" in str(self.image_dir):
            with zipfile.ZipFile(self.image_dir, 'r') as zip_ref:
                with zip_ref.open(os.path.join("image", img_filename)) as image_file:
                    image = Image.open(image_file).convert("RGB")
        else:
            image_path = self.image_dir / img_filename
            image = Image.open(image_path).convert("RGB")

        min_tokens_per_img = 512
        max_tokens_per_img = 2691 # 2645 # 2691  # (1092 // 28) * (1932 // 28)
        width, height = image.size
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=28,
            min_pixels=min_tokens_per_img * 28 * 28 if min_tokens_per_img else 0,
            max_pixels=max_tokens_per_img * 28 * 28 if max_tokens_per_img else 16384 * 28 * 28,
        )
        image = image.resize((resized_width, resized_height))

        return_dict = {
            "item_idx": tensor_item_idx,  # we must have a tensor style value to return
            "uid": row_dict["item_idx"],
            "image": image,
            # "image_path": image_path,
            "query": row_dict["query"],
            "bbox_proportions": row_dict["bbox_proportions"],
        }
        return return_dict

    def image_and_state__getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        img_filename = row_dict["img_filename"]
        image_path = self.image_dir / img_filename
        image = Image.open(image_path).convert("RGB")

        min_tokens_per_img = 512
        max_tokens_per_img = 2691  # (1092 // 28) * (1932 // 28)
        width, height = image.size
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=28,
            min_pixels=min_tokens_per_img * 28 * 28 if min_tokens_per_img else 0,
            max_pixels=max_tokens_per_img * 28 * 28 if max_tokens_per_img else 16384 * 28 * 28,
        )
        image = image.resize((resized_width, resized_height))

        example = {
            "image": image,
            "query": row_dict["query"],
            "bbox_proportions": row_dict["bbox_proportions"],
            "item_idx": row_dict["item_idx"],
        }
        state = init_environment_and_state(
            example,
            self.system_prompt_format,
            self.first_turn_prompt_format,
            self.reply_prompt_format,
            self.action_extractor_fn,
            self.message_history_update_fn,
            max_steps=self.cursor_config.max_steps,
            output_dir=None,
        )
        tensor_item_idx = torch.tensor(row_dict["item_idx"], dtype=torch.long)
        return_dict = {
            "state": state,
            "item_idx": tensor_item_idx,  # we must have a tensor style value to return
            "uid": row_dict["item_idx"],
        }

        # return return_dict
        #     {
        #     "inputs": inputs,
        #     "observation": observation,
        #     "text_input": text_input,
        #     "item_idx": state.item_idx,
        #     "message_inputs": message_inputs,
        # }
        observations = observe_get_message_input(self.processor, state)
        return_dict["image_observation"] = observations["observation"]
        return_dict["text_observation"] = observations["text_input"]
        # input_ids = self.processor.tokenizer.tokenize(observations["inputs"]["prompt"])
        raw_prompt = observations["inputs"]["prompt"]

        return_dict["raw_messages"] = observations["message_inputs"]
        return return_dict

        images = observations["inputs"]["multi_modal_data"]["image"]
        model_inputs = self.processor(text=[raw_prompt], images=images, videos=None, return_tensors="pt")

        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")
        if "second_per_grid_ts" in model_inputs:
            model_inputs.pop("second_per_grid_ts")
        # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
        # row_dict["multi_modal_data"] = observations["inputs"]["multi_modal_data"]
        return_dict["multi_modal_data"] = observations["inputs"]["multi_modal_data"]

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # qwen-vl mrope
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
        elif self.processor is not None and "Glm4vImageProcessor" in self.processor.image_processor.__class__.__name__:
            from verl.models.transformers.glm4v import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        return_dict["input_ids"] = input_ids[0]
        return_dict["attention_mask"] = attention_mask[0]
        return_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        assert self.return_raw_chat
        if self.return_raw_chat:
            return_dict["raw_prompt"] = observations["message_inputs"]
        # get prompts with chat template
        if self.return_full_prompt:
            return_dict["full_prompts"] = raw_prompt  # array of strings

        if "extra_info" not in return_dict or return_dict["extra_info"] is None:
            return_dict["extra_info"] = dict()
        index = return_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = return_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = return_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = return_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, return_dict["data_source"])
        return_dict["index"] = index
        return_dict["tools_kwargs"] = tools_kwargs
        return_dict["interaction_kwargs"] = interaction_kwargs
        return return_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()
