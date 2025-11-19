import asyncio
import heapq
import logging
import os
import random
from abc import ABC, abstractmethod
from typing import Any, Optional

import hydra
import numpy as np
import ray
from sympy import total_degree
import torch
from cachetools import LRUCache
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict
from tensordict import TensorDict
from transformers import AutoProcessor, AutoTokenizer

from verl.experimental.reward import RewardManagerWorker
from verl.protocol import DataProto
from verl.single_controller.ray.base import RayWorkerGroup
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask
from verl.utils.rollout_trace import (
    RolloutTraceConfig,
    rollout_trace_attr,
    rollout_trace_op,
)
from verl.utils.transferqueue_utils import tqbridge
from verl.workers import reward_manager
from verl.workers.rollout.replica import TokenOutput, get_rollout_replica_class
from yarl import Query

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

from verl.experimental.agent_loop import AgentLoopManager
from verl.experimental.agent_loop.agent_loop import AgentLoopWorker, AgentLoopWorkerBase
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, AgentLoopMetrics
from verl.experimental.agent_loop.agent_loop import get_trajectory_info
from verl.experimental.agent_loop.agent_loop import _InternalAgentLoopOutput, _DummyConfig
from verl.experimental.agent_loop.agent_loop import _agent_loop_registry, register
from uuid import uuid4
from verl.utils.profiler import simple_timer


from typing import List, Dict, Tuple, Optional

from verl.workers.rollout.vllm_rollout.vllm_async_server import _qwen2_5_vl_dedup_image_tokens
from copy import deepcopy
from pydantic import BaseModel, ConfigDict
from cursor.rl.cursor_reward_loop_manager import CursorRewardLoopManager

from PIL import Image
from qwen_vl_utils import smart_resize

from cursor.env_and_state import State, Environment, convert_image_to_cursor_position
from cursor.move_loop import init_environment_and_state
from cursor.move_loop import ACTION_EXTRACTOR_FN, MESSAGE_HISTORY_UPDATE_FN
from cursor.move_loop import observe_get_raw_message_input, observe_get_message_input


def get_logit_bias(processor, processor_name):
    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    bias_dict = {image_token_id: -100}

    if processor_name == "Qwen2_5_VLProcessor":
        bad_token = processor.tokenizer.encode(" addCriterion")[0]
        bias_dict[bad_token] = -100

    return bias_dict


class _InternalCursorLoopOutput(AgentLoopOutput):
    """Internal agent loop output with padded sequences."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_ids: Optional[torch.Tensor]
    """Padded prompt token ids."""
    response_ids: Optional[torch.Tensor]
    """Padded response token ids."""
    input_ids: Optional[torch.Tensor]
    """Padded input ids(prompt_ids + response_ids)."""
    position_ids: Optional[torch.Tensor]
    """Padded position ids."""
    response_mask: Optional[torch.Tensor]
    """Padded response mask."""
    attention_mask: Optional[torch.Tensor]
    """Padded attention mask."""
    response_logprobs: Optional[torch.Tensor] = None
    """Padded log probabilities for the response tokens."""
    multi_modal_inputs: Optional[dict[str, torch.Tensor]] = None
    """Multi-modal inputs for processors (e.g., pixel_values, image_grid_thw)."""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""


@register("cursor_agent_loop")
class CursorAgentLoop(AgentLoopBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length
        # self.max_model_len = self.config.actor_rollout_ref.rollout.max_model_len
        self.max_prompt_length = self.config.data.max_prompt_length
        self.max_response_length = self.config.data.max_response_length
        self.max_sequence_length = self.max_prompt_length + self.max_response_length
        self.apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})

        self.cursor_config = self.config.cursor
        self.im_start_token_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_end_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.assistant_token_id = self.tokenizer.convert_tokens_to_ids("assistant")
        self.linebreak_token_id = self.tokenizer.encode("\n")[0]

        self.processor_name: str = self.processor.__class__.__name__
        self.logit_bias: dict = get_logit_bias(self.processor, self.processor_name)

        self.system_prompt_format = open(self.cursor_config.system_prompt_format_path, "r").read()
        self.first_turn_prompt_format = open(self.cursor_config.first_turn_prompt_format_path, "r").read()
        self.reply_prompt_format = open(self.cursor_config.reply_prompt_format_path, "r").read()
        self.action_extractor_fn = ACTION_EXTRACTOR_FN[self.cursor_config.action_extractor_fn_name]
        self.message_history_update_fn = MESSAGE_HISTORY_UPDATE_FN[self.cursor_config.message_history_update_fn_name]

    async def init_state(self, kwargs):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        # image_path = kwargs["image_path"]
        image = kwargs["image"]
        query = kwargs["query"]
        bbox_proportions = kwargs["bbox_proportions"]
        item_idx = kwargs["uid"]

        # begin process image
        # image = Image.open(image_path).convert("RGB")
        # min_tokens_per_img = 512
        # max_tokens_per_img = 2691  # (1092 // 28) * (1932 // 28)
        # width, height = image.size
        # resized_height, resized_width = smart_resize(
        #     height,
        #     width,
        #     factor=28,
        #     min_pixels=min_tokens_per_img * 28 * 28 if min_tokens_per_img else 0,
        #     max_pixels=max_tokens_per_img * 28 * 28 if max_tokens_per_img else 16384 * 28 * 28,
        # )
        # image = image.resize((resized_width, resized_height))
        # end process image


        example = {
            "image": image,
            "query": query,
            "bbox_proportions": bbox_proportions,
            "item_idx": item_idx,
        }

        state = init_environment_and_state(
            example,
            self.system_prompt_format,
            self.first_turn_prompt_format,
            self.reply_prompt_format,
            self.action_extractor_fn,
            self.message_history_update_fn,
            max_steps=self.cursor_config.max_steps,
            max_global_steps=None,
            output_dir=None,
        )
        return state

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:

        # print("kwargs keys")
        # print(kwargs.keys())
        # messages = list(kwargs["raw_prompt"])
        # image_data = (kwargs.get("multi_modal_data") or {}).get("image", None)
        metrics = {}
        request_id = uuid4().hex  # consider to change to example_idx-trajectory_idx-turn_idx

        sampling_params.update({"logit_bias": self.logit_bias, "max_tokens": 512})

        # if kwargs["validate"] is False:
        #     state: State = deepcopy(kwargs["state"])  # avoid sharing same state with different generation
        # else:
        #     state: State = kwargs["state"]

        # messages = kwargs["raw_messages"]

        state: State = await self.init_state(kwargs)

        observations = observe_get_message_input(self.processor, state)

        image_observation = observations["observation"]
        text_observation = observations["text_input"]
        messages = observations["message_inputs"]
        # image_observation = kwargs["image_observation"]
        # text_observation = kwargs["text_observation"]

        image_data = [image_observation]

        # observations = await self.loop.run_in_executor(
        #     None, lambda: observe_get_raw_message_input(state, return_message_inputs=True)
        # )
        # image_observation = observations["observation"]
        # text_observation = observations["text_input"]
        # messages = observations["message_inputs"]
        # assert len(observations["images"]) == 1
        # assert observations["images"][0] is image_observation
        # image_data = [image_observation]
        cur_input_ids: List[int] = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, **self.apply_chat_template_kwargs
            ),
        )
        # messages = list(kwargs["raw_prompt"])
        # cur_input_ids: List[int] = await self.loop.run_in_executor(
        #     None,
        #     lambda: self.tokenizer.apply_chat_template(
        #         messages, add_generation_prompt=True, tokenize=True, **self.apply_chat_template_kwargs
        #     ),
        # )
        # multi-turn cursor moving
        # image_observation = kwargs["image_observation"]
        # text_observation = kwargs["text_observation"]
        # assert len(kwargs["multi_modal_data"]["image"]) == 1
        # image_data = [kwargs["multi_modal_data"]["image"][0]]
        # image_observation = image_data[-1]

        # multi_modal_data = {"image": image_data}
        assistant_mask = []
        initial_prompt_ids = None
        total_num_turns = 0
        num_actions = 0
        with simple_timer("generate_sequences", metrics):

            while True:

                with simple_timer(f"generate_action_{num_actions}", metrics):

                    if len(cur_input_ids) > self.max_sequence_length:
                        breakpoint()
                        raise RuntimeError("Input length exceeds maximum sequence length.")

                    output = await self.server_manager.generate(
                        request_id=request_id,
                        prompt_ids=cur_input_ids,
                        sampling_params=sampling_params,
                        image_data=image_data,
                    )

                    vllm_output_ids = output.token_ids  # assistant output
                    vllm_input_ids = output.input_ids  # the input_ids processed by vllm

                    if initial_prompt_ids is None:
                        initial_prompt_ids = vllm_input_ids

                    new_input_mask = [0] * (len(vllm_input_ids) - len(assistant_mask))
                    assistant_mask.extend(new_input_mask)

                    new_generation_mask = [1] * len(vllm_output_ids)
                    assistant_mask.extend(new_generation_mask)

                    if output.token_ids[-1] != self.im_end_token_id:
                        # replace the last token as im_end, and records it in assistant_mask because it is not generated by the model
                        vllm_output_ids[-1] = self.im_end_token_id
                        assistant_mask[-1] = 0  # im_end is not generated by the model

                    # update state based on the assistant output
                    output_text = await self.loop.run_in_executor(
                        None, lambda: self.tokenizer.decode(vllm_output_ids, skip_special_tokens=True)
                    )
                    state.update_state(image_observation, text_observation, output_text)

                    # for next turn generation
                    cur_input_ids = vllm_input_ids + vllm_output_ids

                    assert len(cur_input_ids) == len(assistant_mask)

                    if state.has_stopped:
                        break

                    total_num_turns += 2
                    # if DEBUG:
                    #     print(vllm_output_ids[-1])
                    #     breakpoint()
                    # check if the last token is <|im_end|> or linebreak

                    # yes: the last token of vllm_output_ids is im_end without a linebreak
                    # we will add a linebreak to concatenate it with new message ids (by removing a dummy system prompt)

                    # get new observation from environment based on the current state
                    observations = await self.loop.run_in_executor(
                        None, lambda: observe_get_raw_message_input(state, return_message_inputs=False)
                    )
                    image_observation = observations["observation"]
                    text_observation = observations["text_input"]

                    content_list = [{"type": "image", "image": image_observation}]
                    if text_observation is not None and text_observation != "":
                        content_list.append({"type": "text", "text": text_observation})

                    new_messages = [
                        {"role": "system", "content": ""},  # for applying chat template and will be moved later
                        {"role": "user", "content": content_list},
                    ]
                    new_input_ids = await self.loop.run_in_executor(
                        None,
                        lambda: self.tokenizer.apply_chat_template(
                            new_messages, add_generation_prompt=True, tokenize=True, **self.apply_chat_template_kwargs
                        ),
                    )
                    # remove the default system prompt, index the first <|im_end|> token
                    new_input_ids = new_input_ids[new_input_ids.index(self.im_end_token_id) + 1 :]
                    # the first token should be linebreak, concatenating im_end, the last token of old messages
                    assert new_input_ids[0] == self.linebreak_token_id
                    cur_input_ids = cur_input_ids + new_input_ids

                    # dedup will be conducted inside server_manager.generate
                    # cur_input_ids = _qwen2_5_vl_dedup_image_tokens(cur_input_ids, self.processor)
                    image_data.append(image_observation)

                    num_actions += 1

        assert len(cur_input_ids) == len(assistant_mask)
        # breakpoint()
        # print(f"Total number of turns: {total_num_turns}")
        assert total_num_turns <= state.environment.max_steps * 2

        return_prompt_and_response = True

        if return_prompt_and_response:

            prompt_ids = initial_prompt_ids
            response_ids = cur_input_ids[len(prompt_ids) :]
            assert sum(assistant_mask[: len(prompt_ids)]) == 0
            response_mask = assistant_mask[len(prompt_ids) :]
            multi_modal_data = {"image": image_data}

            assert len(prompt_ids) <= self.max_prompt_length
            assert len(response_ids) <= self.max_response_length

            output = AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                response_mask=response_mask,
                response_logprobs=None,
                multi_modal_data=multi_modal_data,
                num_turns=total_num_turns,
                metrics=metrics,
                extra_fields={"state": state},
            )
            return output

        else:

            output = AgentLoopOutput(
                prompt_ids=None,
                response_ids=None,
                response_mask=None,
                response_logprobs=None,
                multi_modal_data=multi_modal_data,
                num_turns=total_num_turns,
                metrics=metrics,
                sequence_ids=cur_input_ids,
                sequence_mask=assistant_mask,
                state=state,
            )
            return output


@ray.remote
class CursorLoopWorker(AgentLoopWorkerBase):
    """Agent loop worker takes a batch of messages and run each message in an agent loop."""

    def __init__(
        self, config: DictConfig, server_handles: list[ray.actor.ActorHandle], reward_router_address: str = None
    ):
        """Initialize agent loop manager.
        Args:
            config (DictConfig): YAML config.
            server_handles (List[ray.actor.ActorHandle]): OpenAI compatible LLM server actor handles.
            reward_router_address (str): reward router address.
        """
        super().__init__(config, server_handles, reward_router_address)

    @tqbridge()
    async def generate_sequences(self, batch: DataProto) -> DataProto:
        """Generate sequences from agent loop.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )

        # override sampling params for validation
        if batch.meta_info.get("validate", False):
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["temperature"] = config.val_kwargs.temperature

        # by default, we assume it's a single turn agent
        if "agent_name" not in batch.non_tensor_batch:
            default_agent_loop = config.agent.default_agent_loop
            batch.non_tensor_batch["agent_name"] = np.array([default_agent_loop] * len(batch), dtype=object)

        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(batch))

        trajectory_info = await get_trajectory_info(
            batch.meta_info.get("global_steps", -1), index.tolist(), batch.meta_info.get("validate", False)
        )
        # self.tokenizer = hf_tokenizer("/mnt/ceph_rbd/models/GUI-Cursor_resaved")
        self.tokenizer = hf_tokenizer("Qwen/Qwen2.5-VL-7B-Instruct")
        # self.tokenizer.chat_template = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n <think>{% endif %}"

        tasks = []
        for i in range(len(batch)):
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
            tasks.append(asyncio.create_task(self._run_agent_loop(sampling_params, trajectory_info[i], **kwargs)))

        outputs = await asyncio.gather(*tasks)

        if batch.meta_info.get("validate", False):
            output = self._val_postprocess(outputs)
        else:
            output = self._postprocess(outputs)
        return output

    def _val_postprocess(self, inputs: list[_InternalCursorLoopOutput]) -> DataProto:
        """Process the padded outputs from _run_agent_loop and combine them into a batch."""
        # Convert lists back to tensors and stack them to create a batch.
        batch = TensorDict(
            {"postprocess_chunk_idx": torch.tensor(range(len(inputs)), dtype=torch.int32)},
            batch_size=len(inputs),
        )

        scores = [input.reward_score for input in inputs]
        if all(score is not None for score in scores):
            rm_scores = torch.tensor(scores, dtype=torch.float32)
            rm_scores = rm_scores.unsqueeze(-1)
            batch["rm_scores"] = rm_scores

        non_tensor_batch = {
            "__num_turns__": np.array([input.num_turns for input in inputs], dtype=np.int32),
        }

        # add reward_extra_info to non_tensor_batch
        reward_extra_infos = [input.extra_fields.get("reward_extra_info", {}) for input in inputs]
        reward_extra_keys = list(reward_extra_infos[0].keys())
        for key in reward_extra_keys:
            non_tensor_batch[key] = np.array([info[key] for info in reward_extra_infos])

        # Add multi_modal_inputs to non_tensor_batch if any samples have them
        multi_modal_inputs_list = [input.multi_modal_inputs for input in inputs]
        if any(mmi is not None for mmi in multi_modal_inputs_list):
            non_tensor_batch["multi_modal_inputs"] = np.array(multi_modal_inputs_list, dtype=object)

        metrics = [input.metrics.model_dump() for input in inputs]
        # Collect extra fields from all inputs and convert them to np.ndarray
        extra_fields = {}
        all_keys = set(key for input_item in inputs for key in input_item.extra_fields)
        for key in all_keys:
            temp_arr = np.empty(len(inputs), dtype=object)
            temp_arr[:] = [input.extra_fields.get(key) for input in inputs]
            extra_fields[key] = temp_arr

        non_tensor_batch.update(extra_fields)
        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            meta_info={"metrics": metrics, "reward_extra_keys": reward_extra_keys},
        )

    async def _run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        **kwargs,
    ) -> _InternalAgentLoopOutput:
        with rollout_trace_attr(
            step=trajectory["step"],
            sample_index=trajectory["sample_index"],
            rollout_n=trajectory["rollout_n"],
            validate=trajectory["validate"],
            name="agent_loop",
        ):
            assert (
                agent_name in _agent_loop_registry
            ), f"Agent loop {agent_name} not registered, registered agent loops: {_agent_loop_registry.keys()}"

            is_validate = trajectory["validate"]
            agent_loop_config = _agent_loop_registry[agent_name]
            agent_loop = hydra.utils.instantiate(
                config=agent_loop_config,
                trainer_config=_DummyConfig(config=self.config),
                server_manager=self.server_manager,
                tokenizer=self.tokenizer,
                processor=self.processor,
            )
            output: AgentLoopOutput = await agent_loop.run(sampling_params, validate=is_validate, **kwargs)

            if is_validate:
                prompt_input_ids= None
                response_input_ids = None
                input_ids = None
                response_mask=None
                attention_mask=None
                response_logprobs=None
                multi_modal_inputs=None
                multi_modal_data=None
                position_ids=None
            else:
                self.tokenizer.padding_side = "left"
                prompt_output = self.tokenizer.pad(
                    {"input_ids": output.prompt_ids},
                    padding="max_length",
                    max_length=self.config.actor_rollout_ref.rollout.prompt_length,
                    return_tensors="pt",
                    return_attention_mask=True,
                )
                if prompt_output["input_ids"].dim() == 1:
                    prompt_output["input_ids"] = prompt_output["input_ids"].unsqueeze(0)
                    prompt_output["attention_mask"] = prompt_output["attention_mask"].unsqueeze(0)

                self.tokenizer.padding_side = "right"
                response_output = self.tokenizer.pad(
                    {"input_ids": output.response_ids},
                    padding="max_length",
                    max_length=self.config.actor_rollout_ref.rollout.response_length,
                    return_tensors="pt",
                    return_attention_mask=True,
                )
                if response_output["input_ids"].dim() == 1:
                    response_output["input_ids"] = response_output["input_ids"].unsqueeze(0)
                    response_output["attention_mask"] = response_output["attention_mask"].unsqueeze(0)

                response_mask_output = self.tokenizer.pad(
                    {"input_ids": output.response_mask},
                    padding="max_length",
                    max_length=self.config.actor_rollout_ref.rollout.response_length,
                    return_tensors="pt",
                    return_attention_mask=False,
                )
                if response_mask_output["input_ids"].dim() == 1:
                    response_mask_output["input_ids"] = response_mask_output["input_ids"].unsqueeze(0)

                response_logprobs = None
                if output.response_logprobs is not None:
                    pad_size = self.config.actor_rollout_ref.rollout.response_length - len(output.response_logprobs)
                    response_logprobs = torch.tensor(output.response_logprobs + [0.0] * pad_size).unsqueeze(0)

                response_mask = response_mask_output["input_ids"] * response_output["attention_mask"]
                attention_mask = torch.cat([prompt_output["attention_mask"], response_output["attention_mask"]], dim=1)
                input_ids = torch.cat([prompt_output["input_ids"], response_output["input_ids"]], dim=1)

                multi_modal_inputs = None
                if (
                    self.processor is not None
                    and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__
                ):
                    if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                        from verl.models.transformers.qwen3_vl import get_rope_index
                    else:
                        from verl.models.transformers.qwen2_vl import get_rope_index

                    images = output.multi_modal_data.get("image", None)
                    current_text = self.tokenizer.decode(input_ids.squeeze(0), skip_special_tokens=True)
                    multi_modal_inputs = self.processor(text=[current_text], images=images, return_tensors="pt")
                    multi_modal_inputs.pop("input_ids", None)
                    multi_modal_inputs.pop("attention_mask", None)

                    # We must use dict(multi_modal_inputs) to convert BatchFeature values to a new dict
                    # because np.array() only keeps the keys for BatchFeature.
                    multi_modal_inputs = dict(multi_modal_inputs)

                    image_grid_thw = multi_modal_inputs.get("image_grid_thw")
                    video_grid_thw = multi_modal_inputs.get("video_grid_thw")
                    second_per_grid_ts = multi_modal_inputs.get("second_per_grid_ts")

                    vision_position_ids = get_rope_index(
                        self.processor,
                        input_ids=input_ids.squeeze(0),
                        image_grid_thw=image_grid_thw,
                        video_grid_thw=video_grid_thw,
                        second_per_grid_ts=second_per_grid_ts,
                        attention_mask=attention_mask.squeeze(0),
                    ).unsqueeze(
                        0
                    )  # (1, 3, seq_len)

                    valid_mask = attention_mask[0].bool()
                    text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
                    text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
                    text_position_ids = text_position_ids.unsqueeze(0)
                    position_ids = torch.cat((text_position_ids, vision_position_ids), dim=1)  # (1, 4, seq_length)
                else:
                    position_ids = compute_position_id_with_mask(attention_mask)  # (1, seq_len)

                prompt_input_ids= prompt_output["input_ids"]
                response_input_ids = response_output["input_ids"]

            state: State = output.extra_fields.pop("state")
            result = await self.reward_manager_worker.compute_score.remote([state])

            reward_score = result["reward_score"]
            extra_fields = {
                "reward_extra_info": result["reward_extra_info"],
            }
            # output.extra_fields["reward_extra_info"] = result["reward_extra_info"]

            if trajectory["validate"]:
                msg_traj = convert_image_to_cursor_position(
                    state.message_history, state.position_history, release_image=True
                )
                extra_fields["message_trajectory"] = msg_traj
            else:
                if self.config.trainer.get("rollout_data_dir", None) is not None:
                    msg_traj = convert_image_to_cursor_position(
                        state.message_history, state.position_history, release_image=True
                    )
                    extra_fields["message_trajectory"] = msg_traj

            num_turns = output.num_turns
            metrics = output.metrics
            state.environment.screenshot.close()
            state.environment.cursor_image.close()

            del state, output

            return _InternalCursorLoopOutput(
                prompt_ids=prompt_input_ids,
                response_ids=response_input_ids,
                input_ids=input_ids,
                position_ids=position_ids,
                response_mask=response_mask,
                attention_mask=attention_mask,
                response_logprobs=response_logprobs,
                multi_modal_inputs=multi_modal_inputs,
                multi_modal_data=None,  # output.multi_modal_data,
                reward_score=reward_score,
                num_turns=num_turns,
                metrics=metrics,
                extra_fields=extra_fields,
            )


class CursorAgentLoopManager(AgentLoopManager):
    def __init__(self, config: DictConfig, worker_group: RayWorkerGroup = None, rm_wg: RayWorkerGroup = None):
        # Specify the agent loop worker class before calling super().__init__()
        self.agent_loop_workers_class = CursorLoopWorker
        super().__init__(config, worker_group=worker_group, rm_wg=rm_wg)

    def outside_wakeup(self):
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.wake_up()
        if self.reward_model_manager and self.config.reward_model.rollout.free_cache_engine:
            self.reward_model_manager.wake_up()
    
    def outside_sleep(self):
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.sleep()
        if self.reward_model_manager and self.config.reward_model.rollout.free_cache_engine:
            self.reward_model_manager.sleep()

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Split input batch and dispatch to agent loop workers.

        Args:
            prompts (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
        """

        # if self.config.actor_rollout_ref.rollout.free_cache_engine:
        #     self.wake_up()
        # if self.reward_model_manager and self.config.reward_model.rollout.free_cache_engine:
        #     self.reward_model_manager.wake_up()

        chunkes = prompts.chunk(len(self.agent_loop_workers))
        outputs = ray.get(
            [
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=True)
            ]
        )
        
        output = DataProto.concat(outputs)

        # if self.config.actor_rollout_ref.rollout.free_cache_engine:
        #     self.sleep()
        # if self.reward_model_manager and self.config.reward_model.rollout.free_cache_engine:
        #     self.reward_model_manager.sleep()

        # calculate performance metrics
        metrics = [output.meta_info.pop("metrics") for output in outputs]  # List[List[Dict[str, str]]]
        timing = self._performance_metrics(metrics, output)

        output.meta_info = {"timing": timing, **outputs[0].meta_info}

        del outputs, metrics, chunkes
        
        return output

    def _performance_metrics(self, metrics: list[list[dict[str, str]]], output: DataProto) -> dict[str, float]:
        timing = {}
        t_generate_sequences = np.array([metric["generate_sequences"] for chunk in metrics for metric in chunk])
        # t_tool_calls = np.array([metric["tool_calls"] for chunk in metrics for metric in chunk])
        timing["agent_loop/generate_sequences/min"] = t_generate_sequences.min()
        timing["agent_loop/generate_sequences/max"] = t_generate_sequences.max()
        timing["agent_loop/generate_sequences/mean"] = t_generate_sequences.mean()
        # timing["agent_loop/tool_calls/min"] = t_tool_calls.min()
        # timing["agent_loop/tool_calls/max"] = t_tool_calls.max()
        # timing["agent_loop/tool_calls/mean"] = t_tool_calls.mean()

        # batch sequence generation is bounded by the slowest sample
        # slowest = np.argmax(t_generate_sequences + t_tool_calls)
        if "attention_mask" in output.batch:
            slowest = np.argmax(t_generate_sequences)
            attention_mask = output.batch["attention_mask"][slowest]
            prompt_length = output.batch["prompts"].shape[1]
            timing["agent_loop/slowest/generate_sequences"] = t_generate_sequences[slowest]
            # timing["agent_loop/slowest/tool_calls"] = t_tool_calls[slowest]
            timing["agent_loop/slowest/prompt_length"] = attention_mask[:prompt_length].sum().item()
            timing["agent_loop/slowest/response_length"] = attention_mask[prompt_length:].sum().item()

        return timing
