"""
yuzhaouoe 06/10/2025

vllm should be imported before transformers / processor

sglang is faster when less CPUs are available

vllm is more stable all the time

"""

import re
import vllm
import os
import time
import gc
import asyncio
from pathlib import Path
from io import BytesIO
from PIL import Image
from typing import List
import json

from cursor.env_and_state import MESSAGE_HISTORY_UPDATE_FN, Environment, State
from cursor.prepare_data import load_iterable_dataset
from cursor.utils.extract_action import ACTION_EXTRACTOR_FN
from cursor.utils.log import get_logger


logger = get_logger(__name__)


class CursorMoveBase:
    def __init__(self, model_path):
        self.model_path = model_path

    def predict(self, *args, **kwargs):
        raise NotImplementedError


def get_logit_bias(processor, processor_name="Qwen2_5_VLProcessor"):
    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    bias_dict = {image_token_id: -100}

    if processor_name == "Qwen2_5_VLProcessor":
        bad_token = processor.tokenizer.encode(" addCriterion")[0]
        bias_dict[bad_token] = -100

    return bias_dict


def observe_get_raw_message_input(state: State, return_message_inputs: bool):
    observation, text_input, message_inputs = state.observe(return_message_inputs=return_message_inputs)
    if return_message_inputs:
        images = []
        for msg in message_inputs:
            content = msg["content"]
            if isinstance(content, str):
                continue
            if msg["role"] == "user":
                if isinstance(content, list):
                    for cur_content in content:
                        if cur_content["type"] == "image":
                            images.append(cur_content["image"])
            elif msg["role"] == "screenshot":
                images.append(content[0]["image"])
    else:
        images = None

    return {
        "observation": observation,
        "text_input": text_input,
        "item_idx": state.item_idx,
        "images": images,
        "message_inputs": message_inputs,
    }


def observe_get_message_input(processor, state: State):
    observation, text_input, message_inputs = state.observe()

    prompt = processor.apply_chat_template(message_inputs, tokenize=False, add_generation_prompt=True)
    multi_modal_data = []

    for msg in message_inputs:
        content = msg["content"]
        if isinstance(content, str):
            continue
        if msg["role"] == "user":
            if isinstance(content, list):
                for cur_content in content:
                    if cur_content["type"] == "image":
                        multi_modal_data.append(cur_content["image"])
        elif msg["role"] == "screenshot":
            multi_modal_data.append(content[0]["image"])
    inputs = {
        "prompt": prompt,
        "multi_modal_data": {"image": multi_modal_data},
        # "multi_modal_uuids": {"image": [f'{state.item_idx}_step{idx}' for idx in range(len(multi_modal_data))]},
    }
    return {
        "inputs": inputs,
        "observation": observation,
        "text_input": text_input,
        "item_idx": state.item_idx,
        "message_inputs": message_inputs,
    }


class AsyncVLLMCursorMove(CursorMoveBase):
    def __init__(self, model_path, tp_size=1, batch_size=None):
        super().__init__(model_path)
        from vllm import LLM, SamplingParams
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        from vllm.engine.arg_utils import AsyncEngineArgs
        from transformers import AutoProcessor
        import torch

        # Initialize async engine
        engine_args = AsyncEngineArgs(
            model=model_path,
            tensor_parallel_size=tp_size,
            max_model_len=32768,
            limit_mm_per_prompt={"image": 4, "video": 0, "audio": 0},
            max_num_seqs=batch_size,
            disable_log_stats=False,
            # disable_mm_preprocessor_cache=True,
            mm_processor_cache_gb=128,
        )

        #     @staticmethod
        # def make_async_mp_client(
        #     vllm_config: VllmConfig,
        #     executor_class: type[Executor],
        #     log_stats: bool,
        #     client_addresses: Optional[dict[str, str]] = None,
        #     client_count: int = 1,
        #     client_index: int = 0,
        # ) -> "MPClient":
        #     parallel_config = vllm_config.parallel_config
        #     client_args = (vllm_config, executor_class, log_stats,
        #                 client_addresses, client_count, client_index)
        #     if parallel_config.data_parallel_size > 1:
        #         if parallel_config.data_parallel_external_lb:
        #             # External load balancer - client per DP rank.
        #             return DPAsyncMPClient(*client_args)
        #         # Internal load balancer - client balances to all DP ranks.
        #         return DPLBAsyncMPClient(*client_args)
        #     return AsyncMPClient(*client_args)
        self.async_engine = AsyncLLMEngine.from_engine_args(engine_args)

        self.processor = AutoProcessor.from_pretrained(model_path)

        logit_bias: dict = get_logit_bias(self.processor)
        self.sampling_params = SamplingParams(**{"temperature": 0.0, "max_tokens": 512, "logit_bias": logit_bias})

    async def async_predict_single(self, state: State) -> dict:
        """Async prediction for a single state"""
        input_data = observe_get_message_input(self.processor, state)
        vllm_input = input_data["inputs"]

        # Generate unique request ID
        request_id = f"req_{state.item_idx}_{time.time()}"

        # Use async generation
        final_output = None
        async for request_output in self.async_engine.generate(vllm_input, self.sampling_params, request_id):
            final_output = request_output

        response = final_output.outputs[0].text

        return {
            "response": response,
            "item_idx": input_data["item_idx"],
            "observation": input_data["observation"],
            "text_input": input_data["text_input"],
            "inputs": input_data["inputs"],
            "message_inputs": input_data["message_inputs"],
        }

    async def async_batch_predict(self, states: List[State]) -> dict:
        """Async batch prediction for multiple states concurrently"""
        tasks = [self.async_predict_single(state) for state in states]
        results = await asyncio.gather(*tasks)

        return {
            "batch_responses": [r["response"] for r in results],
            "batch_item_idx": [r["item_idx"] for r in results],
            "batch_observation": [r["observation"] for r in results],
            "batch_text_input": [r["text_input"] for r in results],
            "batch_inputs": [r["inputs"] for r in results],
            "batch_message_inputs": [r["message_inputs"] for r in results],
        }


class VLLMCursorMove(CursorMoveBase):
    def __init__(self, model_path, tp_size=1, batch_size=None):
        super().__init__(model_path)
        from vllm import LLM, SamplingParams
        from transformers import AutoProcessor
        import torch

        self.model = LLM(
            model=model_path,
            tensor_parallel_size=tp_size,
            max_model_len=32768,
            limit_mm_per_prompt={"image": 4, "video": 0, "audio": 0},
            max_num_seqs=batch_size,
            disable_log_stats=False,
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.sampling_params = SamplingParams(**{"temperature": 0.0, "max_tokens": 512})

    def batch_predict(self, states: List[State]) -> dict:
        batch_inputs = [observe_get_message_input(self.processor, state) for state in states]
        vllm_inputs = [item["inputs"] for item in batch_inputs]
        vllm_batch_responses = self.model.generate(vllm_inputs, sampling_params=self.sampling_params)
        breakpoint()
        batch_responses = [output.outputs[0].text for output in vllm_batch_responses]
        return {
            "batch_responses": batch_responses,
            "batch_item_idx": [item["item_idx"] for item in batch_inputs],
            "batch_observation": [item["observation"] for item in batch_inputs],
            "batch_text_input": [item["text_input"] for item in batch_inputs],
            "batch_inputs": [item["inputs"] for item in batch_inputs],
            "batch_message_inputs": [item["message_inputs"] for item in batch_inputs],
        }


def qwen_message_history_update_fn(
    observation: Image, text_input: str, output_text: str, message_history: List[dict]
) -> List[dict]:
    # create a new list, does not change the original message_history
    content_list = [{"type": "image", "image": observation}]
    if text_input is not None and text_input != "":
        content_list.append({"type": "text", "text": text_input})
    message_history = message_history + [
        {
            "role": "user",
            "content": content_list,
        },
        {
            "role": "assistant",
            "content": output_text,
        },
    ]
    if output_text is None:
        message_history = message_history[:-1]

    return message_history


MESSAGE_HISTORY_UPDATE_FN = {
    "qwen": qwen_message_history_update_fn,
}


def init_environment_and_state(
    example,
    system_prompt_format,
    first_turn_prompt_format,
    reply_prompt_format,
    action_extractor_fn,
    message_history_update_fn,
    output_dir,
    cursor_image=None,
    ask_to_move_if_wrong=False,
    paint_trajectory=False,
    anchor_position="top-left",
    max_tokens_per_img=10240,
    min_tokens_per_img=2048,
    add_bounding_box=False,
    latest_screen_only=False,
    max_steps=4,
    save_obs=False,
    max_global_steps=None,
    cursor_focus_sizes=None,
    cursor_center_crop_size=None,
    global_steps=0,
    init_position=None,
    focus_type=None,
) -> State:
    if cursor_image is None:
        cursor_image = Image.open("cursor/resources/cursor-icon-cropped.png").convert("RGBA")
        # resize to 80%
        # cursor_size = cursor_image.size
        # cursor_image = cursor_image.resize((int(cursor_size[0] * 0.8), int(cursor_size[1] * 0.8)))

    environment = Environment(
        screenshot=example["image"],
        cursor_image=cursor_image,
        bbox_proportions=example["bbox_proportions"],
        query=example["query"],
        system_prompt_format=system_prompt_format,
        first_turn_prompt_format=first_turn_prompt_format,
        reply_prompt_format=reply_prompt_format,
        ask_to_move_if_wrong=ask_to_move_if_wrong,
        paint_trajectory=paint_trajectory,
        anchor_position=anchor_position,
        add_bounding_box=add_bounding_box,
        max_tokens_per_img=max_tokens_per_img,
        min_tokens_per_img=min_tokens_per_img,
        latest_screen_only=latest_screen_only,
        max_steps=max_steps,
        max_global_steps=max_global_steps,
        item_idx=example["item_idx"],
        action_extractor_fn=action_extractor_fn,
        message_history_update_fn=message_history_update_fn,
        cursor_focus_sizes=cursor_focus_sizes,
        cursor_center_crop_size=cursor_center_crop_size,
        focus_type=focus_type,
    )
    state = State(
        environment,
        save_obs,
        output_dir,
        global_steps=global_steps,
        init_position=init_position,
    )
    return state


def get_args():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--use_async", action="store_true", help="Use async inference")
    parser.add_argument("--model_path", type=str, default="/mnt/ceph_rbd/models/GUI-Cursor")
    parser.add_argument("--dataset_name", type=str, default="ScreenSpot-v2")
    parser.add_argument("--disable_ccf", action="store_true")
    parser.add_argument("--max_steps", type=int, default=4)
    # parser.add_argument("--disable_mm_preprocessor_cache", action="store_true")
    return parser.parse_args()


async def async_cursor_trajectory(state: State, model):
    """Run a single cursor trajectory asynchronously until completion"""
    trajectory_results = []

    while not state.is_finished and state.step_idx < state.environment.max_steps:
        # Get prediction for current state
        result = await model.async_predict_single(state)

        # Update state with the prediction
        has_stopped = state.update_state(result["observation"], result["text_input"], result["response"])

        trajectory_results.append(
            {
                "step": state.step_idx,
                "response": result["response"],
                "is_within_bbox": state.is_within_bbox(),
                "cursor_position": (state.cursor_x, state.cursor_y) if hasattr(state, "cursor_x") else None,
            }
        )

        if has_stopped:
            break

    # Save final state
    state.save_state_predictions(save_to_disk=True, return_preds=False, release_image=True)

    return {
        "item_idx": state.item_idx,
        "success": state.is_within_bbox(),
        "num_steps": state.step_idx,
        "trajectory": trajectory_results,
    }


async def async_main():
    """Async version of main function for concurrent trajectory processing"""
    debug = False

    if debug:
        model_path = "/mnt/ceph_rbd/models/Qwen/Qwen2.5-VL-3B-Instruct"
        # model_path = "/mnt/ceph_rbd/models/GUI-Cursor"
        dataset_name = "ScreenSpot-v2"
        batch_size = 2
        tp_size = 1
        exp_name = "async-debug"
        use_vllm = True
        use_async = True
        disable_ccf = False
        max_steps = 4
    else:
        args = get_args()
        batch_size = args.batch_size
        tp_size = args.tp_size
        use_async = args.use_async
        model_path = args.model_path
        if args.exp_name is not None:
            exp_name = args.exp_name
        else:
            async_suffix = "-async" if use_async else ""
            exp_name = f"exp{async_suffix}-bs{batch_size}-tp{tp_size}"
        use_vllm = args.use_vllm
        dataset_name = args.dataset_name
        disable_ccf = args.disable_ccf
        max_steps = args.max_steps

    iter_dataset = load_iterable_dataset(dataset_name)

    # Initialize async model
    if use_vllm:
        model = AsyncVLLMCursorMove(model_path=model_path, tp_size=tp_size, batch_size=batch_size)
    else:
        model = AsyncSGLangCursorMove(model_path=model_path, tp_size=tp_size, batch_size=batch_size)

    # Setup prompts and functions
    # system_prompt_format = open("cursor/prompts/system_prompt.txt", "r").read()
    system_prompt_format = open("cursor/prompts/uitars_system_prompt.txt", "r").read()
    first_turn_prompt_format = open("cursor/prompts/query_prompt.txt", "r").read()
    reply_prompt_format = ""
    action_extractor_fn = ACTION_EXTRACTOR_FN["coord_answer_tag"]
    message_history_update_fn = MESSAGE_HISTORY_UPDATE_FN["qwen"]
    output_dir = Path("outputs") / dataset_name / exp_name

    assert not os.path.exists(output_dir / "predictions.jsonl")

    os.makedirs(output_dir, exist_ok=True)

    # Track metrics
    completed_trajectories = []
    active_trajectories = []
    max_concurrent = batch_size

    eval_start_time = time.perf_counter()
    loop_idx = 0

    logger.info(f"Starting async evaluation with max_concurrent={max_concurrent}")

    while True:
        loop_start_time = time.perf_counter()

        # Fill up to max concurrent trajectories
        while len(active_trajectories) < max_concurrent:
            example = next(iter_dataset, None)
            if example is None:
                break

            cursor_focus_sizes = None
            if not disable_ccf:
                # if "data_type" in example and isinstance(example["data_type"], list):
                #     if "Icon" in example["data_type"]:
                #         cursor_focus_sizes = 0.5
                # elif "data_type" in example and isinstance(example["data_type"], str):
                #     if example["data_type"] == "icon":
                #         cursor_focus_sizes = 0.5
                # if cursor_focus_sizes is None:
                #     cursor_focus_sizes = 1920 * 1080
                cursor_focus_sizes = 0.5

            state = init_environment_and_state(
                example,
                system_prompt_format,
                first_turn_prompt_format,
                reply_prompt_format,
                action_extractor_fn,
                message_history_update_fn,
                output_dir,
                min_tokens_per_img=2048,
                cursor_focus_sizes=cursor_focus_sizes,
                focus_type="within_image",
                max_steps=max_steps,
            )

            # Start async trajectory
            task = asyncio.create_task(async_cursor_trajectory(state, model))
            active_trajectories.append(task)

        if not active_trajectories:
            logger.info("No more trajectories to process.")
            break

        # Wait for at least one trajectory to complete
        done, pending = await asyncio.wait(active_trajectories, return_when=asyncio.FIRST_COMPLETED)

        # Process completed trajectories
        for task in done:
            result = await task
            completed_trajectories.append(result)
            logger.info(
                f"Completed trajectory {result['item_idx']}: "
                f"success={result['success']}, steps={result['num_steps']}"
            )

        # Update active trajectories
        active_trajectories = list(pending)

        loop_end_time = time.perf_counter()
        loop_cost = loop_end_time - loop_start_time

        # Calculate current accuracy
        if completed_trajectories:
            successes = sum(1 for t in completed_trajectories if t["success"])
            accuracy = successes / len(completed_trajectories)
            logger.info(
                f"Loop-{loop_idx}: {len(completed_trajectories)} completed, "
                f"accuracy: {accuracy:.3%} ({successes}/{len(completed_trajectories)}), "
                f"active: {len(active_trajectories)}, time: {loop_cost:.2f}s"
            )

        loop_idx += 1

    # Wait for any remaining trajectories
    if active_trajectories:
        logger.info(f"Waiting for {len(active_trajectories)} remaining trajectories...")
        remaining_results = await asyncio.gather(*active_trajectories)
        completed_trajectories.extend(remaining_results)

    eval_end_time = time.perf_counter()
    total_time = eval_end_time - eval_start_time

    # Calculate final metrics
    successes = sum(1 for t in completed_trajectories if t["success"])
    total_completed = len(completed_trajectories)
    accuracy = successes / total_completed if total_completed > 0 else 0.0
    avg_steps = sum(t["num_steps"] for t in completed_trajectories) / total_completed if total_completed > 0 else 0.0

    logger.info(
        f"Async evaluation completed in {total_time:.2f} seconds\n"
        f"Total completed: {total_completed}\n"
        f"Accuracy: {accuracy:.3%} ({successes}/{total_completed})\n"
        f"Average steps: {avg_steps:.2f}\n"
        f"Throughput: {total_completed / total_time:.2f} trajectories/second"
    )

    # Save detailed results
    summary = {
        "total_completed_items": total_completed,
        "accuracy": accuracy,
        "successful_trajectories": successes,
        "average_steps": avg_steps,
        "total_evaluation_time": total_time,
        "throughput_per_second": total_completed / total_time,
        "max_concurrent": max_concurrent,
        "model_type": "vllm" if use_vllm else "sglang",
        "async_mode": True,
    }

    with open(output_dir / "summary.txt", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Results saved to {output_dir}")


def main():
    """Main function that dispatches to sync or async based on arguments"""
    args = get_args()
    use_async = getattr(args, "use_async", False)
    
    if use_async:
        # Run async version
        asyncio.run(async_main())
    else:
        # Run sync version
        sync_main()


def sync_main():
    """Synchronous version of main function"""
    dataset_name = "ScreenSpot-v2"
    debug = False
    if debug:
        model_path = "/mnt/ceph_rbd/models/Qwen/Qwen2.5-VL-3B-Instruct"
        # model_path = "/mnt/ceph_rbd/models/GUI-Cursor"
        batch_size = 8
        tp_size = 1
        exp_name = "debug"
        use_vllm = True
        use_async = False
    else:
        args = get_args()
        batch_size = args.batch_size
        tp_size = args.tp_size
        use_async = args.use_async
        model_path = args.model_path

        if args.exp_name is not None:
            exp_name = args.exp_name
        else:
            async_suffix = "-async" if use_async else ""
            exp_name = f"exp{async_suffix}-bs{batch_size}-tp{tp_size}"
        use_vllm = args.use_vllm

    iter_dataset = load_iterable_dataset(dataset_name)

    # Use sync models only
    if use_vllm:
        model = VLLMCursorMove(model_path=model_path, tp_size=tp_size, batch_size=batch_size)
    else:
        model = SGLangCursorMove(model_path=model_path, tp_size=tp_size, batch_size=batch_size)

    # model = None
    system_prompt_format = open("cursor/prompts/system_prompt.txt", "r").read()
    first_turn_prompt_format = open("cursor/prompts/query_prompt.txt", "r").read()
    reply_prompt_format = ""
    action_extractor_fn = ACTION_EXTRACTOR_FN["coord_answer_tag"]
    message_history_update_fn = MESSAGE_HISTORY_UPDATE_FN["qwen"]
    output_dir = Path("outputs") / dataset_name / exp_name
    os.makedirs(output_dir, exist_ok=True)
    cur_batch: List[State] = []

    correctness: List[bool] = []
    completed_ids: List[int] = []
    loop_time_log = []
    loop_idx = 0

    eval_start_time = time.perf_counter()
    while True:
        loop_start_time = time.perf_counter()
        while len(cur_batch) < batch_size:
            example = next(iter_dataset, None)
            if example is None:
                break
            state = init_environment_and_state(
                example,
                system_prompt_format,
                first_turn_prompt_format,
                reply_prompt_format,
                action_extractor_fn,
                message_history_update_fn,
                output_dir,
            )
            cur_batch.append(state)

        # Use sync batch prediction
        batch_outputs = model.batch_predict(cur_batch)

        cur_batch_size = len(cur_batch)
        if cur_batch_size == 0:
            logger.info("No more items to process.")
            break

        stopped_sids: list[int] = []
        for sidx in range(len(cur_batch)):
            item_idx = batch_outputs["batch_item_idx"][sidx]
            assert item_idx == cur_batch[sidx].item_idx
            observation = batch_outputs["batch_observation"][sidx]
            text_input = batch_outputs["batch_text_input"][sidx]
            output_text = batch_outputs["batch_responses"][sidx]

            has_stopped = cur_batch[sidx].update_state(observation, text_input, output_text)
            if has_stopped:
                stopped_sids.append(sidx)
                cur_batch[sidx].save_state_predictions(save_to_disk=True, return_preds=False, release_image=True)
                correctness.append(cur_batch[sidx].is_within_bbox())
        # pop stopped items
        cur_batch = [cur_batch[sidx] for sidx in range(len(cur_batch)) if sidx not in stopped_sids]
        completed_ids.extend(stopped_sids)
        # if len(stopped_sids) > 0:
        #     gc.collect()

        loop_end_time = time.perf_counter()
        loop_cost = loop_end_time - loop_start_time
        loop_time_log.append(loop_cost)
        logger.info(
            f"Loop-{loop_idx}: processed {cur_batch_size} items in {loop_cost:.2f} seconds, complete {len(completed_ids)}"
        )
        if len(correctness) > 0:
            logger.info(f"avg acc: {sum(correctness) / len(correctness):.3%} ({sum(correctness)}/{len(correctness)})")
        loop_idx += 1

    eval_end_time = time.perf_counter()
    logger.info(
        f"Evaluation completed in {eval_end_time - eval_start_time:.2f} seconds, "
        f"total completed items: {len(completed_ids)}, "
    )

    # save acc and time
    if len(correctness) > 0:
        acc = sum(correctness) / len(correctness)
    else:
        acc = 0.0

    with open(output_dir / "summary.txt", "w") as f:
        json.dump(
            {
                "total_completed_items": len(completed_ids),
                "accuracy": acc,
                "total_evaluation_time": eval_end_time - eval_start_time,
                "async_mode": False,
            },
            f,
        )


if __name__ == "__main__":
    main()


# def image_qa(s, image_file, question):
#     s += user(image(image_file) + question)
#     s += assistant(gen("answer", max_tokens=256))


# image_url = "https://github.com/sgl-project/sglang/blob/main/test/lang/example_image.png?raw=true"
# image_bytes, _ = load_image(image_url)
# state = image_qa(image_bytes, "What is in the image?")
# print_highlight(state["answer"])


# sgl.Engine(
#     # Core model parameters
#     model_path: str,                          # Required: Path to model
#     dtype: str = "auto",                      # Data type: "bfloat16", "float16", "auto"
#     tp_size: int = 1,                         # Tensor parallel size

#     # Memory and performance
#     mem_fraction_static: float = 0.65,        # GPU memory fraction for model weights
#     context_length: Optional[int] = None,     # Context length override
#     max_running_requests: int = 256,          # Max concurrent requests
#     max_total_tokens: Optional[int] = None,   # Max total tokens in memory pool

#     # Attention and compute backends
#     attention_backend: Optional[str] = None,  # "flashinfer", "triton", etc.
#     prefill_attention_backend: Optional[str] = None,
#     decode_attention_backend: Optional[str] = None,

#     # CUDA optimizations
#     disable_cuda_graph: bool = False,         # Disable CUDA graphs
#     cuda_graph_max_bs: int = 8,               # Max batch size for CUDA graphs

#     # Cache and memory management
#     disable_radix_cache: bool = False,        # Disable prefix caching
#     page_size: int = 16,                      # Page size for memory management
#     enable_memory_saver: bool = False,        # Enable memory saver mode

#     # Chunked prefill
#     chunked_prefill_size: Optional[int] = None,  # Chunked prefill size
#     enable_mixed_chunk: bool = True,          # Enable mixed chunking

#     # Quantization
#     quantization: Optional[str] = None,       # "fp8", "int4", "int8", etc.
#     torchao_config: Optional[str] = None,     # TorchAO quantization config

#     # Speculative decoding
#     speculative_algorithm: Optional[str] = None,      # "EAGLE", "EAGLE3"
#     speculative_draft_model_path: Optional[str] = None,
#     speculative_num_steps: Optional[int] = None,
#     speculative_eagle_topk: Optional[int] = None,
#     speculative_num_draft_tokens: Optional[int] = None,

#     # LoRA support
#     lora_paths: Optional[List[str]] = None,   # LoRA adapter paths
#     max_loras_per_batch: int = 4,             # Max LoRAs per batch
#     lora_backend: str = "triton",             # LoRA backend
#     max_lora_rank: Optional[int] = None,      # Max LoRA rank

#     # Distributed inference
#     dp_size: int = 1,                         # Data parallel size
#     ep_size: int = 1,                         # Expert parallel size (for MoE)
#     pp_size: int = 1,                         # Pipeline parallel size
#     base_gpu_id: int = 0,                     # Base GPU ID

#     # Model configuration
#     trust_remote_code: bool = True,           # Trust remote code
#     revision: Optional[str] = None,           # Model revision/branch
#     tokenizer_path: Optional[str] = None,     # Custom tokenizer path

#     # Multimodal
#     is_embedding: bool = False,               # Is embedding model
#     enable_multimodal: bool = True,           # Enable multimodal

#     # Compilation and optimization
#     enable_torch_compile: bool = False,       # Enable torch.compile

#     # Logging and debugging
#     log_level: str = "INFO",                  # Logging level
#     random_seed: int = 42,                    # Random seed

#     # Server specific (when used as server)
#     port: Optional[int] = None,               # Port number
#     host: str = "127.0.0.1",                  # Host address
# )


# 729.23 set max seq nums
