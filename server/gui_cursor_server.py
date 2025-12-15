from copy import deepcopy
from vllm import LLM
from flask import Flask, request, jsonify
import base64
import io
import json
from PIL import Image
import numpy as np
import os
from datetime import datetime
from pathlib import Path
from cursor.move_loop import AsyncVLLMCursorMove, init_environment_and_state
import os
from typing import Optional
import asyncio
import threading
from cursor.env_and_state import MESSAGE_HISTORY_UPDATE_FN, Environment, State
from cursor.utils.extract_action import ACTION_EXTRACTOR_FN
from qwen_vl_utils import smart_resize
import math

model: Optional[AsyncVLLMCursorMove] = None
state_init_kwargs = dict()
event_loop: Optional[asyncio.AbstractEventLoop] = None
loop_thread: Optional[threading.Thread] = None

app = Flask(__name__)


def run_event_loop(loop):
    """Run the event loop in a separate thread."""
    asyncio.set_event_loop(loop)
    loop.run_forever()


def initialize_servers():
    global model, state_init_kwargs, event_loop, loop_thread
    if model is None:
        # Create and start event loop in a separate thread
        event_loop = asyncio.new_event_loop()
        loop_thread = threading.Thread(target=run_event_loop, args=(event_loop,), daemon=True)
        loop_thread.start()
        
        model_path = "/mnt/ceph_rbd/models/new_trained/GUI-Cursor-uitars-160steps"
        # model_path = "/mnt/ceph_rbd/models/GUI-Cursor_resaved"

        
        # Initialize the async model in the event loop
        future = asyncio.run_coroutine_threadsafe(
            initialize_model(model_path), event_loop
        )
        future.result()  # Wait for initialization to complete

        if "uitars" in model_path:
            state_init_kwargs.update(
                {
                    "system_prompt_format": open("cursor/prompts/uitars_no_system_prompt.txt", "r").read(),
                    "first_turn_prompt_format": open("cursor/prompts/uitars_query_prompt.txt", "r").read(),
                    "reply_prompt_format": "",
                    "action_extractor_fn": ACTION_EXTRACTOR_FN["uitars_action"],
                    "message_history_update_fn": MESSAGE_HISTORY_UPDATE_FN["qwen"],
                }
            )

        else:
            state_init_kwargs.update(
                {
                    "system_prompt_format": open("cursor/prompts/system_prompt.txt", "r").read(),
                    "first_turn_prompt_format": open("cursor/prompts/query_prompt.txt", "r").read(),
                    "reply_prompt_format": "",
                    "action_extractor_fn": ACTION_EXTRACTOR_FN["coord_answer_tag"],
                    "message_history_update_fn": MESSAGE_HISTORY_UPDATE_FN["qwen"],
                }
            )


async def initialize_model(model_path):
    """Initialize the model in the async event loop."""
    global model
    model = AsyncVLLMCursorMove(model_path=model_path, tp_size=1, batch_size=8)



# def translate_coord(pred_item, image, cursor_focus_size):
#     translated_position_history = []

#     with_crop = len(pred_item["position_history_global"]) > 0

#     if with_crop:
#         before_crop_positions = pred_item["position_history_global"][0]
#         after_crop_positions = pred_item["position_history"]
#     else:
#         before_crop_positions = pred_item["position_history"]
#         after_crop_positions = None

#     original_width, original_height = image.size
#     before_height, before_width = smart_resize(original_height, original_width, factor=28, min_pixels=2048, max_pixels=10240)
#     for before_pos in before_crop_positions:
#         x, y = before_pos
#         before_x_proportion = x / before_width
#         before_y_proportion = y / before_height
#         translated_x = int(before_x_proportion * original_width)
#         translated_y = int(before_y_proportion * original_height)
#         translated_position_history.append((translated_x, translated_y))

#     if after_crop_positions is not None:
#         if cursor_focus_size > 1.0:
#             cropped_image_pixels = cursor_focus_size
#             crop_width = int(math.sqrt((cropped_image_pixels * original_width) / original_height))
#             crop_height = int(math.sqrt((cropped_image_pixels * original_height) / original_width))
#         else:
#             crop_width = int(original_width * cursor_focus_size)
#             crop_height = int(original_height * cursor_focus_size)

#         after_height, after_width = smart_resize(crop_height, crop_width, min_pixels=2048, max_pixels=10240)

#         if pred_item["stop_reason"] == "target_reached":
#             after_crop_positions = after_crop_positions[:-1]

#         crop_left, crop_top = pred_item["focus_left_top_history"][0]
#         for after_pos in after_crop_positions:
#             x, y = after_pos
#             after_x_proportion = x / after_width
#             after_y_proportion = y / after_height
#             translated_x = int(after_x_proportion * crop_width) + crop_left
#             translated_y = int(after_y_proportion * crop_height) + crop_top
#             translated_position_history.append((translated_x, translated_y))
#     return translated_position_history, with_crop


async def async_cursor_trajectory(state: State):
    """Run a single cursor trajectory asynchronously until completion"""
    while state.step_idx < state.environment.max_steps:  # not state.is_finished and
        result = await model.async_predict_single(state)
        has_stopped = state.update_state(result["observation"], result["text_input"], result["response"])
        if has_stopped:
            break

    pred_results = state.save_state_predictions(save_to_disk=False, return_preds=True)

    # position_history, with_crop = translate_coord(
    #     pred_results, state.environment.original_screenshot, state.environment.cursor_focus_sizes
    # )

    return {
        "item_idx": state.item_idx,
        # "success": state.is_within_bbox(),
        # "num_steps": state.step_idx,
        # "position_history": position_history,
        # "with_crop": with_crop,
        "position_history": pred_results["position_history"],
        "position_prop_history": pred_results["position_prop_history"],
        "position_history_global": pred_results.get("position_history_global", []),
        "focus_left_top_history": pred_results.get("focus_left_top_history", []),
    }


def decode_base64_image(image_base64):
    """Decode base64 image string to PIL Image."""
    try:
        # Remove data URL prefix if present
        if image_base64.startswith("data:image"):
            image_base64 = image_base64.split(",")[1]

        # Decode base64 to bytes
        image_bytes = base64.b64decode(image_base64)

        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        return image
    except Exception as e:
        raise ValueError(f"Failed to decode image: {str(e)}")


@app.route("/gui_cursor_grounding", methods=["POST"])
def gui_cursor_grounding():
    """Handle GUI cursor grounding requests."""
    try:
        # Ensure servers are initialized
        if model is None:
            initialize_servers()

        # Parse JSON request
        data = request.get_json()
        print(f"receive idx: {data.get("idx", None)}")
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        # Extract required fields
        image_base64 = data.get("image")
        query = data.get("query")

        if not image_base64:
            return jsonify({"error": "Missing 'image' field"}), 400

        if not query:
            return jsonify({"error": "Missing 'query' field"}), 400

        # Decode image
        try:
            image = decode_base64_image(image_base64)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        # Predict coordinates

        # coordinates = predict_cursor_coordinates(image, query)
        # save image and query to server_log/gui_cursor_grounding/{timestamp}
        timestamp = datetime.now().strftime(r"%Y%m%d_%H%M%S")
        request_log_dir = Path(f"server/log/gui_cursor_grounding/{timestamp}")
        os.makedirs(request_log_dir, exist_ok=True)
        # image.save(request_log_dir / "image.png")
        with open(request_log_dir / "query.txt", "w") as f:
            f.write(query)

        # Run async function to get prediction
        async def predict_coordinate():
            example = {
                "image": image,
                "query": query,
                "bbox_proportions": None,
                "item_idx": 0,
            }
            env_kwargs = data.get("env_kwargs", {})
            state = init_environment_and_state(
                example=example,
                output_dir=str(request_log_dir),
                min_tokens_per_img=env_kwargs.get("min_tokens_per_img", 2048),
                cursor_focus_sizes=env_kwargs.get("cursor_focus_sizes", 1920 * 1080),
                focus_type=env_kwargs.get("focus_type", "within_image"),
                max_steps=env_kwargs.get("max_steps", 1),
                move_until_max_steps=env_kwargs.get("move_until_max_steps", False),
                **state_init_kwargs,
            )
            result = await async_cursor_trajectory(state)
            return result

        # Run the async function in the persistent event loop
        future = asyncio.run_coroutine_threadsafe(predict_coordinate(), event_loop)
        result = future.result()  # Wait for the result

        # Return response
        response = {"result": result, "status": "success"}
        print("predicted coordinate:", result["position_history"][-1])
        return jsonify(response), 200

    except Exception as e:
        print(f"fail to predict coordinate")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy"}), 200


if __name__ == "__main__":
    # Only initialize servers in the main process, not in Flask's reloader process
    # if not os.environ.get("WERKZEUG_RUN_MAIN"):
    #     print("Starting GUI Cursor Grounding Server...")
    #     print("Server will be available at: http://localhost:54301")
    #     print("Endpoint: POST /gui_cursor_grounding")
    #     print("Expected request format:")
    #     print("  {")
    #     print('    "image": "<base64_encoded_image>",')
    #     print('    "query": "<text_query>"')
    #     print("  }")
    #     print("\nPress Ctrl+C to stop the server")
    # else:
    #     # This is the reloader process, initialize servers here
    initialize_servers()

    # Run the Flask app
    app.run(host="0.0.0.0", port=54302, debug=False)
