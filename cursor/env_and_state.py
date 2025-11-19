from copy import deepcopy
import uuid
from PIL import Image, ImageDraw
from typing import Optional, Tuple, List
import os
import json
from pathlib import Path
import shutil
from cursor.utils.image import (
    downscale_image,
    add_bbox_to_screenshot,
    bbox_coords_from_proportions,
    is_within_bbox,
    smart_resize_min_max_tokens_per_img,
    crop_image_at_coordinate,
    crop_image_with_padding,
)
import logging
import math
from qwen_vl_utils import smart_resize

logging.basicConfig(
    format="%(asctime)s - %(levelname)s %(name)s %(lineno)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.WARNING)



class Environment:
    def __init__(
        self,
        screenshot: Image.Image,
        cursor_image: Image.Image,
        bbox_proportions: Tuple[float, float, float, float],
        query: str,
        system_prompt_format: str,
        first_turn_prompt_format: str,
        reply_prompt_format: str,
        ask_to_move_if_wrong: bool = False,
        paint_trajectory: bool = False,
        anchor_position: str = "top-left",
        add_bounding_box: bool = False,
        max_tokens_per_img: Optional[int] = None,
        min_tokens_per_img: Optional[int] = None,
        latest_screen_only: bool = False,
        max_steps: Optional[int] = None,
        item_idx: Optional[int] = None,
        action_extractor_fn: Optional[callable] = None,
        message_history_update_fn: Optional[callable] = None,
        max_global_steps: Optional[int] = None,
        cursor_focus_sizes: Optional[int | float] = None,
        cursor_center_crop_size: Optional[int] = None,
        focus_type: Optional[str] = None,
        store_message_history: bool = True,
        store_tokenized_ids_and_mm_inputs: bool = False,
    ):
        # if max_tokens_per_img is not None:
        if cursor_focus_sizes is not None or cursor_center_crop_size is not None:
            self.original_screenshot = screenshot.copy()
        screenshot = smart_resize_min_max_tokens_per_img(
            image=screenshot, min_tokens_per_img=min_tokens_per_img, max_tokens_per_img=max_tokens_per_img
        )

        width, height = screenshot.size

        if bbox_proportions is not None:
            bbox_coords = bbox_coords_from_proportions(width, height, bbox_proportions)

            if add_bounding_box:
                screenshot = add_bbox_to_screenshot(screenshot, bbox_coords)
            target_center = (
                int((bbox_proportions[0] + bbox_proportions[2]) * width / 2),
                int((bbox_proportions[1] + bbox_proportions[3]) * height / 2),
            )
        else:
            self.bbox_coords = None
            self.target_center = None
            self.bbox_proportions = None

        self.bbox_proportions = bbox_proportions

        self.screenshot = screenshot
        self.cursor_image = cursor_image
        self.anchor_position = anchor_position
        self.add_bounding_box = add_bounding_box
        self.max_tokens_per_img = max_tokens_per_img
        self.min_tokens_per_img = min_tokens_per_img
        self.query = query
        self.system_prompt_format = system_prompt_format
        self.first_turn_prompt_format = first_turn_prompt_format
        self.reply_prompt_format = reply_prompt_format
        self.bbox_coords = bbox_coords
        self.target_center = target_center
        self.latest_screen_only = latest_screen_only
        self.max_steps = max_steps
        self.max_global_steps = max_global_steps
        self.action_extractor_fn = action_extractor_fn
        self.message_history_update_fn = message_history_update_fn
        # debug
        self.ask_to_move_if_wrong = ask_to_move_if_wrong
        self.paint_trajectory = paint_trajectory
        self.item_idx = item_idx

        self.min_x_pos = 0  # cursor_w // 2
        self.min_y_pos = 0  # cursor_h // 2
        self.max_x_pos = self.screenshot.size[0] - 5
        self.max_y_pos = self.screenshot.size[1] - 5
        # cursor_w, cursor_h = self.cursor_image.size
        # self.max_x_pos = self.screenshot.size[0] - cursor_w // 2
        # self.max_y_pos = self.screenshot.size[1] - cursor_h // 2
        # self.min_x_pos = cursor_w // 2
        # self.min_y_pos = cursor_h // 2
        self.cursor_focus_sizes = cursor_focus_sizes
        if focus_type is None:
            self.focus_type = os.environ.get("FOCUS_TYPE", None)
        else:
            self.focus_type = focus_type
        self.cursor_center_crop_size = cursor_center_crop_size

        self.store_message_history = store_message_history
        self.store_tokenized_ids_and_mm_inputs = store_tokenized_ids_and_mm_inputs

    def __repr__(self):
        return f"Environment(name={self.name})"


class State:
    def __init__(
        self,
        environment: Environment,
        save_obs: bool,
        output_dir: Path | str,
        global_steps: int = 0,
        init_position: Optional[Tuple[int, int]] = None,
    ):
        self.uuid = str(uuid.uuid4())

        width, height = environment.screenshot.size

        if init_position is not None:
            init_cursor_x, init_cursor_y = init_position
        else:
            init_cursor_x, init_cursor_y = width // 2, height // 2

        self.init_position = (init_cursor_x, init_cursor_y)

        self.cursor_x = init_cursor_x
        self.cursor_y = init_cursor_y
        self.position_history = [(init_cursor_x, init_cursor_y)]
        self.is_init = True
        self.is_finished = False
        self.step_idx = 0
        self.at_screen_edge = False
        self.max_steps_reached = False
        self.success = False
        self.has_stopped = False

        system_prompt = environment.system_prompt_format.format(
            screen_width=environment.screenshot.size[0],
            screen_height=environment.screenshot.size[1],
            cursor_x=init_cursor_x,
            cursor_y=init_cursor_y,
        )

        if environment.store_message_history:
            self.message_history = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
            ]
        else:
            self.message_history = None
        
        self.environment: Environment = environment
        self.item_idx = environment.item_idx
        self.save_obs = save_obs
        if self.save_obs:
            if not isinstance(output_dir, Path):
                output_dir = Path(output_dir)
            self.output_dir = output_dir
            self.obs_save_dir = output_dir / "obs" / f"{self.item_idx:06d}-{self.uuid}"
            if os.path.exists(self.obs_save_dir):
                shutil.rmtree(self.obs_save_dir)
            os.makedirs(self.obs_save_dir, exist_ok=True)
        else:
            self.obs_save_dir = None
            self.prediction_save_path = None
            self.message_history_save_path = None

        self.output_dir = output_dir
        if self.output_dir is not None:
            if not isinstance(output_dir, Path):
                output_dir = Path(output_dir)
            self.prediction_save_path = output_dir / "predictions.jsonl"
            self.message_history_save_path = output_dir / "message_history.jsonl"

        self.global_steps = global_steps
        self.stop_reason = None
        self.reward_scores: dict = None
        self.within_bbox_history = [self.is_within_bbox()]

        if self.environment.cursor_focus_sizes is not None:
            self.position_history_global = []
            self.message_history_global = []
            self.focus_left_top_history = []
        else:
            self.position_history_global, self.message_history_global = None, None
            self.focus_left_top_history = None

        if self.environment.store_tokenized_ids_and_mm_inputs:
            self.tokenized_ids = []
            self.multi_modal_inputs = {"image": []}
        else:
            self.tokenized_ids = None
            self.multi_modal_inputs = None

    @classmethod
    def from_state(cls, state: "State"):
        """
        Create a new State from an existing one,
        New output directory using uuid4
        Share the same environment object
        """
        new_state: State = cls(
            environment=state.environment,
            save_obs=state.save_obs,
            output_dir=state.output_dir,
            global_steps=state.global_steps,
            init_position=state.init_position,
        )
        new_state.cursor_x = state.cursor_x
        new_state.cursor_y = state.cursor_y
        new_state.position_history = deepcopy(state.position_history)
        new_state.within_bbox_history = deepcopy(state.within_bbox_history)
        new_state.is_init = state.is_init
        new_state.is_finished = state.is_finished
        new_state.step_idx = state.step_idx
        new_state.at_screen_edge = state.at_screen_edge
        new_state.max_steps_reached = state.max_steps_reached
        new_state.success = state.success
        new_state.has_stopped = state.has_stopped
        new_state.message_history = deepcopy(state.message_history)
        new_state.global_steps = state.global_steps
        new_state.stop_reason = state.stop_reason
        new_state.reward_scores = state.reward_scores
        return new_state

    def _extract_action_and_update_cursor(self, observation, text_input, output_text):
        logger.debug(f"step {self.step_idx} - llm response: {output_text}")
        extractor = self.environment.action_extractor_fn

        message_history_update_fn = self.environment.message_history_update_fn
        self.message_history = message_history_update_fn(
            observation=observation,
            text_input=text_input,
            output_text=output_text,
            message_history=self.message_history,
        )

        action, action_x, action_y = extractor(output_text)

        if action is None:
            raise ValueError(f"step {self.step_idx} - extract action failed from: {output_text}")

        # if successful, update the state
        self.is_init = False
        self.step_idx += 1
        self.global_steps += 1
        width, height = self.environment.screenshot.size

        if action == "TARGET_REACHED":
            self.is_finished = True
        elif action == "MOVE":
            dx, dy = action_x, action_y
            self.cursor_x += dx
            self.cursor_y += dy

        elif action == "GOTO" or action == "MOVE_TO":
            self.cursor_x = action_x
            self.cursor_y = action_y
        else:
            raise NotImplementedError("action not implemented: {action}")

        self.cursor_x = min(
            max(self.cursor_x, self.environment.min_x_pos),
            self.environment.max_x_pos,
        )
        self.cursor_y = min(
            max(self.cursor_y, self.environment.min_y_pos),
            self.environment.max_y_pos,
        )
        self.position_history.append((self.cursor_x, self.cursor_y))
        self.within_bbox_history.append(self.is_within_bbox())

        # if (
        #     self.cursor_x == 0
        #     or self.cursor_x == width
        #     or self.cursor_y == 0
        #     or self.cursor_y == height
        # ):
        #     self.at_screen_edge = True

        if (
            self.cursor_x <= self.environment.min_x_pos
            or self.cursor_x >= self.environment.max_x_pos
            or self.cursor_y <= self.environment.min_y_pos
            or self.cursor_y >= self.environment.max_y_pos
        ):
            self.at_screen_edge = True
        else:
            self.at_screen_edge = False

        if self.step_idx >= self.environment.max_steps:
            self.max_steps_reached = True
        else:
            self.max_steps_reached = False

        logger.debug(
            f"step {self.step_idx} - action = {action}({action_x}, {action_y}),"
            f"cursor moved from {self.position_history[-2]} to {self.position_history[-1]}, "
            f"is_within_bbox = {self.within_bbox_history[-1]}"
        )

        if self.step_idx == 1:
            focus_done = self.focus_to_cursor()
            if focus_done:
                print(f"Focus to the cursor area, {self.focus_left_top_history[-1]}")

    def focus_to_cursor(self):
        if not (self.environment.cursor_focus_sizes is not None and self.step_idx == 1):
            return False

        org_width, org_height = self.environment.original_screenshot.size

        if org_width * org_height <= 2073600:  # 2073600 1920*1080, do not apply ccf when processing low resolution
            return False

        if org_height > org_width:  # do not applied for mobile images
            return False

        width, height = self.environment.screenshot.size
        cursor_x_ratio = self.cursor_x / width
        cursor_y_ratio = self.cursor_y / height
        org_bbox_coords = bbox_coords_from_proportions(org_width, org_height, self.environment.bbox_proportions)

        # focus to one monitor
        if (
            (org_width == 3840 and org_height == 1080) or (org_width == 5120 and org_height == 1440)
        ) and cursor_x_ratio != 0.5:
            if cursor_x_ratio < 0.5:
                top, left = 0, 0
                print(f"focus to left monitor")
            else:
                top, left = 0, org_width // 2
                print(f"focus to right monitor")
            # (left, top, right, bottom)
            cropped_image = self.environment.original_screenshot.crop(
                (left, top, left + org_width // 2, top + org_height)
            )
            crop_width, crop_height = cropped_image.size
        else:
            cursor_org_x = int(org_width * cursor_x_ratio)
            cursor_org_y = int(org_height * cursor_y_ratio)
            if self.environment.cursor_focus_sizes > 1.0:
                cropped_image_pixels = self.environment.cursor_focus_sizes
                crop_width = int(math.sqrt((cropped_image_pixels * org_width) / org_height))
                crop_height = int(math.sqrt((cropped_image_pixels * org_height) / org_width))
            else:
                cropped_factor = self.environment.cursor_focus_sizes
                crop_width = int(org_width * cropped_factor)
                crop_height = int(org_height * cropped_factor)

            if crop_width >= org_width or crop_height >= org_height:
                return False

            if self.environment.focus_type == "within_image":
                cropped_image, left, top = crop_image_at_coordinate(
                    self.environment.original_screenshot, cursor_org_x, cursor_org_y, crop_width, crop_height
                )
            elif self.environment.focus_type == "pad_image":
                cropped_image, left, top = crop_image_with_padding(
                    self.environment.original_screenshot, cursor_org_x, cursor_org_y, crop_width, crop_height
                )
            else:
                raise ValueError
        # print(f"env_size: {self.environment.screenshot.size}")
        # print(f"cursor_x, cursor_y: {self.cursor_x}, {self.cursor_y}")
        # print(f"org_size: {self.environment.original_screenshot.size}, cursor_org_x_y = {cursor_org_x}, {cursor_org_y}")
        print(
            f"Focus, original: {org_width}x{org_height}, left-top: ({left}, {top}), crop to {crop_width}x{crop_height}"
        )

        cropped_bbox_coords = [
            org_bbox_coords[0] - left,
            org_bbox_coords[1] - top,
            org_bbox_coords[2] - left,
            org_bbox_coords[3] - top,
        ]
        cropped_bbox_proportions = [
            cropped_bbox_coords[0] / crop_width,
            cropped_bbox_coords[1] / crop_height,
            cropped_bbox_coords[2] / crop_width,
            cropped_bbox_coords[3] / crop_height,
        ]

        self.focus_left_top_history.append([left, top])
        environment = Environment(
            screenshot=cropped_image,
            cursor_image=self.environment.cursor_image,
            bbox_proportions=cropped_bbox_proportions,
            query=self.environment.query,
            system_prompt_format=self.environment.system_prompt_format,
            first_turn_prompt_format=self.environment.first_turn_prompt_format,
            reply_prompt_format=self.environment.reply_prompt_format,
            ask_to_move_if_wrong=self.environment.ask_to_move_if_wrong,
            paint_trajectory=self.environment.paint_trajectory,
            anchor_position=self.environment.anchor_position,
            add_bounding_box=self.environment.add_bounding_box,
            max_tokens_per_img=self.environment.max_tokens_per_img,
            min_tokens_per_img=self.environment.min_tokens_per_img,
            latest_screen_only=self.environment.latest_screen_only,
            max_steps=self.environment.max_steps,
            item_idx=self.environment.item_idx,
            action_extractor_fn=self.environment.action_extractor_fn,
            message_history_update_fn=self.environment.message_history_update_fn,
            max_global_steps=self.environment.max_global_steps,
            cursor_focus_sizes=None,
        )
        self.environment = environment
        self.position_history_global = [self.position_history]
        self.message_history_global = [
            convert_image_to_cursor_position(self.message_history, self.position_history, release_image=False)
        ]
        # reset to center
        # shift a distance to avoid the centered key information being obscured by the cursor
        center_shift = (crop_width // 2 - 100, crop_height // 2 - 100)
        self.reset(reset_to=center_shift)
        # reset to the cursor position in the cropped image
        # new_cursor_x = self.cursor_x - left
        # new_cursor_y = self.cursor_y - top
        # self.reset(reset_to=(new_cursor_x, new_cursor_y))
        return True

    def update_state(self, observation, text_input, output_text) -> bool:
        stop_reason = None
        try:
            self._extract_action_and_update_cursor(observation, text_input, output_text)
        except Exception as e:
            import traceback

            logger.error(traceback.format_exc())
            logger.error(f"Error extracting action for item {self.item_idx}")
            stop_reason = "extract_action_failed"

        if self.is_finished:
            assert stop_reason is None
            stop_reason = "target_reached"
        elif self.max_steps_reached:
            assert stop_reason is None
            max_global_steps = self.environment.max_global_steps
            if max_global_steps is None or (max_global_steps is not None and self.global_steps >= max_global_steps):
                stop_reason = "max_steps_reached"
            else:
                stop_reason = "max_steps_reached_but_continue"

        if stop_reason is not None:
            # if self.save_obs:
            #     self.save_state_predictions(save_to_disk=True, return_preds=False, release_image=False)
            if stop_reason == "max_steps_reached_but_continue":
                self.reset()

        if stop_reason is None or stop_reason == "max_steps_reached_but_continue":
            self.has_stopped = False
        else:
            self.has_stopped = True
            self.stop_reason = stop_reason

        return self.has_stopped

    def save_state_predictions(self, save_to_disk=False, return_preds=False, release_image=True, disk_save_dir=None):
        if disk_save_dir is not None:
            assert save_to_disk
            if not isinstance(disk_save_dir, Path):
                disk_save_dir = Path(disk_save_dir)
            os.makedirs(disk_save_dir, exist_ok=True)
            prediction_save_path = disk_save_dir / "predictions.jsonl"
            message_history_save_path = disk_save_dir / "message_history.jsonl"
        else:
            prediction_save_path = self.prediction_save_path
            message_history_save_path = self.message_history_save_path

        if self.has_stopped is False:
            self.success = False
            self.stop_reason = None
        else:
            self.success = (self.stop_reason == "target_reached") and self.is_within_bbox()
        preds = {
            "item_idx": self.item_idx,
            "success": self.success,
            "is_within_bbox": self.is_within_bbox(),
            "final_cursor_position": (self.cursor_x, self.cursor_y),
            "position_history": self.position_history,
            "within_bbox_history": self.within_bbox_history,
            "global_steps": self.global_steps,
            "stop_reason": self.stop_reason,
        }
        if self.position_history_global is not None:
            preds["position_history_global"] = self.position_history_global
            preds["focus_left_top_history"] = self.focus_left_top_history
            preds["cursor_focus_sizes"] = self.environment.cursor_focus_sizes

        message_history_to_save = convert_image_to_cursor_position(
            self.message_history, self.position_history, release_image=release_image
        )
        message_history_to_save = {
            "item_idx": self.item_idx,
            "global_steps": self.global_steps,
            "message_history": message_history_to_save,
        }
        if self.message_history_global is not None:
            message_history_to_save["message_history_global"] = self.message_history_global

        if save_to_disk:
            with open(prediction_save_path, "a") as f:
                f.write(json.dumps(preds) + "\n")
            if self.save_obs and self.stop_reason != "max_steps_reached_but_continue":
                final_obs = self.get_screenshot_observation(add_bbox=True)
                final_obs.save(self.obs_save_dir / "final_obs.png")

            with open(message_history_save_path, "a") as f:
                f.write(
                    json.dumps(
                        message_history_to_save,
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            logger.info(f"save item-{self.item_idx} global_steps-{self.global_steps}")

        if return_preds:
            return preds

    def observe(self, return_message_inputs=True) -> Tuple[Image.Image, str, List[dict]]:
        environment = self.environment

        observation = self.get_screenshot_observation(add_bbox=environment.add_bounding_box)

        if self.save_obs:
            file_name = f"{self.global_steps:02d}_{self.step_idx:02d}.png"
            observation.save(self.obs_save_dir / file_name)

        if self.is_init:
            text_input = environment.first_turn_prompt_format.format(query=environment.query)
        elif self.is_finished:
            if not environment.ask_to_move_if_wrong:
                text_input = None
            elif environment.ask_to_move_if_wrong and not self.is_within_bbox():
                text_input = "The cursor is not at the target position based on the latest screenshot. Please continue to move the cursor."
                self.is_finished = False
            else:
                text_input = None
        # elif state.is_finished:
        #     text_input = None
        else:
            text_input = environment.reply_prompt_format.format(
                cursor_x=self.cursor_x,
                cursor_y=self.cursor_y,
            )
        if self.at_screen_edge:
            at_right, at_bottom = (
                self.cursor_x >= self.environment.max_x_pos,
                self.cursor_y >= self.environment.max_y_pos,
            )
            if at_right and at_bottom:
                text_input += "\n\nThe cursor is at the bottom-right corner, which may not be visible."
            elif at_right:
                text_input += "\n\nThe cursor is at the right edge, which may not be visible."
            elif at_bottom:
                text_input += "\n\nThe cursor is at the bottom edge, which may not be visible."

            text_input = text_input.strip()

        if return_message_inputs:
            # get message_inputs in observe() does not change the state's message_history
            # we need to update the state's message_history in update_state()
            message_history_update_fn = self.environment.message_history_update_fn
            message_inputs = message_history_update_fn(
                observation=observation,
                text_input=text_input,
                message_history=self.message_history,
                output_text=None,  # No output text at this point
            )
            assert message_inputs != self.message_history, "they are different objects"

            if self.environment.latest_screen_only:
                message_inputs = keep_last_n_images(message_inputs, n=1)
        else:
            message_inputs = None
        return observation, text_input, message_inputs

    def is_within_bbox(self) -> bool:
        if self.environment.bbox_coords is None:
            return False
        return is_within_bbox(self.cursor_x, self.cursor_y, self.environment.bbox_coords)

    def get_screenshot_observation(self, add_bbox: bool = False):
        environment = self.environment
        observation = environment.screenshot.copy()
        if environment.anchor_position == "top-left":
            observation.paste(
                environment.cursor_image,
                (self.cursor_x, self.cursor_y),
                environment.cursor_image,
            )
        elif environment.anchor_position == "center":
            cursor_width, cursor_height = environment.cursor_image.size
            observation.paste(
                environment.cursor_image,
                (self.cursor_x - cursor_width // 2, self.cursor_y - cursor_height // 2),
                environment.cursor_image,
            )
        else:
            raise NotImplementedError

        if add_bbox:
            observation = add_bbox_to_screenshot(observation, environment.bbox_coords)

        if self.step_idx > 0:
            size = self.environment.cursor_center_crop_size
            if size is not None and size > 0:
                org_width, org_height = self.environment.original_screenshot.size
                width, height = self.environment.screenshot.size
                cursor_x_ratio = self.cursor_x / width
                cursor_y_ratio = self.cursor_y / height
                cursor_org_x = int(org_width * cursor_x_ratio)
                cursor_org_y = int(org_height * cursor_y_ratio)

                cropped_observation = self.environment.original_screenshot.copy()
                cropped_observation, _, _ = crop_image_at_coordinate(
                    cropped_observation, cursor_org_x, cursor_org_y, size, size
                )
                cropped_observation = smart_resize_min_max_tokens_per_img(
                    image=cropped_observation,
                    min_tokens_per_img=128,
                    max_tokens_per_img=128,
                )
                cropped_observation.paste(
                    environment.cursor_image,
                    (cropped_observation.width // 2, cropped_observation.height // 2),
                    environment.cursor_image,
                )
                observation = cropped_observation
                print(f"Use cursor-centered cropped observation: {observation.size}")
        return observation

    def reset(self, reset_to: Optional[tuple[int, int]] = None):
        assert self.has_stopped is False
        environment = self.environment
        # if reset_to == "center":
        #     width, height = environment.screenshot.size
        #     init_cursor_x, init_cursor_y = width // 2, height // 2
        #     self.cursor_x, self.cursor_y = init_cursor_x, init_cursor_y
        # elif reset_to == "left-top":
        #     self.cursor_x, self.cursor_y = 5, 5
        #     init_cursor_x, init_cursor_y = self.cursor_x, self.cursor_y
        if reset_to is not None:
            self.cursor_x, self.cursor_y = reset_to
            init_cursor_x, init_cursor_y = self.cursor_x, self.cursor_y
        else:
            init_cursor_x, init_cursor_y = self.cursor_x, self.cursor_y
        self.position_history = [(init_cursor_x, init_cursor_y)]
        if self.at_screen_edge:
            self.cursor_x = min(
                max(self.cursor_x, self.environment.min_x_pos),
                self.environment.max_x_pos,
            )
            self.cursor_y = min(
                max(self.cursor_y, self.environment.min_y_pos),
                self.environment.max_y_pos,
            )
        self.is_init = True
        self.is_finished = False
        self.step_idx = 0
        self.max_steps_reached = False

        system_prompt = environment.system_prompt_format.format(
            screen_width=environment.screenshot.size[0],
            screen_height=environment.screenshot.size[1],
            cursor_x=init_cursor_x,
            cursor_y=init_cursor_y,
        )
        del self.message_history
        self.message_history = [
            {
                "role": "system",
                "content": system_prompt,
            },
        ]


def keep_last_n_images(message_history, n=1):
    num_images = 0
    new_messages = []
    for msg in reversed(message_history):
        content = msg["content"]
        if msg["role"] == "screenshot":
            num_images += 1
            if num_images <= n:
                new_messages.append(msg)
        elif isinstance(content, list):
            new_content = []
            for cur_content in content:
                assert isinstance(cur_content, dict)
                if cur_content["type"] in ["image", "image_url", "input_image"]:
                    num_images += 1
                    if num_images <= n:
                        new_content.append(cur_content)
                else:
                    new_content.append(cur_content)
            new_messages.append(
                {
                    "role": msg["role"],
                    "content": new_content,
                }
            )
        elif isinstance(content, dict):
            if content["type"] in ["image", "image_url", "input_image"]:
                num_images += 1
                if num_images <= n:
                    new_messages.append(msg)
            else:
                new_messages.append(msg)
        else:
            assert isinstance(content, str)
            new_messages.append(msg)
    new_messages.reverse()
    return new_messages


def convert_image_to_cursor_position(message_history, position_history, release_image=True):
    if not release_image:
        # do not release image, keep the original message_history
        message_history = deepcopy(message_history)
    image_idx = 0
    images_to_close = []  # Collect PIL images to close
    for msg_idx in range(len(message_history)):
        msg = message_history[msg_idx]
        content = msg["content"]
        if isinstance(content, list):
            for c_idx in range(len(content)):
                cur_content = content[c_idx]
                if cur_content["type"] in ["image", "image_url", "input_image"]:
                    keys = list(cur_content.keys())
                    keys.remove("type")
                    assert len(keys) == 1
                    image_key = keys[0]
                    # Collect PIL image for cleanup if release_image is True
                    if release_image and hasattr(cur_content[image_key], 'close'):
                        images_to_close.append(cur_content[image_key])
                    cursor_position = position_history[image_idx]
                    cur_content[image_key] = cursor_position
                    image_idx += 1
        elif isinstance(content, dict):
            if msg["content"]["type"] in ["image", "image_url", "input_image"]:
                cursor_position = position_history[image_idx]
                keys = list(msg["content"].keys())
                keys.remove("type")
                assert len(keys) == 1
                image_key = keys[0]
                # Collect PIL image for cleanup if release_image is True
                if release_image and hasattr(msg["content"][image_key], 'close'):
                    images_to_close.append(msg["content"][image_key])
                msg["content"][image_key] = cursor_position
                image_idx += 1
        else:
            assert isinstance(content, str)
    
    # Close all PIL images to release memory
    if release_image:
        for img in images_to_close:
            try:
                img.close()
            except:
                pass
    
    # assert image_idx == len(position_history)
    return message_history


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


def yuzhao_message_history_update_fn(
    observation: Image, text_input: str, output_text: str, message_history: List[dict]
) -> List[dict]:
    # create a new list, does not change the original message_history
    message_history = message_history + [
        {
            "role": "screenshot",
            "content": [
                {"type": "image", "image": observation},
            ],
        },
    ]
    if text_input is not None and text_input != "":
        message_history += [
            {
                "role": "user",
                "content": text_input,
            }
        ]
    if output_text is not None and output_text != "":
        message_history += [
            {
                "role": "assistant",
                "content": output_text,
            }
        ]
    return message_history


MESSAGE_HISTORY_UPDATE_FN = {
    "qwen": qwen_message_history_update_fn,
    "yuzhao": yuzhao_message_history_update_fn,
}