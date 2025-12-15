from ast import Not
from email.mime import image
import enum
from turtle import position
from PIL import Image, ImageDraw, ImageFont
import os
import json
from cursor.prepare_data import load_iterable_dataset
from cursor.utils.image import (
    downscale_image,
    add_bbox_to_screenshot,
    bbox_coords_from_proportions,
    is_within_bbox,
    smart_resize_min_max_tokens_per_img,
    crop_image_at_coordinate,
    crop_image_with_padding,
)
from pathlib import Path
import math
from qwen_vl_utils import smart_resize


def load_jsonl(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    # sort by item_idx
    # data.sort(key=lambda x: x["item_idx"])

    item_idx_to_data = dict()  # deduplicate
    for item in data:
        item_idx_to_data[item["item_idx"]] = item
    data = list(item_idx_to_data.values())
    return data


def draw_dot(image, position):
    draw = ImageDraw.Draw(image)
    draw.ellipse([position[0] - 5, position[1] - 5, position[0] + 5, position[1] + 5], fill="red")
    return image


def draw_cursor(image, position, cursor_image):
    image.paste(cursor_image, (position[0], position[1]), cursor_image)
    return image


def find_non_overlapping_text_position(pos, text, draw, used_positions, font=None, text_offset_distance=20):
    """
    Find a non-overlapping position for text near the given position.
    Tries multiple offset positions in a spiral pattern around the point.

    Args:
        pos: The position of the cursor dot
        text: The text to draw
        draw: ImageDraw object
        used_positions: List of already used text positions
        font: Font to use for text
        text_offset_distance: Base distance (in pixels) to offset text from cursor. Default is 20.
    """
    # Get text bounding box
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Define candidate positions in order of preference (clockwise spiral)
    # Format: (x_offset, y_offset) relative to the dot position
    # Scale all offsets based on text_offset_distance parameter
    base_dist = text_offset_distance
    far_dist = text_offset_distance * 1.75

    offsets = [
        (base_dist, base_dist / 2),  # right
        (base_dist, -base_dist),  # top-right
        (-text_width - base_dist * 0.75, base_dist / 2),  # left
        (-text_width - base_dist * 0.75, -base_dist),  # top-left
        (base_dist, base_dist * 1.5),  # bottom-right
        (-text_width - base_dist * 0.75, base_dist * 1.5),  # bottom-left
        (base_dist, -base_dist * 2),  # far top-right
        (-text_width - base_dist * 0.75, -base_dist * 2),  # far top-left
        (far_dist, base_dist / 2),  # far right
        (-text_width - far_dist * 0.85, base_dist / 2),  # far left
        (base_dist, base_dist * 2.25),  # far bottom-right
        (-text_width - base_dist * 0.75, base_dist * 2.25),  # far bottom-left
    ]

    def check_overlap(test_pos, test_width, test_height):
        """Check if this position overlaps with any used position."""
        for used_x, used_y, used_w, used_h in used_positions:
            # Check for rectangle overlap
            if not (
                test_pos[0] + test_width < used_x
                or test_pos[0] > used_x + used_w
                or test_pos[1] + test_height < used_y
                or test_pos[1] > used_y + used_h
            ):
                return True
        return False

    # Try each offset position
    for offset_x, offset_y in offsets:
        candidate_pos = (pos[0] + offset_x, pos[1] + offset_y)
        if not check_overlap(candidate_pos, text_width, text_height):
            return candidate_pos, text_width, text_height

    # If all positions overlap, return the first one anyway
    return (pos[0] + offsets[0][0], pos[1] + offsets[0][1]), text_width, text_height


def translate_position(
    image_width, image_height, position_history, position_history_global, focus_left_top_history, cursor_focus_sizes
):
    if focus_left_top_history is None or len(focus_left_top_history) == 0:
        return position_history

    assert len(focus_left_top_history) == 1
    assert len(position_history_global) == 1

    crop_left, crop_top = focus_left_top_history[0]
    after_ccf_positions = position_history_global[0]

    translated_position_history = [pos for pos in position_history]

    if (image_width == 3840 and image_height == 1080) or (image_width == 5120 and image_height == 1440):
        cursor_x_ratio = translated_position_history[-1][0] / image_width
        if cursor_x_ratio < 0.5:
            crop_top, crop_left = 0, 0
        else:
            crop_top, crop_left = 0, image_width // 2
        crop_width, crop_height = image_width // 2, image_height
    elif cursor_focus_sizes > 1.0:
        cropped_image_pixels = cursor_focus_sizes
        crop_width = int(math.sqrt((cropped_image_pixels * image_width) / image_height))
        crop_height = int(math.sqrt((cropped_image_pixels * image_height) / image_width))
    else:
        crop_width = int(image_width * cursor_focus_sizes)
        crop_height = int(image_height * cursor_focus_sizes)

    after_height, after_width = smart_resize(
        crop_height, crop_width, factor=28, min_pixels=2048 * 28 * 28, max_pixels=10240 * 28 * 28
    )
    for after_pos in after_ccf_positions:
        x, y = after_pos
        after_x_proportion = x / after_width
        after_y_proportion = y / after_height
        translated_x = int(after_x_proportion * crop_width) + crop_left
        translated_y = int(after_y_proportion * crop_height) + crop_top
        translated_position_history.append((translated_x, translated_y))
    print(translated_position_history)
    return translated_position_history


def draw_from_trajectory(
    image,
    position_history,
    position_history_global,
    cursor_focus_sizes,
    focus_left_top_history,
    cursor_image,
    text_offset_distance=20,
):
    """
    Draw trajectory on image with dots/cursor and numbered labels.

    Args:
        image: PIL Image to draw on
        position_history: List of cursor positions
        position_history_global: Global position history (if using cursor focus)
        cursor_focus_sizes: Sizes of cursor focus regions
        focus_left_top_history: History of focus region positions
        cursor_image: Cursor image to draw at each position
        text_offset_distance: Distance (in pixels) to offset text labels from cursor. Default is 20.
    """
    with_ccf = True
    if focus_left_top_history is None or len(focus_left_top_history) == 0:
        with_ccf = False

    if with_ccf:
        # translate the position to one image
        position_history = translate_position(
            image.width,
            image.height,
            position_history,
            position_history_global,
            focus_left_top_history,
            cursor_focus_sizes,
        )

    draw = ImageDraw.Draw(image)

    font = ImageFont.load_default(size=16)
    # Track used text positions to avoid overlaps
    used_positions = []
    # First pass: draw connecting lines between dots
    for idx in range(len(position_history) - 1):
        draw.line([position_history[idx], position_history[idx + 1]], fill="blue", width=2)

    # Second pass: draw all dots
    for idx, pos in enumerate(position_history):
        # image = draw_dot(image, pos)
        image = draw_cursor(image, (pos[0], pos[1]), cursor_image)

    # Third pass: draw text labels without collisions
    for idx, pos in enumerate(position_history):
        text = str(idx)
        text_pos, text_width, text_height = find_non_overlapping_text_position(
            pos, text, draw, used_positions, font, text_offset_distance=text_offset_distance
        )

        # Draw text with background for better visibility
        # Get the actual bounding box for precise positioning
        bbox = draw.textbbox(text_pos, text, font=font)
        padding = 4  # Increased padding for larger font

        # Draw background rectangle using actual text bounds
        draw.rectangle(
            [
                bbox[0] - padding,
                bbox[1] - padding,
                bbox[2] + padding,
                bbox[3] + padding,
            ],
            fill=(255, 255, 255, 200),
            outline="blue",
        )
        draw.text(text_pos, text, fill="blue", font=font)

        # Record this position as used (using actual bbox)
        used_positions.append(
            (bbox[0] - padding, bbox[1] - padding, bbox[2] - bbox[0] + 2 * padding, bbox[3] - bbox[1] + 2 * padding)
        )

    return image


def draw_a_result(exp_name, dataset_name, text_offset_distance=20):
    """
    Draw trajectory results for a dataset.

    Args:
        exp_name: Name of the experiment
        dataset_name: Name of the dataset
        text_offset_distance: Distance (in pixels) to offset text labels from cursor. Default is 20.
    """
    save_path = Path("./traj_check/") / dataset_name / exp_name
    os.makedirs(save_path, exist_ok=True)

    path = Path("./outputs") / dataset_name / exp_name / "predictions.jsonl"
    results = load_jsonl(path)
    idx2pred = {item["item_idx"]: item for item in results}

    msg_path = Path("./outputs") / dataset_name / exp_name / "message_history.jsonl"
    msg_results = load_jsonl(msg_path)
    idx2msg = {item["item_idx"]: item for item in msg_results}

    iter_dataset = load_iterable_dataset(dataset_name, load_image=True)
    cursor_image = Image.open("./cursor/resources/cursor-icon-cropped.png").convert("RGBA")
    for idx, item in enumerate(iter_dataset):
        if idx >= 10:
            break

        pred = idx2pred[item["item_idx"]]
        msg = idx2msg[item["item_idx"]]

        focus_left_top_history = pred.get("focus_left_top_history", None)
        position_history_global = pred.get("position_history_global", None)
        cursor_focus_sizes = 1920 * 1080
        position_history = pred["position_history"]
        bbox_proportions = item["bbox_proportions"]
        correct = pred["within_bbox_history"][-1]
        image = item["image"]

        item_save_dir = save_path / f"{idx}-{correct}-{item['item_idx']}"
        if not correct:
            os.makedirs(item_save_dir, exist_ok=True)
            image.save(item_save_dir / f"{idx}-original-{item['item_idx']}.png")

        image = smart_resize_min_max_tokens_per_img(image=image, min_tokens_per_img=2048, max_tokens_per_img=10240)
        image = add_bbox_to_screenshot(image, bbox_coords_from_proportions(image.width, image.height, bbox_proportions))
        if pred["stop_reason"] == "target_reached":
            position_history = position_history[:-1]
        image = draw_from_trajectory(
            image,
            position_history,
            position_history_global,
            cursor_focus_sizes,
            focus_left_top_history,
            cursor_image,
            text_offset_distance=text_offset_distance,
        )

        if not correct:
            image.save(item_save_dir / "image.png")
            json.dump(msg, open(item_save_dir / "message.json", "w"), indent=2)


def main():

    # exp_name = "GUI-Cursor-uitars-120steps"
    # dataset_name = "OSWorld-G_refined"
    dataset_name = "ScreenSpot-Pro"
    exp_name = "GUI-Cursor-uitars-120steps"
    # Adjust text_offset_distance to control how far text labels are from cursor
    # Default is 20, increase for more distance, decrease for less
    draw_a_result(exp_name, dataset_name, text_offset_distance=20)


if __name__ == "__main__":
    main()
