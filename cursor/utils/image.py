import math
from PIL import Image, ImageDraw
from qwen_vl_utils import smart_resize


def calculate_distance(x1: int, y1: int, x2: int, y2: int) -> float:
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def is_within_bbox(x, y, bbox_coords):
    top_left_x, top_left_y, bottom_right_x, bottom_right_y = bbox_coords
    return top_left_x <= x <= bottom_right_x and top_left_y <= y <= bottom_right_y


def add_bbox_to_screenshot(screenshot: Image.Image, bbox_coords) -> Image.Image:
    draw = ImageDraw.Draw(screenshot)
    draw.rectangle(bbox_coords, outline="red", width=2)
    return screenshot


def bbox_coords_from_proportions(width, height, prop_bbox):
    top_left_x, top_left_y, bottom_right_x, bottom_right_y = prop_bbox
    bbox_coords = (
        int(top_left_x * width),
        int(top_left_y * height),
        int(bottom_right_x * width),
        int(bottom_right_y * height),
    )
    return bbox_coords


def get_distance_ratio_to_center(px, py, bbox, image_width, image_height) -> dict[str, float]:
    xmin, ymin, xmax, ymax = bbox

    # Normalize point coordinates
    norm_px = px / image_width
    norm_py = py / image_height

    # Normalize bounding box coordinates
    norm_xmin = xmin / image_width
    norm_ymin = ymin / image_height
    norm_xmax = xmax / image_width
    norm_ymax = ymax / image_height

    # Ensure correct order for normalized bbox (though original bbox should be correct)
    if norm_xmin > norm_xmax:
        norm_xmin, norm_xmax = norm_xmax, norm_xmin
    if norm_ymin > norm_ymax:
        norm_ymin, norm_ymax = norm_ymax, norm_ymin

    # Calculate the center of the normalized bounding box
    norm_center_x = (norm_xmin + norm_xmax) / 2
    norm_center_y = (norm_ymin + norm_ymax) / 2

    distance_to_corner = math.sqrt((norm_xmin - norm_center_x) ** 2 + (norm_ymin - norm_center_y) ** 2)
    distance_to_point = math.sqrt((norm_px - norm_center_x) ** 2 + (norm_py - norm_center_y) ** 2)
    ratio = distance_to_point / distance_to_corner if distance_to_corner != 0 else 0
    return ratio


def cal_normalized_distances_to_bbox(point, bbox, image_width, image_height):
    """
    Calculates distances from a point to a bounding box using coordinates
    normalized by the image dimensions (image treated as 1x1).

    Args:
        point (tuple): A tuple (x, y) representing the original pixel coordinate.
        bbox (tuple): A tuple (xmin, ymin, xmax, ymax) representing the
                      original pixel coordinates of the bounding box.
        image_width (float): The width of the image in pixels.
        image_height (float): The height of the image in pixels.

    Returns:
        dict: A dictionary containing:
              'distance_to_center': The Euclidean distance to the bbox center
                                    in the normalized 1x1 space.
              'shortest_distance_to_box': The shortest Euclidean distance to
                                          the bbox edge/corner in the
                                          normalized 1x1 space.
              'normalized_point': The (x, y) point normalized.
              'normalized_bbox': The (xmin, ymin, xmax, ymax) bbox normalized.
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image width and height must be positive.")

    px, py = point
    xmin, ymin, xmax, ymax = bbox

    # Normalize point coordinates
    norm_px = px / image_width
    norm_py = py / image_height

    # Normalize bounding box coordinates
    norm_xmin = xmin / image_width
    norm_ymin = ymin / image_height
    norm_xmax = xmax / image_width
    norm_ymax = ymax / image_height

    # Ensure correct order for normalized bbox (though original bbox should be correct)
    if norm_xmin > norm_xmax:
        norm_xmin, norm_xmax = norm_xmax, norm_xmin
    if norm_ymin > norm_ymax:
        norm_ymin, norm_ymax = norm_ymax, norm_ymin

    # Calculate the center of the normalized bounding box
    norm_center_x = (norm_xmin + norm_xmax) / 2
    norm_center_y = (norm_ymin + norm_ymax) / 2

    # Calculate the distance to the center in normalized space
    distance_to_center = math.sqrt((norm_px - norm_center_x) ** 2 + (norm_py - norm_center_y) ** 2)

    # Calculate the shortest distance to the normalized box
    # Determine the closest x and y coordinates on the normalized box to the normalized point
    closest_norm_x = max(norm_xmin, min(norm_px, norm_xmax))
    closest_norm_y = max(norm_ymin, min(norm_py, norm_ymax))

    distance_to_bbox = math.sqrt((norm_px - closest_norm_x) ** 2 + (norm_py - closest_norm_y) ** 2)

    return {
        "distance_to_center": distance_to_center,
        "distance_to_bbox": distance_to_bbox,
        "normalized_point": (norm_px, norm_py),
        "normalized_bbox": (norm_xmin, norm_ymin, norm_xmax, norm_ymax),
    }


def cal_distance_to_bbox(point, bbox):
    """
    Calculates the distance from a point to the center of a bounding box
    and the shortest distance from the point to the bounding box.

    Args:
        point (tuple): A tuple (x, y) representing the coordinate.
        bbox (tuple): A tuple (xmin, ymin, xmax, ymax) representing the
                      bounding box, where (xmin, ymin) is the bottom-left
                      corner and (xmax, ymax) is the top-right corner.

    Returns:
        dict: A dictionary containing:
              'distance_to_center': The Euclidean distance to the bbox center.
              'shortest_distance_to_box': The shortest Euclidean distance to
                                          the bbox edge or corner.
    """
    px, py = point
    xmin, ymin, xmax, ymax = bbox

    if xmin > xmax:
        xmin, xmax = xmax, xmin
    if ymin > ymax:
        ymin, ymax = ymax, ymin

    # Calculate the center of the bounding box
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2

    # Calculate the distance to the center
    distance_to_center = math.sqrt((px - center_x) ** 2 + (py - center_y) ** 2)

    # Calculate the shortest distance to the box
    # Determine the closest x and y coordinates on the box to the point
    closest_x = max(xmin, min(px, xmax))
    closest_y = max(ymin, min(py, ymax))

    # If the point is inside the box, the shortest distance is 0
    # (or negative if we consider signed distance, but for geometric distance it's 0)
    # However, the calculation below will yield 0 if inside.

    distance_to_bbox = math.sqrt((px - closest_x) ** 2 + (py - closest_y) ** 2)

    return {
        "distance_to_center": distance_to_center,
        "distance_to_bbox": distance_to_bbox,
        "x_distance": math.fabs(px - closest_x),
        "y_distance": math.fabs(py - closest_y),
    }


def downscale_image(image: Image.Image, max_tokens_per_img=None) -> Image.Image:
    width, height = image.size
    if width * height <= max_tokens_per_img * 28 * 28:
        return image
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=28,
        min_pixels=0,
        max_pixels=max_tokens_per_img * 28 * 28,
    )
    image = image.resize((resized_width, resized_height))
    return image


def smart_resize_min_max_tokens_per_img(
    image: Image.Image, min_tokens_per_img=None, max_tokens_per_img=None
) -> Image.Image:
    width, height = image.size
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=28,
        min_pixels=min_tokens_per_img * 28 * 28 if min_tokens_per_img else 0,
        max_pixels=max_tokens_per_img * 28 * 28 if max_tokens_per_img else 16384 * 28 * 28,
    )
    image = image.resize((resized_width, resized_height))
    return image


def crop_image_at_coordinate(image, center_x, center_y, crop_width, crop_height):
    """
    Crops a specified area from an image, centered around a given coordinate.

    This function calculates the bounding box for the crop, ensuring it does not
    extend beyond the image's boundaries. If the desired crop area is near an
    edge, the box is adjusted to stay within the image, effectively shifting
    the center.

    Args:
        image (PIL.Image.Image): The input image to crop.
        center_x (int): The x-coordinate for the center of the crop.
        center_y (int): The y-coordinate for the center of the crop.
        crop_width (int): The desired width of the cropped image.
        crop_height (int): The desired height of the cropped image.

    Returns:
        PIL.Image.Image: The cropped image.
    """
    from PIL import Image, ImageDraw

    img_width, img_height = image.size

    # Calculate the top-left corner coordinates of the crop box
    # The goal is to make (center_x, center_y) the center of the box
    left = center_x - crop_width // 2
    top = center_y - crop_height // 2

    # Calculate the bottom-right corner coordinates
    right = left + crop_width
    bottom = top + crop_height

    # --- Edge Case Handling ---
    # Adjust the crop box if it extends beyond the image boundaries.

    # If the box is too far to the left
    if left < 0:
        left = 0
        right = crop_width

    # If the box is too far to the right
    if right > img_width:
        right = img_width
        left = img_width - crop_width

    # If the box is too high
    if top < 0:
        top = 0
        bottom = crop_height

    # If the box is too low
    if bottom > img_height:
        bottom = img_height
        top = img_height - crop_height

    # Ensure the final crop dimensions are not larger than the image itself
    if crop_width > img_width:
        left, right = 0, img_width
    if crop_height > img_height:
        top, bottom = 0, img_height

    # Define the final crop box as a tuple
    crop_box = (left, top, right, bottom)

    # print(f"Original image size: {image.size}")
    # print(f"Requested crop center: ({center_x}, {center_y})")
    # print(f"Calculated crop box: {crop_box}")

    # Crop the image using the calculated box
    cropped_image = image.crop(crop_box)

    return cropped_image, left, top


def crop_image_with_padding(image, center_x, center_y, crop_width, crop_height):
    """
    Crops an area from an image, centered around a coordinate, and pads with
    a white background if the crop extends beyond image boundaries.

    It also returns the coordinates of the top-left corner of the crop box
    relative to the original image's coordinate system.

    Args:
        image (PIL.Image.Image): The input image to crop.
        center_x (int): The x-coordinate for the center of the crop.
        center_y (int): The y-coordinate for the center of the crop.
        crop_width (int): The desired width of the cropped image.
        crop_height (int): The desired height of the cropped image.

    Returns:
        tuple[PIL.Image.Image, int, int]: A tuple containing:
            - The cropped and potentially padded image.
            - The 'left' x-coordinate of the output image in the original image's space.
            - The 'top' y-coordinate of the output image in the original image's space.
    """
    img_width, img_height = image.size

    # Define the color for padding based on the image mode
    padding_color = (255, 255, 255, 255) if image.mode == 'RGBA' else (255, 255, 255)

    # Create a new image with the specified crop size and white background
    background = Image.new(image.mode, (crop_width, crop_height), padding_color)

    # --- Calculate Coordinates ---

    # Calculate the top-left corner of the crop box in the original image's coordinate system.
    # These can be negative if the crop extends beyond the top or left edge.
    left_in_original = center_x - crop_width // 2
    top_in_original = center_y - crop_height // 2

    # Determine the region of the original image that will be copied.
    src_left = max(0, left_in_original)
    src_top = max(0, top_in_original)
    src_right = min(img_width, left_in_original + crop_width)
    src_bottom = min(img_height, top_in_original + crop_height)

    # If the crop area is valid, perform the crop and paste
    if src_right > src_left and src_bottom > src_top:
        source_crop = image.crop((src_left, src_top, src_right, src_bottom))

        # Determine where to paste the cropped section onto the new background.
        paste_x = max(0, -left_in_original)
        paste_y = max(0, -top_in_original)

        background.paste(source_crop, (paste_x, paste_y))

    # Return the final image along with the top-left coordinates
    return background, left_in_original, top_in_original