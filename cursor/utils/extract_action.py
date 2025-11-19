import re
from typing import Optional, Tuple, List, Union


def extract_coord_from_answer_tag(response_text: str) -> Optional[Tuple[int, int]]:
    match = re.search(r"<answer>(.*?)</answer>", response_text)

    if not match:
        print("Error: <answer> tag not found.")
        return None

    content = match.group(1).strip()

    # Check if the content is "STOP"
    if content == "STOP":
        return "TARGET_REACHED", None, None

    # Regular expression to find integer coordinates in the format (x, y)
    # coord_match = re.match(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", content)
    coord_match = re.match(r'^\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)$', content)  #
    try:
        if coord_match:
            x = int(coord_match.group(1))
            y = int(coord_match.group(2))
            return "GOTO", x, y
        else:
            return None, None, None
    except Exception:
        return None, None, None


def extract_action_tag(response_text: str) -> tuple[str | None, int | None, int | None]:
    pattern = r"<action>(.*?)</action>"
    match = re.search(pattern, response_text, re.DOTALL)
    if match:
        action = match.group(1).strip()
        if "STOP" in action or "TARGET_REACHED" in action:
            return "TARGET_REACHED", None, None
        else:
            return extract_from_action(action)
    return None, None, None


def extract_from_action(llm_response: str) -> tuple[str | None, int | None, int | None]:
    pattern = r"(MOVE|GOTO)\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)"
    all_matches = list(re.finditer(pattern, llm_response))

    if all_matches:
        last_match = all_matches[-1]
        try:
            action = last_match.group(1)
            x_str = last_match.group(2)
            y_str = last_match.group(3)

            x = int(x_str)
            y = int(y_str)

            return action, x, y
        except ValueError:
            return None, None, None
    elif "TARGET_REACHED" in llm_response:
        return "TARGET_REACHED", None, None

    else:
        return None, None, None


def extract_move_action(llm_response: str) -> tuple[str | None, int | None, int | None]:
    pattern = r"MOVE\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)"
    all_matches = list(re.finditer(pattern, llm_response))
    if all_matches:
        # Get the last match
        last_match = all_matches[-1]
        try:
            dx_str = last_match.group(1)
            dy_str = last_match.group(2)
            dx = int(dx_str)
            dy = int(dy_str)
            return "MOVE", dx, dy
        except ValueError:
            return None, None, None
    elif "TARGET_REACHED" in llm_response:
        return "TARGET_REACHED", None, None

    else:
        return None, None, None


def extract_last_coordinate(text: str) -> Optional[Tuple[int, int]]:
    pattern = r"\(([-]?\d+),\s*([-]?\d+)\)|\[([-]?\d+),\s*([-]?\d+)\]"
    matches = list(re.finditer(pattern, text))
    if not matches:
        return None, None, None
    last_match = matches[-1]
    x_str = last_match.group(1) or last_match.group(3)
    y_str = last_match.group(2) or last_match.group(4)
    x = int(x_str)
    y = int(y_str)
    return "MOVE_TO", x, y


def extract_uitars_action(response_text: str) -> tuple[str | None, int | None, int | None]:
    """
    Extract action from the format:
    Thought: ...
    Action: GOTO(x, y) or STOP
    """
    # Pattern to match GOTO action with coordinates
    goto_pattern = r"Action:\s*GOTO\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)"
    goto_match = re.search(goto_pattern, response_text)
    
    if goto_match:
        try:
            x = int(goto_match.group(1))
            y = int(goto_match.group(2))
            return "GOTO", x, y
        except ValueError:
            return None, None, None
    
    # Pattern to match STOP action
    stop_pattern = r"Action:\s*STOP"
    stop_match = re.search(stop_pattern, response_text)
    
    if stop_match:
        return "TARGET_REACHED", None, None
    
    return None, None, None


def debug():
    # print(extract_move_action("MOVE(10, 29) and MOVE(30, 40.0)"))
    print(extract_coord_from_answer_tag("<answer>(123.0, 456)</answer>"))


ACTION_EXTRACTOR_FN = {
    "move_action": extract_move_action,
    "last_coordinate": extract_last_coordinate,
    "action_tag": extract_action_tag,
    "coord_answer_tag": extract_coord_from_answer_tag,
    "uitars_action": extract_uitars_action,
}

if __name__ == "__main__":
    debug()