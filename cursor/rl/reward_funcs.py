import re
from turtle import distance
from typing import Any, Dict, List
from cursor.utils.extract_action import ACTION_EXTRACTOR_FN
from cursor.env_and_state import State, Environment
from cursor.utils.image import is_within_bbox, cal_normalized_distances_to_bbox, get_distance_ratio_to_center
import math
import numpy as np


def get_answer_without_think_format_reward(state: State):
    responses = [msg["content"] for msg in state.message_history if msg["role"] == "assistant"]
    history_responses = responses[:-1]
    last_responses = responses[-1]

    pattern = r"<answer>\(\d+,\s*\d+\)</answer>"
    matches = [re.fullmatch(pattern, content.strip(), re.DOTALL) for content in history_responses]
    reward = [1.0 if match else 0.0 for match in matches]

    pattern = r"<answer>(\(\d+,\s*\d+\)|STOP)</answer>"
    matches = re.fullmatch(pattern, last_responses.strip(), re.DOTALL)
    reward.append(1.0 if matches else 0.0)
    reward = sum(reward) / len(reward)
    return reward


def get_uitars_think_answer_format_reward(state: State) -> float:
    responses = [msg["content"] for msg in state.message_history if msg["role"] == "assistant"]
    history_responses = responses[:-1]
    last_responses = responses[-1]

    pattern = r"Thought:.*?Action:\s*GOTO\(\s*\d+\s*,\s*\d+\s*\)"
    matches = [re.fullmatch(pattern, content.strip(), re.DOTALL) for content in history_responses]
    reward = [1.0 if match else 0.0 for match in matches]

    pattern = r"Thought:.*?Action:\s*(GOTO\(\s*\d+\s*,\s*\d+\s*\)|STOP)"
    matches = re.fullmatch(pattern, last_responses.strip(), re.DOTALL)
    reward.append(1.0 if matches else 0.0)
    reward = sum(reward) / len(reward)
    return reward


def get_think_answer_format_reward(state: State) -> float:
    responses = [msg["content"] for msg in state.message_history if msg["role"] == "assistant"]
    history_responses = responses[:-1]
    last_responses = responses[-1]

    pattern = r"<think>.*?</think>\s*<answer>\(\d+,\s*\d+\)</answer>"
    matches = [re.fullmatch(pattern, content.strip(), re.DOTALL) for content in history_responses]
    reward = [1.0 if match else 0.0 for match in matches]

    pattern = r"<think>.*?</think>\s*<answer>(\(\d+,\s*\d+\)|STOP)</answer>"
    matches = re.fullmatch(pattern, last_responses.strip(), re.DOTALL)
    reward.append(1.0 if matches else 0.0)
    reward = sum(reward) / len(reward)
    return reward


def get_success_reward(state: State) -> float:
    if state.stop_reason == "target_reached" and is_within_bbox(
        state.cursor_x, state.cursor_y, state.environment.bbox_coords
    ):
        return 1.0
    else:
        return 0.0


def get_distance_to_edge_reward(state: State, distance_score_func: callable) -> float:
    if is_within_bbox(state.cursor_x, state.cursor_y, state.environment.bbox_coords):
        return 1.0
    else:
        if state.stop_reason == "extract_action_failed":
            return 0.0
        return distance_score_func(state, state.cursor_x, state.cursor_y)


def get_distance_to_center_reward(state: State, x, y) -> float:
    if not is_within_bbox(x, y, state.environment.bbox_coords):
        return 0.0

    ratio = get_distance_ratio_to_center(
        x,
        y,
        state.environment.bbox_coords,
        state.environment.screenshot.width,
        state.environment.screenshot.height,
    )
    return 1 - ratio**5


def get_steps_reward(state: State):
    if state.stop_reason == "target_reached":
        # fist position is the initial one
        # the second step can be stop (the cursor is at the target occasionally) or move
        move_steps = len(state.position_history) - 1
        return 1 / move_steps  # **0.5
    else:
        return 0.0


def is_move_correct(state: State, move_position_history):
    pos_first = move_position_history[0]
    pos_last = move_position_history[-1]
    first_distance = cal_normalized_distances_to_bbox(
        (pos_first[0], pos_first[1]),
        state.environment.bbox_coords,
        state.environment.screenshot.width,
        state.environment.screenshot.height,
    )["distance_to_bbox"]
    last_distance = cal_normalized_distances_to_bbox(
        (pos_last[0], pos_last[1]),
        state.environment.bbox_coords,
        state.environment.screenshot.width,
        state.environment.screenshot.height,
    )["distance_to_bbox"]
    if last_distance < first_distance:
        return True
    else:
        return False


def no_repeated_position(move_position_history) -> bool:
    seen_positions = set()
    for pos in move_position_history:
        if (pos[0], pos[1]) in seen_positions:
            return False
        seen_positions.add((pos[0], pos[1]))
    return True


def linear_distance_score(state: State, x, y):
    distance = cal_normalized_distances_to_bbox(
        (x, y),
        state.environment.bbox_coords,
        state.environment.screenshot.width,
        state.environment.screenshot.height,
    )["distance_to_bbox"]
    return 1.0 - distance


def powerfunc_distance_score(state: State, x, y):
    distance = cal_normalized_distances_to_bbox(
        (x, y),
        state.environment.bbox_coords,
        state.environment.screenshot.width,
        state.environment.screenshot.height,
    )["distance_to_bbox"]
    return 1.0 - distance**0.5


def get_in_bbox_reward(state: State):
    final_position = state.position_history[-1]
    if is_within_bbox(final_position[0], final_position[1], state.environment.bbox_coords):
        return 1.0
    else:
        return 0.0


def get_traj_reward(
    state: State,
    distance_score_func: callable,
    distance_to_center_weight: float = 0.1,
    false_stop_penalty: float = -0.5,
    false_move_penalty: float = -0.2,
    correct_move_bonus: float = 0.1,
    repeated_position_penalty: float = -0.2,
    reached_but_no_stop_penalty: float = -0.1,  # add when stop because maximum steps reached
) -> float:
    assert (
        false_stop_penalty <= 0
        and false_move_penalty <= 0
        and repeated_position_penalty <= 0
        and reached_but_no_stop_penalty <= 0
        and correct_move_bonus >= 0
    )

    penalty_log = {
        "false_stop": False,  # false_stop_penalty
        "false_direction": False,  # false_direction_penalty
        "repeated_position": False,  # repeated_position_penalty
        "false_move": False,  # false_move_penalty
    }

    cursor_is_within_bbox = is_within_bbox(state.cursor_x, state.cursor_y, state.environment.bbox_coords)
    if state.stop_reason == "extract_action_failed":
        # stop because fail to extract action, format error
        return {"score": 0.0, "penalty_log": penalty_log}
    if state.stop_reason == "target_reached":
        # stop because the model output STOP
        if cursor_is_within_bbox:
            # correct trajectory
            distance_to_center_score = get_distance_to_center_reward(state, state.cursor_x, state.cursor_y)
            return {"score": 1.0 + distance_to_center_weight * distance_to_center_score, "penalty_log": penalty_log}
        else:
            # output STOP but not at the target
            distance_to_center_score = distance_score_func(state, state.cursor_x, state.cursor_y)
            # move_position_history: exclude the initial (center) and final position (last one is STOP)
            move_position_history = state.position_history[1:-1]

            if not no_repeated_position(move_position_history):
                penalty_log["repeated_position"] = True
            # if any(reached_target_but_no_stop_history):
            #     penalty_log["false_move"] = True

            if len(move_position_history) < 2:
                penalty_log["false_stop"] = True
                return {"score": false_stop_penalty + distance_to_center_score, "penalty_log": penalty_log}
            else:
                if is_move_correct(state, move_position_history):
                    penalty_log["false_stop"] = True
                    return {
                        "score": false_stop_penalty + distance_to_center_score + correct_move_bonus,
                        "penalty_log": penalty_log,
                    }
                else:
                    penalty_log["false_stop"] = True
                    penalty_log["false_direction"] = True
                    return {
                        "score": false_stop_penalty + distance_to_center_score + false_move_penalty,
                        "penalty_log": penalty_log,
                    }
    if state.stop_reason == "max_steps_reached":
        # stop because max steps reached, the model did not output STOP
        penalty = 0.0
        distance_to_edge_score = distance_score_func(state, state.cursor_x, state.cursor_y)
        distance_to_center_score = get_distance_to_center_reward(
            state, state.cursor_x, state.cursor_y
        )  # 0 if not within bbox
        distance_score = distance_to_edge_score + distance_to_center_weight * distance_to_center_score

        # the max_steps_reached but there is a coorect position in the history, give a penalty
        reached_target_but_no_stop_history = [
            is_within_bbox(position[0], position[1], state.environment.bbox_coords)
            for position in state.position_history[:-1]
        ]
        if any(reached_target_but_no_stop_history):
            penalty_log["false_move"] = True
            penalty += reached_but_no_stop_penalty
        # max_steps_reached but explored the same position multiple times, give a penalty
        move_position_history = state.position_history[1:]
        assert len(move_position_history) >= 2
        if not no_repeated_position(move_position_history):
            penalty += repeated_position_penalty
            penalty_log["repeated_position"] = True
        # judge whether the movement is correct, if not, give a penalty
        if not is_move_correct(state, move_position_history):
            penalty += false_move_penalty
            penalty_log["false_direction"] = True
        assert penalty <= 0
        return {"score": distance_score + penalty, "penalty_log": penalty_log}


def get_num_steps(state: State) -> int:
    num_steps = len(state.position_history)
    if state.stop_reason == "target_reached":
        num_steps -= 1
    return num_steps


def calculate_rewards(
    states: List[State],
    format_weight: float = 0.1,
    steps_weight: float = 0.1,
    distance_weight: float = 0.3,
    success_weight: float = 0.5,
    traj_weight: float = 0.9,
    distance_func_name="linear",
    distance_to_center_weight: float = 0.1,
    false_stop_penalty: float = -0.5,
    false_move_penalty: float = -0.2,
    correct_move_bonus: float = 0.1,
    repeated_position_penalty: float = -0.2,
    reached_but_no_stop_penalty: float = -0.1,
    reward_method_name="traj_reward",
    format_without_think=False,
    format_function_name="qwen_format_reward",
):
    if distance_func_name == "linear":
        distance_score_func = linear_distance_score
    elif distance_func_name == "powerfunc":
        distance_score_func = powerfunc_distance_score
    else:
        raise ValueError(f"Unknown distance function name: {distance_func_name}, valid options are: linear, powerfunc")

    scores: List[Dict[str, float]] = []

    if reward_method_name == "traj_reward":
        assert format_weight + traj_weight == 1.0

        for state in states:
            if state.environment.max_steps == 1:
                traj_reward = distance_score_func(state, state.cursor_x, state.cursor_y)
                traj_penalty_log = {}
            else:
                traj_results = get_traj_reward(
                    state,
                    distance_score_func=distance_score_func,
                    distance_to_center_weight=distance_to_center_weight,
                    false_stop_penalty=false_stop_penalty,
                    false_move_penalty=false_move_penalty,
                    correct_move_bonus=correct_move_bonus,
                    repeated_position_penalty=repeated_position_penalty,
                    reached_but_no_stop_penalty=reached_but_no_stop_penalty,
                )
                traj_reward = traj_results["score"]
                traj_penalty_log = traj_results["penalty_log"]
            if format_without_think:
                format_reward = get_answer_without_think_format_reward(state)
            else:
                if format_function_name == "uitars_format_reward":
                    format_reward = get_uitars_think_answer_format_reward(state)
                elif format_function_name == "qwen_format_reward":
                    format_reward = get_think_answer_format_reward(state)
                else:
                    raise ValueError
            reward = {
                "overall": traj_weight * traj_reward + format_weight * format_reward,
                # the following values are for logging
                "format": format_reward,
                "distance": get_distance_to_edge_reward(state, linear_distance_score),
                "steps": get_steps_reward(state),
                "num_steps": get_num_steps(state),
                "success": get_success_reward(state),
                "in_bbox": get_in_bbox_reward(state),
                "traj": traj_reward,
            }
            # add traj_penalty_log to the reward dict for logging
            # change True/False to 1.0/0.0
            for key in traj_penalty_log:
                traj_penalty_log[key] = 1.0 if traj_penalty_log[key] else 0.0
            reward.update(traj_penalty_log)
            scores.append(reward)

    elif reward_method_name == "distance_success_format_step":
        assert distance_weight + success_weight + format_weight + steps_weight == 1.0
        for state in states:
            format_reward = get_think_answer_format_reward(state)
            steps_reward = get_steps_reward(state)
            success_reward = get_success_reward(state)
            distance_score = get_distance_to_edge_reward(state, distance_score_func)

            reward = {
                "overall": distance_weight * distance_score
                + success_weight * success_reward
                + format_weight * format_reward
                + steps_weight * steps_reward,
                # the following values are for logging
                "format": format_reward,
                "distance": distance_score,
                "steps": steps_reward,
                "success": success_reward,
            }
            scores.append(reward)

    elif reward_method_name == "discrete_success_reward":
        for state in states:
            in_bbox_reward = get_in_bbox_reward(state)
            if format_without_think:
                format_reward = get_answer_without_think_format_reward(state)
            else:
                format_reward = get_think_answer_format_reward(state)

            reward = {
                "overall": (1.0 - format_weight) * in_bbox_reward + format_weight * format_reward,
                # the following values are for logging
                "format": format_reward,
                "distance": get_distance_to_edge_reward(state, linear_distance_score),
                "steps": get_steps_reward(state),
                "success": get_success_reward(state),
                "in_bbox": in_bbox_reward,
            }
            scores.append(reward)

    else:
        raise NotImplementedError(f"Unknown reward method name: {reward_method_name}")

    return scores
