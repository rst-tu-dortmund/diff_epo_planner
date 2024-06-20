# ------------------------------------------------------------------------------
# Copyright:    ZF Friedrichshafen AG
# Project:      KISSaF
# Created by:   ZF AI LAB SBR
# ------------------------------------------------------------------------------
import copy
import logging
import math
import pickle
from pathlib import Path
from typing import Union, Tuple, Optional, List

import numpy as np
import torch

import data.utils.helpers as du
from data import constants as c
from data.utils.helpers import (
    transform_in_ego_coordinates,
    quaternion_from_2d_rotation,
)


def approximate_yaw(
    trajectory: np.array,
    last_input_timestep: int,
    method_flag: int = 1,
) -> float:
    """approximates yaw for the last input timestep from trajetory to generate
    a reference pose.

    :param trajectory:
    :param last_input_timestep:
    :param method_flag:
    :return:
    """
    if method_flag == c.ONE:
        yaw = linear_interpolation(trajectory, last_input_timestep)
    elif method_flag == c.TWO:
        yaw = linear_yaw_interpolation_with_velocity_threshold(
            trajectory,
            last_input_timestep,
        )
    else:
        raise ValueError(f"Unknown method_flag={method_flag}.")

    return yaw


def linear_interpolation(trajectory: np.array, last_input_timestep) -> float:
    """As in Lane GCN (data.py: L148-155) but we use x axis as vehicle
    orientation.

    :param trajectory:
    :param last_input_timestep:
    :return:
    """
    point1 = trajectory[last_input_timestep - 1, c.INDICES_COORDINATES]
    point2 = trajectory[last_input_timestep, c.INDICES_COORDINATES]
    yaw = calc_yaws_from_points(point1, point2)
    return yaw


def linear_yaw_interpolation_with_velocity_threshold(
    trajectory: np.array,
    last_input_timestep,
    velocity_threshold: float = 0.2,
) -> float:
    """Like linear_interpolation, but when velocity at last_input_timestep is
    below threshold, we look for the next time_step with sufficient velocity to
    estimate the orientation. If no velocity >= velocity_threshold found, use
    the time_step with the largest velocity.

    :param trajectory: trajectory with shape=[n_time_steps, (x,y position)]
    :param last_input_timestep: first time step to compute orientation
    :return:
    """

    def search_with_velocity_threshold(direction: int, init_step: int, max_step: int):
        v_max = -1.0
        time_step_max = None
        assert abs(direction) == 1
        time_step = init_step
        while direction * time_step <= max_step:
            velocity_sq = np.sum(
                np.square(
                    trajectory[time_step, c.INDICES_COORDINATES]
                    - trajectory[time_step - 1, c.INDICES_COORDINATES],
                ),
            )
            if velocity_sq > v_max:
                v_max = velocity_sq
                time_step_max = time_step

            if velocity_sq < velocity_threshold_sq:
                time_step += direction
                continue
            yaw = linear_interpolation(trajectory, time_step)
            return True, yaw, v_max, time_step_max

        return (
            False,
            linear_interpolation(trajectory, time_step_max),
            v_max,
            time_step_max,
        )

    velocity_threshold_sq = velocity_threshold**2
    last_input_timestep_init = last_input_timestep
    # look forward
    found, yaw_1, v_max_1, time_step_max_1 = search_with_velocity_threshold(
        1,
        last_input_timestep,
        trajectory.shape[0] - 1,
    )
    if found is True:
        return yaw_1
    # look backward
    found, yaw_2, v_max_2, time_step_max_2 = search_with_velocity_threshold(
        -1,
        last_input_timestep_init - 1,
        1,
    )
    if found is True:
        return yaw_2

    if v_max_1 > v_max_2:
        return yaw_1
    else:
        return yaw_2


def calc_yaws_from_points(start_point: np.ndarray, end_point: np.ndarray):
    """calculate yaw based on vector start point and vector end point.

    :param end_point:
    :param start_point:
    :return:
    """
    with np.errstate(all="ignore"):
        delta = end_point - start_point
    yaw = calc_yaws_from_vecs(delta)
    return yaw


def calc_yaws_from_vecs(delta: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """calculate yaw from 2D vector.

    :param delta:
    :return:
    """
    if isinstance(delta, torch.Tensor):
        yaw = torch.atan2(delta[..., 1], delta[..., 0])
    else:
        yaw = np.arctan2(delta[..., 1], delta[..., 0])
    return yaw


def yaws_from_vectors(vector: torch.Tensor, used_deltas: bool) -> torch.Tensor:
    """get the yaws from trajectory vectors and differs if vector format
    already contains the direction or are defined by start and end point if
    vector formats contains directions (deltas between trajectory points it is
    possible to directly use these deltas )

    :param vector:
    :param used_deltas: set to true if vector format contain deltas
    :return:
    """
    if used_deltas:
        yaw = calc_yaws_from_vecs(vector[..., c.DELTAS])
    else:
        yaw = calc_yaws_from_points(
            start_point=vector[..., c.INDICES_VECTOR_START],
            end_point=vector[..., c.INDICES_VECTOR_END],
        )
    return yaw


def delta_yaw_per_map_node(map: torch.Tensor, used_deltas: bool) -> torch.Tensor:
    """compute the overall difference in yaw between the start of the lane and
    the end of the lane on one side.

    :param map:
    :param used_deltas:
    :return:
    """
    device = map.device
    subnode_map = start_and_end_node(map)
    yaws = yaws_from_vectors(subnode_map, used_deltas=used_deltas)
    _, batch, nodes = list(yaws.shape)
    yaws.reshape(-1, batch * nodes)
    yaws = yaws.cpu().numpy()
    yaws_1 = yaws[c.FIRST_VALUE, ...]
    yaws_2 = yaws[c.SECOND_VALUE, ...]
    d_yaws = du.v_calc_delta_yaw(yaws_1, yaws_2)
    d_yaws = torch.from_numpy(d_yaws).reshape(batch, nodes, c.ONE).to(device)
    return d_yaws


def start_yaw_per_node(map: torch.Tensor, used_deltas: bool) -> torch.Tensor:
    """compute start yaw of a lane.

    :param map:
    :param used_deltas:
    :return:
    """
    subnodes = map[c.FIRST_VECTOR_ONE_SIDE, ...]
    yaws = yaws_from_vectors(subnodes, used_deltas).unsqueeze(c.LAST_VALUE).float()
    return yaws


def start_point(map: torch.Tensor) -> torch.Tensor:
    """get the first point of the lane.

    :param map:
    :return:
    """
    first_point = map[c.FIRST_VECTOR_ONE_SIDE, :, :, c.INDICES_VECTOR_END]
    return first_point.float()


def last_point(map: torch.Tensor) -> torch.Tensor:
    """get the last point of the lane.

    :param map:
    :return:
    """
    first_point = map[c.LAST_VECTOR_ONE_SIDE, :, :, c.INDICES_VECTOR_END]
    return first_point.float()


def start_and_end_node(map: torch.Tensor) -> torch.Tensor:
    """get start and end point of one lane marking side.

    :param map:
    :return:
    """
    indices = (
        torch.Tensor([c.FIRST_VECTOR_ONE_SIDE, c.LAST_VECTOR_ONE_SIDE])
        .long()
        .to(map.device)
    )
    subnodes_map = map.index_select(dim=c.FIRST_VALUE, index=indices)
    return subnodes_map


def approximate_arc_length(map: torch.Tensor, used_deltas: bool) -> torch.Tensor:
    """approximate the length ofone side lane marking.

    :param map:
    :param used_deltas:
    :return:
    """
    if used_deltas:
        one_side_deltas = map[c.TIMESTEPS_FOR_LENGTH, :, :, c.DELTAS]
    else:
        one_side_deltas = (
            map[c.TIMESTEPS_FOR_LENGTH, :, :, c.INDICES_VECTOR_END]
            - map[c.TIMESTEPS_FOR_LENGTH, :, :, c.INDICES_VECTOR_START]
        )
    arc_lengths = torch.sum(torch.norm(one_side_deltas, dim=c.LAST), dim=c.ZERO)
    return arc_lengths.unsqueeze(c.LAST_VALUE)


def generate_reference_pose(
    trajectory: np.array,
    last_input_timestep: int,
    yaw: Optional[float] = None,
    method_flag: int = 1,
) -> dict:
    """generate reference pose from trajectory with translation as vector of
    displacement and rotation as quaternion.

    :param trajectory: trajectory for which ref pose is computed
    :param last_input_timestep: time step of trajectory which contains reference pose
    :param yaw: optionally provide accurate yaw (computed numerically from trajectory otherwise)
    :param method_flag: method for yaw approximation
    :return: dict with reference pose
    """
    reference_pose = dict()
    reference_pose["translation"] = trajectory[
        last_input_timestep,
        c.INDICES_COORDINATES,
    ]
    if yaw is None:
        yaw = approximate_yaw(trajectory, last_input_timestep, method_flag)
    quaternion_yaw = quaternion_from_2d_rotation(yaw)
    reference_pose["rotation"] = list(quaternion_yaw)
    return reference_pose


def generate_reference_poses(
    trajectories: np.ndarray,
    input_frames: int,
    yaws: Optional[np.ndarray] = None,
) -> list:
    """generates a reference pose for each agent in the sample This is mainly
    used for online inference if agent centric representations of scene are
    generated.

    :param trajectories: dynamic agents (n_timesteps, n_agents, 3)
    :param yaws: yaw angles of agent (if None, compute numerically)
    :param input_frames: defines timestep of the pose that is used as reference
    :return:
    """
    if yaws is None:
        ref_poses = [
            generate_reference_pose(trajectories[:, i, :], input_frames)
            for i in range(trajectories.shape[1])
        ]

    else:
        ref_poses = [
            generate_reference_pose(
                trajectories[:, i, :],
                input_frames,
                yaws[input_frames, i],
            )
            for i in range(trajectories.shape[1])
        ]
    return ref_poses


def pad_sample_subnodes(sample: np.array, max_number_features: int) -> np.array:
    """padding of a single sample wth zeros.

    :param sample: sample to pad (vectors, batch, agents/ lane_nodes, parameters)
    :param max_number_features: max number of current batch along 3rd Dimension
    :return: padded sample
    """
    if sample.shape[2] < max_number_features:
        if len(sample.shape) == c.FOUR:
            padding_vector = np.zeros(
                (
                    sample.shape[c.ZERO],
                    sample.shape[c.ONE],
                    max_number_features - sample.shape[c.TWO],
                    sample.shape[c.THREE],
                ),
            )
        else:
            padding_vector = np.zeros(
                (
                    sample.shape[c.ZERO],
                    sample.shape[c.ONE],
                    max_number_features - sample.shape[c.TWO],
                ),
            )
        new_sample = np.concatenate((sample, padding_vector), axis=c.TWO)
    else:
        new_sample = sample
    return new_sample


def global_padding(samples: list, flag: str = None) -> np.array:
    """determine max number of agents in batch and pad acordingly.

    :param samples:
    :param flag:
    :return:
    """
    if flag == "lane_graph":
        number_features = [tensor.shape[c.FIRST_VALUE] for tensor in samples]
        max_number_features = max(number_features)
    number_subnodes = [tensor.shape[c.THIRD_VALUE] for tensor in samples]
    max_number_subnodes = max(number_subnodes)
    new_samples = []
    for sample in samples:
        if flag == "lane_graph":
            sample = global_repetative_padding(sample, max_number_features)
        sample = pad_sample_subnodes(sample, max_number_subnodes)
        new_samples.append(sample)
    new_samples = np.concatenate(new_samples, axis=c.BATCH_DIM_SEQUENCE)
    return new_samples


def global_repetative_padding(sample: np.ndarray, max_subnodes: int):
    """pads different maximas in nodes / lane graph subnode within batch.

    :param sample:
    :param max_subnodes:
    :return:
    """
    nbr_repeats = max_subnodes - sample.shape[c.FIRST_VALUE]
    if nbr_repeats > c.ZERO:
        repeated_sample = np.repeat(
            sample[-1:, :, :, :],
            nbr_repeats,
            axis=c.FIRST_VALUE,
        )
        sample = np.concatenate((sample, repeated_sample), axis=c.FIRST_VALUE)
    return sample


def relative_timestamps(timestamps: list, input_frames: int) -> np.array:
    """calculation of relative timestamps w.r.t. last input timestamps.

    :param timestamps:
    :param input_frames:
    :return:
    """
    timestamps = np.array(timestamps)
    relative_timesteps = timestamps - timestamps[input_frames]
    return relative_timesteps


def zero_padded_positions(sample: np.array, tracking_states: np.array) -> np.array:
    """padded trajectory points are shifted due to coordinate transformations
    in this function those data points are reset to zero to avoid large values
    even if they might get ignored.

    :param sample: data array which is updated
    :param tracking_states: masks which array entries should be kept
    :return: updated sample
    """
    tracking_states = np.repeat(
        tracking_states,
        sample.shape[c.LAST_VALUE],
        axis=c.LAST_VALUE,
    )
    sample = sample * tracking_states
    return sample


def prev_locations(trajectories: np.ndarray) -> np.ndarray:
    """compute start points for vector input.

    :param trajectories:
    :return:
    """
    check_2d_coordinates(trajectories)
    return trajectories[: c.LAST_VALUE, ...]


def calc_deltas(trajectories: np.ndarray) -> np.ndarray:
    """calculates deltas between trajectory points.

    :param trajectories:
    :return:
    """
    check_2d_coordinates(trajectories)
    deltas = trajectories[c.SECOND_VALUE :, ...] - trajectories[: c.LAST_VALUE, ...]
    return deltas


def check_2d_coordinates(coordinates: np.ndarray) -> bool:
    assert coordinates.shape[-1] == c.TWO, "2D coordinates have to be of shape [..., 2]"
    return True


def get_roi_indices_l2(coordinates: np.array, roi: int) -> np.ndarray:
    """returns indices of relative 2D points which L2 norm is within roi.

    :param coordinates:
    :param roi:
    :return:
    """
    distance = np.linalg.norm(coordinates, ord=c.TWO, axis=c.LAST)
    indices = np.argwhere(distance < roi)[:, c.FIRST_VALUE]
    return indices


def contains_deltas(cfgs: dict) -> bool:
    """checks if the delta input format is used.

    :param cfgs: configuration dict
    :return: flag whether delta input format is used
    """
    input_format = cfgs["config_params"]["vn_input_format"]
    use_deltas = input_format == c.DELTA_AND_END_POINT
    return use_deltas


def make_agents_centric_trajectories(
    trajectories: np.ndarray,
    ref_poses: list,
    deltas=False,
    origins=False,
) -> list:
    """generate a batch where each sample is an agent centric representation of
    the same scene.

    :param trajectories:  trajectories of each agent (x, y, tracking state)
    :param ref_poses: reference poses of each agent
    :return: list of agent centric trajectories
    """
    agent_centric_trajectories = []

    for ref_pose in ref_poses:
        agent_centric_trajectories.append(
            transform_in_ego_coordinates(
                trajectories,
                ref_pose,
                deltas=deltas,
                origins=origins,
            ),
        )
    return agent_centric_trajectories


def get_last_index_before_repetitive_padding(array: np.ndarray):
    """Get last index before values are repeated along first axis.

    :param array: input array with shape [n_values, ..., n_features]
    :return: index of last unqiue value
    """
    array_rev = np.flip(array, axis=0)
    diff = np.diff(array_rev, axis=0)
    unique = np.any(np.abs(diff) >= c.THRESHOLD_SMALL_NUMERICAL_VALUE, axis=-1)
    if not np.any(diff):
        return 0
    last_index = np.argmax(unique)
    return array.shape[0] - last_index - 1


def make_agent_centric_maps(
    map_: np.ndarray,
    ref_poses: list,
    region_of_interest: Optional[float],
    deltas=False,
    origins=False,
):
    """Transform map to reference poses.

    :param map_: original map
    :param ref_poses: reference poses relative to origin of map_
    :param region_of_interest: radius of roi (inf if None)
    :param deltas: map_ in deltas format
    :param origins: map_ in origins format
    :return:
    """
    maps = []
    for ref_pose in ref_poses:
        map_tmp = transform_in_ego_coordinates(
            map_,
            ref_pose,
            deltas=deltas,
            origins=origins,
        )
        if region_of_interest is not None:
            map_tmp = filter_map_region_of_interest_with_padding(
                map_tmp,
                region_of_interest,
            )
        maps.append(map_tmp)
    return maps


def collate_vector_scenes(data: list, online: bool = False) -> dict:
    """collate vector based scenes.

    :param online: switch to handle online kissaf data
    :param data: see README.md-> "Data formats" for details on the content.
    :return: collated sample data
    """
    all_inputs = []
    all_reference_poses = []
    all_lane_graphs = []
    all_target_indices = []
    all_paths = []
    all_lengths = []
    all_widths = []
    has_reference_path = "reference_path" in data[0]
    if has_reference_path:
        all_reference_paths = []
    if not online:
        all_targets = []
        all_loss_masks = []
    for sample in data:
        all_inputs.append(np.expand_dims(sample["x"], axis=c.BATCH_DIM_SEQUENCE))
        all_lane_graphs.append(np.expand_dims(sample["map"], axis=c.BATCH_DIM_SEQUENCE))
        all_reference_poses.append(sample["reference_pose"])
        all_target_indices.append(sample["index_target"])
        if not online:
            all_targets.append(np.expand_dims(sample["y"], axis=c.BATCH_DIM_SEQUENCE))
            all_loss_masks.append(
                np.expand_dims(sample["loss_masks"], axis=c.BATCH_DIM_SEQUENCE),
            )
        if "file_path" in sample.keys():
            all_paths.append(sample["file_path"])
        if "agent_length" in sample:
            all_lengths.append(sample["agent_length"])
        if "agent_width" in sample:
            all_widths.append(sample["agent_width"])
        if has_reference_path:
            all_reference_paths.append(
                np.expand_dims(sample["reference_path"][:, ..., :3], axis=c.ZERO),
            )

    all_inputs = global_padding(all_inputs)
    all_lane_graphs = global_padding(all_lane_graphs, flag="lane_graph")
    if not online:
        all_targets = global_padding(all_targets)
        target_futures = torch.from_numpy(all_targets)
        all_loss_masks = global_padding(all_loss_masks)
        batch = {
            "x": torch.from_numpy(all_inputs),
            "y": target_futures,
            "loss_masks": torch.from_numpy(all_loss_masks),
            "reference_pose": all_reference_poses,
            "map": torch.from_numpy(all_lane_graphs),
            "index_target": all_target_indices,
        }
        if len(all_paths) != 0:
            batch["file_path"] = all_paths
        if len(all_lengths) != 0:
            batch["agent_length"] = all_lengths
        if len(all_widths) != 0:
            batch["agent_width"] = all_widths
    else:
        batch = {
            "x": torch.from_numpy(all_inputs),
            "reference_pose": all_reference_poses,
            "map": torch.from_numpy(all_lane_graphs),
            "index_target": all_target_indices,
        }
    if has_reference_path:
        batch["reference_path"] = torch.from_numpy(
            np.concatenate(all_reference_paths, axis=c.ZERO),
        ).float()

    return batch


def clean_from_padding(sample: dict) -> dict:
    """cleans a sample from paddings based on input trajectories Assumption
    batch size is 1 In case of learning from map with no dynamic data all
    future would be removed Therefore if all samples would be removed it is
    tried to use the first future tracking state.

    :param sample:
    :type sample:
    :return:
    :rtype:
    """
    sample = copy.deepcopy(sample)
    last_input_tracking_states = sample["x"][c.LAST, c.FIRST, :, c.TRACKING_DIM_INPUT]
    no_padding_indices = torch.nonzero(last_input_tracking_states, as_tuple=True)[
        c.FIRST
    ]
    if no_padding_indices.nelement() == c.ZERO:
        first_output_tracking_states = sample["loss_masks"][c.FIRST, c.FIRST, :]
        no_padding_indices = torch.nonzero(first_output_tracking_states, as_tuple=True)[
            c.FIRST
        ]
    sample["x"] = sample["x"].index_select(
        dim=c.AGENT_DIM_INPUT_VEC,
        index=no_padding_indices,
    )
    sample["y"] = sample["y"].index_select(
        dim=c.AGENT_DIM_PREDICTION,
        index=no_padding_indices,
    )
    sample["loss_masks"] = sample["loss_masks"].index_select(
        dim=c.AGENT_DIM_PREDICTION,
        index=no_padding_indices,
    )
    return sample


def filter_map_region_of_interest_with_padding(
    map_array: np.ndarray,
    region_of_interest,
) -> np.ndarray:
    """filter all nodes outside roi and add repetitive padding.

    :param region_of_interest:
    :param map_array:
    :return:
    """

    filtered_map = []

    map_swapped = np.swapaxes(map_array, 0, 1)
    for subnode in map_swapped:
        is_inside = False
        first_valid_sample = None
        for sample in subnode:
            distance = np.sqrt(np.square(sample[c.ZERO]) + np.square(sample[c.ONE]))
            if distance < region_of_interest:
                first_valid_sample = sample
                break
        if first_valid_sample is not None:  # found at least one valid node
            for i in range(len(subnode)):
                sample = subnode[i, :]
                distance = np.sqrt(
                    np.square(sample[c.ZERO]) + np.square(sample[c.ONE]),
                )
                if distance > region_of_interest:
                    sample = first_valid_sample
                    subnode = np.delete(subnode, i, 0)
                    subnode = np.insert(subnode, 0, sample, axis=0)
                else:
                    is_inside = True
            if is_inside:
                filtered_map.append(np.expand_dims(subnode, axis=c.FIRST_VALUE))

    filtered_map = min_viable_map_sample(filtered_map)
    filtered_map = np.concatenate(filtered_map, axis=c.FIRST_VALUE)
    filtered_map = np.swapaxes(filtered_map, 0, 1)
    return filtered_map


def remove_consecutive_duplicates(array) -> Tuple[np.ndarray, np.ndarray]:
    """Remove rows from array with consecutive duplicates. E.g. [[1, 1], [1,
    1], [1, 0], [1, 1]] -> [[1, 1], [1, 0], [1, 1]]

    :param array: shape: [n_vertices, n_states]
    :return: array rows of consecutive duplicates removed.
    """
    if array.shape[0] <= 1:
        return array, np.ones([array.shape[0]], dtype=bool)

    diff = np.diff(array, axis=0, prepend=np.inf)
    duplicate_indices = ~np.all(diff == 0, axis=1)
    masked_arr = array[duplicate_indices, ...]
    return masked_arr, duplicate_indices


def crop_and_pad_with_max_distance(polylines, distance):
    """Crops polylines that exceed distance from origin. If polyline contains
    multiple sequences of valid vertices, the polyline is split and appended to
    polylines.

    :param polylines: array with shape [n_vertices, n_polylines, n_features]
    :param distance: max distance from origin
    :return: cropped, padded polylines
    """
    n_vertices, n_lines, _ = polylines.shape

    new_lines = []
    for line_idx in range(n_lines):
        line = polylines[:, line_idx, :]
        valid_indices = (
            np.sum(line[:, c.INDICES_COORDINATES] ** 2, axis=-1) <= distance**2
        )

        if not np.any(valid_indices):
            # Skip lines with no valid vertices
            continue

        # Find the indices where the line is split into different sequences
        diff = np.diff(valid_indices.astype(int), prepend=False)
        split_indices_start = np.where(diff == 1)[0]
        split_indices_end = np.where(diff == -1)[0]

        # Split the line into different sequences and add them as separate lines
        for split_idx_start, split_idx_end in zip(
            split_indices_start,
            split_indices_end,
        ):
            new_lines.append(line[split_idx_start:split_idx_end, :])

        # Add the last sub-line
        if (
            len(split_indices_end) == 0
            or split_indices_start[-1] > split_indices_end[-1]
        ):
            new_lines.append(line[split_indices_start[-1] :, :])

    new_lines = [l for l in new_lines if l.shape[0] > 1]
    if len(new_lines) == 0:
        return np.zeros([0, 0, polylines.shape[-1]])

    max_length = max(len(sub_line) for sub_line in new_lines)
    padded_lines = [
        np.pad(sub_line, ((0, max_length - len(sub_line)), (0, 0)), mode="edge")
        for sub_line in new_lines
    ]

    return np.array(padded_lines).transpose([1, 0, 2])


def shorten_gt_to_prediction_length(sample, prediction_length: int, fps: int):
    """Resizes "y" (ground_truth) and "loss_masks" if sample contains longer
    time horizon than required for model.

    :param sample: sample in Scene_Prediction format including "y" and "loss_masks".
    :param prediction_length: time horizon of model [s]
    :param fps: frames per second of sample data [1/s]
    :return: modified sample
    """
    prediction_steps = prediction_length * fps
    sample["y"] = sample["y"][:prediction_steps, ...]
    sample["loss_masks"] = sample["loss_masks"][:prediction_steps, ...]
    return sample

def get_gaussian_noise(mean: float = 0.0, std: float = 1.0, size=1):
    """Calculates Gaussian Noise for mean, std. Since we model a 2D Gaussian of
    orthogonal Coordinates, we need to adapt the std deviation with pythagoras.
    This results in normalizing std with sqrt(2).

    :param mean: mean of the distribution
    :param std: standard deviation
    :param size: Output shape of the noise
    :return: gaussian noise sampled to size
    """
    norm = np.sqrt(2)
    std = std / norm
    noise = np.random.normal(
        loc=mean,
        scale=std,
        size=size,
    )

    return noise


def remove_lane_ids(sample: dict) -> dict:
    """remove lane ids due to low signal to noise ratio based on different
    values range; this function can be removed after preprocessing is performed
    again.

    :param sample: vector sample
    :return: sample with corrected map entry
    """
    sample["map"] = sample["map"][..., : c.VN_INPUT_SIZE_MAP]
    return sample


def calc_origins_and_directions(
    vectors: np.ndarray,
    contains_deltas: bool = False,
    is_prediction: bool = False,
) -> Tuple[np.array, np.ndarray]:
    """handle different input formats by transforming vectors into origins and
    directions.

    :param vectors: map or trajectory data in vector format
    :param contains_deltas: if deltas are used to describe vectors
    :return: oring and direction arrays
    """
    if is_prediction:
        deltas = vectors[1:, :] - vectors[:-1, :]
        origins = vectors[:-1, :]
        directions = deltas
    else:
        deltas = get_deltas(vectors, contains_deltas)
        if contains_deltas:
            origins = vectors[..., c.INDICES_VECTOR_END] - deltas
            directions = deltas
        else:
            origins = vectors[..., c.INDICES_VECTOR_START]
            directions = deltas
    return origins, directions


def get_deltas(vectors: np.ndarray, contains_deltas) -> np.ndarray:
    """method to calculate poistion deltas of vector-format data.

    :param vectors: vector-format data
    :param contains_deltas: if data already contains poistion deltas the data at the
    specific index is returned
    :return: array of poition deltas
    """
    if contains_deltas:
        deltas = vectors[..., c.DELTAS]
    else:
        deltas = (
            vectors[..., c.INDICES_VECTOR_END] - vectors[..., c.INDICES_VECTOR_START]
        )
    return deltas


def first_sample_empty(sample: dict) -> bool:
    """Check if the first sample ist empty in order to prevent crashes in the
    forward method while updating plots.

    :param sample:
    :return: bool
    """
    sample = select_first_batch(sample)
    sample = clean_from_padding(sample)
    return is_empty(sample)


def skip_sample(batch: dict) -> bool:
    """deciding if a sample should be skipped in training. possible reasons a
    sample should not be used for training:

    - the sample is empty; does not contain any trajectory data
    :param batch:
    :return:
    """
    _skip = False
    if is_empty(batch):
        _skip = True
    if _skip and "file_path" in batch.keys():
        logging.debug("skiped following files:")
        for path in batch["file_path"]:
            logging.debug(path)
    return _skip


def is_empty(batch: dict) -> bool:
    """checks if a sample is empty."""
    if batch["x"].shape[c.AGENT_DIM_INPUT_VEC] == c.ZERO:
        return True
    else:
        return False


def select_first_batch(sample: dict) -> dict:
    """Select first sample from batched sample.

    :param sample: batched vector sample
    :return: sample with data from first sample from batch
    """
    return select_ith_batch(sample, c.FIRST_VALUE)


def select_ith_batch(sample: dict, batch_index: int) -> dict:
    """Select first batch from sample.

    :param sample: batched vector sample
    :param batch_index: index of batch that is extracted
    :return: sample with data from sample at batch_index
    """
    sample = copy.deepcopy(sample)
    batch_slice = slice(batch_index, batch_index + 1)
    sample["x"] = sample["x"][:, batch_slice, ...]
    sample["index_target"] = [sample["index_target"][batch_index]]
    if "y" in sample:
        sample["y"] = sample["y"][:, batch_slice, ...]
    sample["map"] = sample["map"][:, batch_slice, :, :]
    if "loss_masks" in sample:
        sample["loss_masks"] = sample["loss_masks"][:, batch_slice, ...]
    if "reference_path" in sample:
        sample["reference_path"] = sample["reference_path"][batch_slice, ...]
    return sample


def select_from_batch(sample: dict) -> dict:
    _batch_size = sample["x"].shape[c.BATCH_DIM_SEQUENCE]
    for i in range(_batch_size):
        _sample = select_ith_batch(sample, i)
        _sample = clean_from_padding(_sample)
        if not is_empty(_sample):
            return _sample
    raise NoTrajectoryDataException


def min_viable_map_sample(subnodes: list) -> list:
    """checks if a list of nodes is empty and adds the minimum viable sample.

    :param subnodes:
    :type subnodes:
    :return:
    :rtype:
    """
    if len(subnodes) == 0:
        min_sample = torch.zeros(c.MIN_SHAPE_LANENODE)
        subnodes.append(min_sample)
    return subnodes


def append_yaw_to_polyline(polylines: np.ndarray):
    """Add yaws to coordinates in last column of one or more polylines.

    :param polylines: polylines with shape [n_vertices, (n_polylines), 2]. dim n_polylines is optional
    :return: polylines with shape [n_vertices, (n_polylines), 3]
    """
    yaw = np.zeros_like(polylines[..., 1])
    yaw[:-1, ...] = calc_yaws_from_points(
        start_point=polylines[:-1, ...],
        end_point=polylines[1:, ...],
    )
    if yaw.shape[0] > 1:
        yaw[-1, ...] = yaw[-2, ...]
    return np.concatenate([polylines, yaw[..., np.newaxis]], axis=-1)


def append_velocity_to_polyline(polylines: np.ndarray, dt: float):
    vx = np.gradient(polylines[..., 0], dt, axis=0)
    vy = np.gradient(polylines[..., 1], dt, axis=0)
    v = np.sqrt(vx**2 + vy**2)
    return np.concatenate([polylines, v[..., np.newaxis]], axis=-1)


def get_lane_boundaries(lanes: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Extract right and left lane boundaries from lane vector.

    :param lanes: array with lanes, shape [2*n_vertices, n_lanes, states]
    :return:
    """
    left_lanes = []
    right_lanes = []
    for i_lane in range(lanes.shape[1]):
        lane = lanes[:, i_lane, :]
        n_vertices = int((get_last_index_before_repetitive_padding(lane) + 1) / 2)
        # assert 2 * n_vertices == lanes.shape[0]
        left_lanes.append(lane[:n_vertices, c.INDICES_COORDINATES])
        right_lanes.append(lane[n_vertices : 2 * n_vertices, c.INDICES_COORDINATES])

    return right_lanes, left_lanes


def append_yaw_to_polyline(polylines: np.ndarray):
    """Add yaws to coordinates in last column of one or more polylines.

    :param polylines: polylines with shape [n_vertices, (n_polylines), 2]. dim n_polylines is optional
    :return: polylines with shape [n_vertices, (n_polylines), 3]
    """
    yaw = np.zeros_like(polylines[..., 1])
    yaw[:-1, ...] = calc_yaws_from_points(
        start_point=polylines[:-1, ...],
        end_point=polylines[1:, ...],
    )
    if yaw.shape[0] > 1:
        yaw[-1, ...] = yaw[-2, ...]
    return np.concatenate([polylines, yaw[..., np.newaxis]], axis=-1)


def compute_center_lines(
    lanes: np.ndarray,
    add_yaw: bool = False,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Compute center lines for lanes in vector format.

    :param lanes:
    :param add_yaw: append yaw to center line vertices
    :return: center lines with shape [n_vertices, n_lanes, 2 or 3 (depending on add_yaw)]
    """
    right_lanes, left_lanes = get_lane_boundaries(lanes)
    center_lines = [
        0.5 * (right_lane + left_lane)
        for right_lane, left_lane in zip(right_lanes, left_lanes)
    ]

    if add_yaw:
        center_lines = [
            append_yaw_to_polyline(center_line) for center_line in center_lines
        ]

    max_len = max([len(c) for c in center_lines])
    center_lines_padded = np.stack(
        [
            np.pad(
                c,
                pad_width=[(0, max_len - c.shape[0]), (0, 0)],
                mode="edge",
            )
            for c in center_lines
        ],
        axis=1,
    )

    return center_lines, center_lines_padded


def cutoff_path_before_point(
    path: np.ndarray,
    point: np.ndarray,
    margin: int = 0,
) -> np.ndarray:
    """Remove vertices before point.

    :param path: path to cut off
    :param point: point before which is cut off
    :param margin: number of points to keep before point
    :return: cut-off path
    """
    current_location = np.argmin(np.linalg.norm(path[:, :2] - point[:2], axis=-1))
    return path[max(0, current_location - margin) :, :]


def resample_polyline_with_fixed_number_points(
    polyline,
    num_points=50,
    debug=False,
) -> np.ndarray:
    """Resample polyline with fixed distance after.

    :param polyline: list of vertices with shape [n_vertices, 2 or 3] (x,y and either with of without yaw)
    :param num_points: number of points after resampling
    :param debug: print debug info
    :return: resampled array [n_vertices, 2 or 3] (x,y and either with of without yaw)
    """
    if polyline.shape[0] <= 1:
        return polyline

    has_yaw = polyline.shape[-1] == 3
    polyline = polyline[..., :2]
    distances = np.cumsum(np.sqrt(np.sum(np.diff(polyline, axis=0) ** 2, axis=1)))
    distances = np.concatenate([[0], distances])
    total_distance = np.max(distances[np.isfinite(distances)])
    step = total_distance / (num_points - 1)

    resampled_points = []
    indices = []
    for i in range(num_points):
        distance = i * step
        index = np.searchsorted(distances, distance, side="right")
        indices.append((i, index))

        if index == 0:
            resampled_points.append(polyline[0])
        elif index == len(polyline):
            resampled_points.append(polyline[-1])
        else:
            t = (distance - distances[index - 1]) / (
                distances[index] - distances[index - 1]
            )
            point = (1 - t) * polyline[index - 1] + t * polyline[index]
            resampled_points.append(point)

    if debug:
        print("resample_polyline_with_fixed_number_points", indices)

    resampled_points = np.array(resampled_points)
    if has_yaw:
        resampled_points = append_yaw_to_polyline(resampled_points)

    return resampled_points


def extrapolate_polyline(polyline: np.ndarray, length: float):
    """Append linear extrapolation to polyline.

    :param polyline: polyline to extrapolate
    :param length: length of extrapolation
    :return: extrapolated polyline
    """
    delta_vec = polyline[-1, :2] - polyline[-2, :2]
    delta_len = np.linalg.norm(delta_vec)
    if delta_len == 0.0:
        raise ValueError(
            f"Cannot extrapolate polyline with delta distance = 0.0 at last index:\n:{polyline[-2:, :]}",
        )

    n_points = math.floor(length / delta_len)

    if n_points > 0:
        extension = np.stack(
            [
                np.linspace(delta_vec[0], n_points * delta_vec[0], n_points),
                np.linspace(delta_vec[1], n_points * delta_vec[1], n_points),
            ],
            axis=1,
        )
    else:
        extension = np.zeros([0, 2])

    if length % delta_len > 0.0:
        extension = np.append(
            extension,
            delta_vec[np.newaxis, :] * length / delta_len,
            axis=0,
        )

    extension += polyline[-1, :2]
    if polyline.shape[1] == 3:
        extension = np.concatenate([extension, polyline[-1, 2]], axis=-1)

    return np.concatenate([polyline, extension], axis=0)


def resample_polyline_with_fixed_distance(
    polyline,
    delta_distance=0.2,
    max_distance: Optional[float] = 350,
    debug=False,
) -> np.ndarray:
    """Resample polyline with fixed distance after.

    :param polyline: list of vertices with shape [n_vertices, 2 or 3] (x,y and either with of without yaw)
    :param delta_distance: distance between two vertices after resampling
    :param max_distance: fixes sizes of returned array using max. cumulative distance. If cum. distance is smaller, remaining entries are padded with np.inf
    :param debug: print debug info
    :return: resampled array [n_vertices, 2 or 3] (x,y and either with of without yaw)
    """
    if polyline.shape[0] <= 1:
        return polyline
    polyline = polyline[np.all(np.isfinite(polyline), axis=-1), :]

    has_yaw = polyline.shape[-1] == 3
    polyline = polyline[..., :2]

    distances = np.cumsum(np.sqrt(np.sum(np.diff(polyline, axis=0) ** 2, axis=1)))
    distances = np.insert(distances, 0, 0.0)

    total_distance = np.max(distances[np.isfinite(distances)])
    t_values = distances

    num_points = int(np.ceil(total_distance / delta_distance)) + 1
    resampled_distances = np.linspace(0, total_distance, num_points)

    if max_distance is not None:
        max_size = int(max_distance / delta_distance)
        num_points = min(num_points, max_size)
    else:
        max_size = num_points

    resampled_points = np.ones((max_size, 2)) * np.inf

    indices = []
    for i, resampled_distance in enumerate(resampled_distances):
        if i >= num_points:
            break
        index = np.searchsorted(t_values, resampled_distance, side="right")
        indices.append([i, index])

        if index == 0:
            resampled_points[i, :] = polyline[0]
        elif index == len(polyline):
            resampled_points[i] = polyline[-1]
        else:
            t_segment = (resampled_distance - t_values[index - 1]) / (
                t_values[index] - t_values[index - 1]
            )
            resampled_points[i] = (1 - t_segment) * polyline[
                index - 1
            ] + t_segment * polyline[index]

    if debug:
        print("t_va dist", t_values)
        print("resa dist", resampled_distances)
        print("resample_polyline_with_fixed_distance", indices)

    if has_yaw:
        resampled_points = append_yaw_to_polyline(resampled_points)

    return resampled_points


class NoTrajectoryDataException(Exception):
    def __init__(self, message="No element with trajectory data found."):
        self.message = message
        super().__init__(self.message)
