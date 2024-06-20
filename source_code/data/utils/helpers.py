# ------------------------------------------------------------------------------
# Copyright:    ZF Friedrichshafen AG
# Project:      KISSaF
# Created by:   ZF AI LAB SBR
# ------------------------------------------------------------------------------
import copy
import datetime
import logging
import math
import os
import pathlib
from pathlib import Path
from typing import Union, Optional

import numpy as np
import torch
from pyquaternion import Quaternion

from data import constants as c
from data.constants import tensor_or_array

def get_storage_path(opts):
    base_path = opts["general"]["log_dir"] + "/logs"
    return get_unique_path(base_path, opts, mkdir=opts["general"]["write_logs"])

def get_loss_masks(samples: list, number_frames: int) -> np.array:
    """function to obtain loss masks which handle insufficient observation of
    ground truth during training.

    :param samples:
    :param number_frames:
    :return: loss mask which contain ones and zeros and is element-wise multiplied in
    loss function
    """
    loss_masks = []
    for sample in samples:
        next_sample_annotation = sample["next"]
        if next_sample_annotation == "":
            curr_loss_mask = c.ZERO
        else:
            curr_loss_mask = c.ONE
        loss_masks.append(np.reshape(curr_loss_mask, c.SINGLE_LOSSMASK_SHAPE))
    if len(loss_masks) == c.ZERO:
        logging.debug("no values in mask")
    loss_masks = np.concatenate(loss_masks, axis=c.TIME_DIM_INPUT)
    if loss_masks.shape[c.TIME_DIM_INPUT] != number_frames:
        loss_masks = pad_with_zeros(sample=loss_masks, number_frames=number_frames)
    return loss_masks


def pad_with_zeros(sample: np.array, number_frames: int) -> np.array:
    """padding of samples with insufficient gt observation to make
    concatenation along agents possible.

    :param sample:  data sample which needs to be padded
    :param number_frames: number of frames the sample should contain
    :return: padded data sample
    """
    return np.concatenate(
        (
            sample,
            np.zeros(
                (
                    number_frames - sample.shape[c.TIME_DIM_INPUT],
                    sample.shape[c.INDEX_AGENT_DIM_INP],
                    sample.shape[c.INDEX_PARAMETER_DIM],
                ),
            ),
        ),
        axis=c.TIME_DIM_INPUT,
    )


def get_yaw(sample: dict) -> float:
    """function to get yaw angle

    :param sample:
    :return: yaw
    """
    return Quaternion(sample["rotation"]).yaw_pitch_roll[c.FIRST_VALUE]


def get_unique_path(parent: str, opts: dict, mkdir: bool = True) -> tuple:
    """function to make a directory based on a unique path for a model encoding
    info from time and configuration.

    :param parent: path to parent directory
    :param opts: training configurations
    :param mkdir: option to create a new directory at the new location
    :return: tuple
    """
    if c.TIMESTAMP is None:
        parent = Path(parent)
        c.TIMESTAMP = datetime.datetime.now()
        if opts["general"]["restore_model"]:
            id_string = parse_restore_path(opts)
        else:
            current_dt = c.TIMESTAMP.strftime("%y%m%d%H%M%S")
            num_modes = "_mm{}".format(opts["hyperparameters"]["num_modes"])
            id_string = current_dt + num_modes
        logger_path_candidate = parent / id_string
        folder_suffix_counter = 1
        if mkdir:
            while logger_path_candidate.exists():
                folder_suffix_counter += 1
                logger_path_candidate = parent / (
                    id_string + "_" + str(folder_suffix_counter)
                )
            logger_path_candidate.mkdir(parents=True, exist_ok=False)
            if folder_suffix_counter > 1:
                id_string = id_string + str(folder_suffix_counter)
        c.ID_STRING = id_string
        c.LOGGER_PATH_CANDIDATE = logger_path_candidate
    else:
        logger_path_candidate = c.LOGGER_PATH_CANDIDATE
        id_string = c.ID_STRING
    return logger_path_candidate, id_string


def parse_restore_path(opts):
    path = opts["general"]["model_path"]
    split_path = os.path.normpath(path).split(os.path.sep)
    return split_path[c.SECOND_LAST_VALUE]


def translate(inputs, translation_vector):
    inputs[..., c.INDEX_X_COORDINATE] += translation_vector[c.INDEX_X_COORDINATE]
    inputs[..., c.INDEX_Y_COORDINATE] += translation_vector[c.INDEX_Y_COORDINATE]
    return inputs


def translate_rotate_vector(inputs, translation, rotation_matrix, deltas: bool = False):
    """Translate then rotate input in vector representation (in place).

    :param inputs: input vectors, shape=[..., 4]
    :param translation: translation, shape=[2]
    :param rotation_matrix: rotation matrix, shape=[2,2]
    :param deltas: vectors are using deltas representation
    :return: transformed vector, shaped like inputs
    """
    inputs[..., 0:2] = translate(inputs[..., 0:2], translation)
    if deltas is False:
        inputs[..., 2:4] = translate(inputs[..., 2:4], translation)

    inputs[..., 0:2] = rotate_coordinates_with_matrix(inputs[..., 0:2], rotation_matrix)
    inputs[..., 2:4] = rotate_coordinates_with_matrix(inputs[..., 2:4], rotation_matrix)
    return inputs


def quaternion_from_2d_rotation(angle) -> Quaternion:
    """Return quaternion representation of rotation in 2D.

    :param angle: angle of the rotation
    :return: quaternion
    """
    return Quaternion(radians=angle, axis=[0, 0, 1.0])


def transform_yaw(yaw_angle: float, reference_quaternion: Quaternion) -> float:
    """transform a yaw with quaternion Changed multiplication to multiplication
    with inverse quaterion on 22/3/30.

    :param yaw_angle:
    :param reference_quaternion:
    :return:
    """
    yaw_as_quaternion = quaternion_from_2d_rotation(yaw_angle)
    transformed_quaternion = yaw_as_quaternion * reference_quaternion.inverse
    return transformed_quaternion.yaw_pitch_roll[c.FIRST_VALUE]


def calc_delta_yaw(yaw_1: float, yaw_2: float) -> float:
    """calculate delta between two yaw values.

    :param yaw_1: first angle
    :param yaw_2: second angle
    :return: delta between yaw values
    """
    yaw_1_quaternion = Quaternion(
        scalar=math.cos(yaw_1 / 2),
        vector=[0, 0, math.sin(yaw_1 / 2)],
    )
    delta_yaw = transform_yaw(yaw_2, yaw_1_quaternion)
    return delta_yaw


def v_calc_delta_yaw(yaw_1: np.ndarray, yaw_2: np.ndarray) -> np.ndarray:
    v_func = np.vectorize(calc_delta_yaw)
    d_yaws = v_func(yaw_1, yaw_2)
    return d_yaws


def transform_yaws(yaw_angles: np.array, reference_rotation: Quaternion) -> np.array:
    """vectorized version of transform_yaw function.

    :param yaw_angles:
    :param reference_rotation:
    :return:
    """
    v_transform_yaw = np.vectorize(transform_yaw)
    return v_transform_yaw(yaw_angles, reference_rotation)


def rotate_coordinates_with_matrix(
    inputs: Union[np.array, torch.tensor],
    transform_matrix: Union[np.array, torch.tensor],
) -> Union[np.array, torch.tensor]:
    if type(inputs) == torch.Tensor:
        outputs = torch.zeros_like(inputs)
    else:
        outputs = np.zeros_like(inputs)
    outputs[..., 0] = (
        transform_matrix[0, 0] * inputs[..., 0]
        + transform_matrix[0, 1] * inputs[..., 1]
    )
    outputs[..., 1] = (
        transform_matrix[1, 0] * inputs[..., 0]
        + transform_matrix[1, 1] * inputs[..., 1]
    )
    return outputs


def get_rotation_matrix(
    reference_pose: dict,
    as_torch_tensor: bool = False,
    to_world: bool = False,
) -> Union[np.array, torch.Tensor]:
    current_quaternion_world_to_ego = Quaternion(reference_pose["rotation"])
    if not to_world:
        current_quaternion_world_to_ego = current_quaternion_world_to_ego.inverse
    if as_torch_tensor:
        return torch.tensor(current_quaternion_world_to_ego.rotation_matrix)
    else:
        return current_quaternion_world_to_ego.rotation_matrix


def rotate_coordinate_pair(
    inputs: Union[np.array, torch.tensor],
    reference_pose: dict,
    to_world: bool = False,
) -> Union[np.array, torch.tensor]:
    if type(inputs) == torch.Tensor:
        as_tensor = True
    else:
        as_tensor = False
    rotation_matrix = get_rotation_matrix(
        reference_pose,
        as_torch_tensor=as_tensor,
        to_world=to_world,
    )
    return rotate_coordinates_with_matrix(
        inputs=inputs,
        transform_matrix=rotation_matrix,
    )


def translate_coordinates(
    inputs: np.ndarray,
    reference_pose: dict,
    inverse: bool = False,
    index_x: int = c.INDEX_X_COORDINATE,
    index_y: int = c.INDEX_Y_COORDINATE,
) -> np.ndarray:
    """translation of trajectory coordinates.

    :param inputs:
    :param reference_pose:
    :param inverse:
    :return:
    """
    if inverse:
        current_translation = -reference_pose["translation"]
    else:
        current_translation = reference_pose["translation"]
    inputs[..., index_x] = (
        inputs[..., index_x] - current_translation[c.INDEX_X_COORDINATE]
    )

    inputs[..., index_y] = (
        inputs[..., index_y] - current_translation[c.INDEX_Y_COORDINATE]
    )
    return inputs


def rotate_multiple_scene_coordinates(output, sample):
    for reference_pose, scene_index in zip(
        sample["ego_poses"],
        sample["scene_indices"],
    ):
        start = scene_index[c.FIRST_VALUE]
        end = scene_index[c.LAST_VALUE]
        output[:, :, start:end, :] = rotate_coordinate_pair(
            output[:, :, start:end, :],
            reference_pose=reference_pose,
        )
    return output


def world_to_ego_quaternion(reference_pose, planar=True):
    if planar:
        yaw_ref = Quaternion(reference_pose["rotation"]).yaw_pitch_roll[c.FIRST_VALUE]
        _quaternion = Quaternion(
            scalar=np.cos(yaw_ref / 2),
            vector=[0, 0, np.sin(yaw_ref / 2)],
        )
    else:
        _quaternion = Quaternion(reference_pose["rotation"])
    return _quaternion.inverse


def ego_to_world_quaterion(reference_pose, planar=True):
    if planar:
        yaw_ref = Quaternion(reference_pose["rotation"]).yaw_pitch_roll[0]
        _quaternion = Quaternion(
            scalar=np.cos(yaw_ref / 2),
            vector=[0, 0, np.sin(yaw_ref / 2)],
        )
    else:
        _quaternion = Quaternion(reference_pose["rotation"])
    return _quaternion


def delta_to_relative(
    fw_pass: Union[torch.Tensor, np.ndarray],
    sample: dict,
    targets: bool = False,
    except_targets: bool = False,
) -> Union[torch.Tensor, np.ndarray]:
    """convert deltas to coordinates relative to reference pose.

    :param fw_pass:
    :param sample:
    :param targets: generate prediction only for target agents
    :param except_targets: True if targets are already in relative coordinates
    :return:
    """
    device = fw_pass.device
    init_positions = sample["x"][c.LAST_VALUE, :, :, c.INDICES_COORDINATES]
    if targets:
        if except_targets:
            return fw_pass
        init_positions = get_targets(init_positions, sample["index_target"]).to(device)
    else:
        if except_targets:
            mask_targets = torch.zeros(
                init_positions.shape[:2],
                dtype=torch.bool,
                device=device,
            )
            mask_targets[
                list(range(len(sample["index_target"]))),
                sample["index_target"],
            ] = 1
            mask_targets = mask_targets.flatten()
            fw_pass_target = fw_pass[..., mask_targets, :]
        init_positions = reduce_batch_dim(init_positions).to(device)
    fw_pass[:, c.FIRST_VALUE, :, c.INDICES_COORDINATES] = fw_pass[
        :, c.FIRST_VALUE, :, c.INDICES_COORDINATES
    ] + init_positions.to(device)
    fw_pass[..., c.INDICES_COORDINATES] = fw_pass[..., c.INDICES_COORDINATES].cumsum(
        dim=c.TIME_DIM_PREDICTION,
    )
    if except_targets:
        fw_pass[..., mask_targets, :] = fw_pass_target
    return fw_pass


def transform_in_ego_coordinates(
    inputs: np.ndarray,
    reference_pose: dict,
    coordinates: bool = True,
    yaw: bool = False,
    deltas: bool = False,
    origins: bool = False,
    yaw_index: int = c.INDEX_YAW,
) -> np.ndarray:
    """transforms coordinates into coordinate system of a reference ego pose.

    :param inputs: np array containing coordinates
    :param reference_pose: ego_pose
    :param coordinates: set to true if input contains coordinates which should be
    transformed
    :param yaw: set to true if input contains yaw angles which should be transformed
    :param deltas: set to true if input contains delta coordinates which should be
    transformed
    :param origins: set to true if input contains origin coordinates which should be
    transformed
    :return: np array in ego coordinates
    """
    assert not (deltas and origins)
    inputs = inputs.copy()
    inputs_transformed = inputs.copy()

    # rotation (2D projection in plane orthogonal to z with z = 0)
    rotation_matrix = get_rotation_matrix(reference_pose=reference_pose)

    if coordinates:
        inputs = translate_coordinates(inputs=inputs, reference_pose=reference_pose)
        inputs_transformed[..., 0:2] = rotate_coordinates_with_matrix(
            inputs=inputs[..., 0:2],
            transform_matrix=rotation_matrix,
        )

    if deltas or origins:
        if origins:
            inputs = translate_coordinates(
                inputs=inputs,
                reference_pose=reference_pose,
                index_x=2,
                index_y=3,
            )

        inputs_transformed[..., 2:4] = rotate_coordinates_with_matrix(
            inputs=inputs[..., 2:4],
            transform_matrix=rotation_matrix,
        )
    if yaw:
        inputs_transformed[..., yaw_index] = transform_yaws(
            yaw_angles=inputs[..., yaw_index],
            reference_rotation=world_to_ego_quaternion(reference_pose),
        )

    return inputs_transformed


def transform_sample_to_ego_coordinates(
    sample: dict,
    reference_pose: dict,
    deltas: bool = False,
) -> dict:
    """Transform the whole sample in place to reference pose.

    :param sample: sample with x,y, map data that is transformed
    :param reference_pose: reference pose which is the origin after transformation
    :param deltas: vectors are using deltas representation
    :return: transformed sample
    """
    translation = -reference_pose["translation"]
    rotation_matrix = get_rotation_matrix(reference_pose=reference_pose)

    sample["x"] = translate_rotate_vector(
        sample["x"],
        translation,
        rotation_matrix,
        deltas,
    )
    sample["y"] = translate_rotate_vector(
        sample["y"],
        translation,
        rotation_matrix,
        deltas,
    )
    sample["map"] = translate_rotate_vector(
        sample["map"],
        translation,
        rotation_matrix,
        deltas,
    )

    if "reference_path" in sample:
        sample["reference_path"][..., :2] = translate(
            sample["reference_path"][..., :2],
            translation,
        )
        sample["reference_path"][..., :2] = rotate_coordinates_with_matrix(
            sample["reference_path"][..., :2],
            rotation_matrix,
        )
        if sample["reference_path"].shape[-1] >= 3:
            sample["reference_path"][..., 2] = transform_yaws(
                sample["reference_path"][..., 2],
                Quaternion(reference_pose["rotation"]),
            )

    return sample


def translate_rotate_sample(sample, translation: Optional[np.ndarray] = None, yaw=0.0):
    """Translate then rotate all elements in sample related to positions.

    :param sample:
    :param translation:
    :param yaw:
    :return:
    """
    if translation is None:
        translation = np.array([0.0, 0.0])
    reference_pose = {
        "translation": -translation,
        "rotation": quaternion_from_2d_rotation(-yaw),
    }

    return transform_sample_to_ego_coordinates(sample, reference_pose)


def agent_to_scene_centric(
    prediction: torch.Tensor,
    reference_pose: dict,
) -> torch.Tensor:
    """transform predixtion with ref pose into a scene centric formulation.

    :param prediction:
    :param reference_pose:
    :return:
    """
    prediction = prediction.clone()

    rotation_matrix = get_rotation_matrix(reference_pose=reference_pose, to_world=True)

    prediction[..., 0:2] = rotate_coordinates_with_matrix(
        inputs=prediction[..., 0:2],
        transform_matrix=rotation_matrix,
    )
    prediction = translate_coordinates(
        inputs=prediction,
        reference_pose=reference_pose,
        inverse=True,
    )
    return prediction


def agents_to_scene_centric(
    predictions: torch.Tensor,
    sample: dict,
) -> torch.Tensor:
    """transform agent centric prediction into scene centric prediction with
    the origin at the target agent.
    Note: Moved this functionality to online converter after 10.10.2023

    :param predictions: trajectories of agents in agent-centric coordinates
    :param sample: agent-centric sample
    :return: trajectories of agents in scene-centric coordinates
    """
    predictions = predictions.permute(c.TWO, c.ZERO, c.ONE, c.THREE)
    scene_centric_pred = []
    for prediction, ref_pose in zip(predictions, sample["reference_pose"]):
        if (
            "global_reference_pose" in sample
            and sample["global_reference_pose"] is not None
        ):
            ref_pose = rotate_translate_pose(ref_pose, sample["global_reference_pose"])
        transformed_prediction = agent_to_scene_centric(prediction, ref_pose)
        scene_centric_pred.append(transformed_prediction)
    scene_centric_pred = torch.stack(scene_centric_pred)
    scene_centric_pred = scene_centric_pred.permute(c.ONE, c.TWO, c.ZERO, c.THREE)
    return scene_centric_pred


def poses_normed_by_ego(sample: dict) -> dict:
    """normalise poses for a scene centric representation in online converter.

    :param sample: sample from Online converter
    :return:poses in ego coordinate system
    """
    poses = sample["reference_pose"]
    transformed_poses = []
    ego_index = sample["ego_index"]
    ref_pose = copy.deepcopy(poses[ego_index])
    for pose in poses:
        transformed_poses.append(transform_pose(pose=pose, reference_pose=ref_pose))
    return transformed_poses


def transform_pose(pose: dict, reference_pose: dict) -> dict:
    """transform pose into another coordinate system described by
    reference_pose. Note that pose and reference_pose need to be in the same
    coordinate system.

    :param pose: pose to transform. Consists of 2D Translation and Quaternion
    :param reference_pose: pose describing new coordinate system. Same structure as
    pose.
    :return: transformed pose
    """
    new_translation = np.subtract(pose["translation"], reference_pose["translation"])
    new_translation = rotate_coordinate_pair(new_translation, reference_pose)
    pose["translation"] = new_translation
    orientation = Quaternion(pose["rotation"])
    ref_orientation = Quaternion(reference_pose["rotation"])
    pose["rotation"] = list(orientation * ref_orientation.inverse)
    return pose


def rotate_translate_pose(pose: dict, pose_other: dict) -> dict:
    """transform pose into another coordinate system described by
    reference_pose. Note that pose and reference_pose need to be in the same
    coordinate system.

    :param pose: pose to transform. Consists of 2D Translation and Quaternion
    :param reference_pose: pose describing new coordinate system. Same structure as
    pose.
    :return: transformed pose
    """
    new_translation = rotate_coordinate_pair(
        pose["translation"],
        pose_other,
        to_world=True,
    )
    new_translation = new_translation + pose_other["translation"]

    pose["translation"] = new_translation
    orientation = Quaternion(pose["rotation"])
    orientation_other = Quaternion(pose_other["rotation"])
    pose["rotation"] = list(orientation * orientation_other)
    return pose


def check_for_roi(discrete_coordinates, grid_size) -> bool:
    """checking if discrete coordinate of agent is within region of interest
    defined by grid_size/pixel_per_meter.

    :param discrete_coordinates:
    :param grid_size:
    :return:
    """
    return (
        np.abs(discrete_coordinates[c.INDEX_X_COORDINATE] - (grid_size / 2))
        < (grid_size / 2)
    ) and (
        np.abs(discrete_coordinates[c.INDEX_Y_COORDINATE] - (grid_size / 2))
        < (grid_size / 2)
    )


def set_device_type(opts: dict):
    if opts["general"]["gpuIDs"]:
        device = torch.device("cuda:{}".format(opts["general"]["gpuIDs"]))
    else:
        device = torch.device("cpu")
    c.DEVICE = device
    return device


def init_events():
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    return start, end


def calculate_time(start, end):
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def load_label_values() -> dict:
    """
    loads the mapping between label number and data set keys
    Returns:

    """
    return c.LABEL_VALUES


def reduce_batch_dim(
    sequence: tensor_or_array,
) -> tensor_or_array:
    shape = sequence.shape
    if preds_with_mm_dim(shape):
        sequence = sequence.reshape(
            (shape[c.ZERO], shape[c.ONE], shape[c.TWO] * shape[c.THREE], shape[c.FOUR]),
        )
    elif mode_probabilities_or_global_graph(shape):
        sequence = sequence.reshape((shape[c.ZERO] * shape[c.ONE], shape[c.TWO]))
    else:
        sequence = sequence.reshape(
            (shape[c.ZERO], shape[c.ONE] * shape[c.TWO], shape[c.THREE]),
        )
    return sequence


def reduce_batch_dim_loss_mask(
    loss_mask: Union[np.array, torch.Tensor],
) -> Union[np.array, torch.Tensor]:
    """reduce batch dim in loss masks loss masks is a special case of 3 dim
    arrays where first dim is the time dim.

    :param loss_mask:
    :return:
    """
    shape = loss_mask.shape
    loss_mask = loss_mask.reshape(
        shape[c.FIRST_VALUE],
        shape[c.SECOND_VALUE] * shape[c.THIRD_VALUE],
    ) # t x (b p)
    return loss_mask


def preds_with_mm_dim(shape: list) -> bool:
    return len(shape) == 5


def mode_probabilities_or_global_graph(shape: list) -> bool:
    return len(shape) == 3


def get_fps(opts: dict) -> int:
    fps = c.FPS_WAYMO

    return fps


def gather_target_vectors(
    agent_encodings: torch.Tensor,
    joint_encoding: torch.Tensor,
    indices: list,
) -> torch.Tensor:
    """create an embedding for each target.

    :param agent_encodings:
    :param joint_encoding:
    :param indices:
    :return:
    """
    target_encodings = get_targets(agent_encodings, indices)
    target_encodings = torch.cat(
        [target_encodings, joint_encoding[..., 0]],
        c.LAST_VALUE,
    )
    return target_encodings


def get_targets(data: torch.Tensor, indices: list) -> torch.Tensor:
    batch_indices = [i for i in range(len(indices))]
    if sequence_of_vectors(data):
        data = torch.cat(
            [
                data[:, b, a, :].unsqueeze(c.SECOND_VALUE)
                for b, a in zip(batch_indices, indices)
            ],
            c.SECOND_VALUE,
        )
    else:
        data = torch.cat(
            [
                data[b, a, :].unsqueeze(c.FIRST_VALUE)
                for b, a in zip(batch_indices, indices)
            ],
        )
    return data


def get_targets_from_predictions(data: torch.Tensor, indices: list) -> torch.Tensor:
    """extract target trajectories from prediction.

    :param data: trajectories of all agents
    :param indices: indices of target agents
    :return: tensor of target predicted trajectories
    """
    batch_indices = [i for i in range(len(indices))]
    if sequence_of_vectors(data):
        output = torch.cat(
            [
                data[:, batch_index : batch_index + 1, agent_index : agent_index + 1, :]
                for batch_index, agent_index in zip(batch_indices, indices)
            ],
            c.SECOND_VALUE,
        )
    else:
        output = torch.cat(
            [
                data[
                    :,
                    :,
                    batch_index : batch_index + 1,
                    agent_index : agent_index + 1,
                    :,
                ]
                for batch_index, agent_index in zip(batch_indices, indices)
            ],
            c.THIRD_VALUE,
        )
    return output


def get_agents(data: torch.Tensor, indices: int) -> torch.Tensor:
    """extract agents from global graph.

    :param data:
    :param indices:
    :return:
    """
    if sequence_of_vectors(data):
        data = data[:, :, :indices, :]
    else:
        data = data[:, :indices, :]
    return data


def get_target_loss_masks(data: torch.Tensor, indices: list) -> torch.Tensor:
    data = data.permute(c.ONE, c.TWO, c.ZERO)
    data = get_targets(data, indices)
    data = data.permute(c.ONE, c.ZERO).unsqueeze(c.LAST_VALUE)
    return data


def sequence_of_vectors(data):
    return len(data.shape) == c.FOUR


def get_fps_for_dataset(dataset: int) -> int:
    """
    Args:
        dataset: Currently selected dataset
    Returns:
        fps: frames per second depending on the dataset
    """
    if c.WAYMO == dataset:
        fps = c.FPS_WAYMO
    else:
        raise ValueError("Invalid dataset value!")
    return fps


def get_output_format(opts: dict) -> int:
    loss_id = opts["general"]["loss"]
    return c.OUTPUT_PARAMS_COORDINATES


def int_with_check(number_to_convert: float) -> int:
    out = -1
    if number_to_convert % 1 == 0:
        out = int(number_to_convert)
    else:
        raise ValueError(
            "number needs to be an integer number but is {}".format(number_to_convert),
        )
    return out


def extract_relevant_info(sample: tuple, indices: list) -> np.ndarray:
    """Extracts information at element positions specified in indices.

    :param sample:
    :param indices:
    :return:
    """
    new_sample = np.array([sample[i] for i in indices])
    return np.expand_dims(new_sample, axis=0)


class Padder:
    def __init__(self, no_value_sample):
        self._no_value_sample = no_value_sample

    @property
    def all_timestamps(self) -> np.ndarray:
        return self._all_timestamps

    @all_timestamps.setter
    def all_timestamps(self, all_timestamps: np.ndarray):
        self._all_timestamps = all_timestamps

    @all_timestamps.deleter
    def all_timestamps(self):
        del self._all_timestamps

    def pad_trajectory(
        self,
        agent_timestamps: np.ndarray,
        trajectory: np.ndarray,
        no_value_sample_value=c.NO_VALUE_SAMPLE,
    ) -> np.ndarray:
        """Pad trajectory states at time stamps without data.

        :param agent_timestamps: time stamps of agent's trajectory
        :param trajectory: trajectory of agent
        :param no_value_sample_value: values to insert at missing time stamps
        :return: padded trajectory
        """
        for i, timestep in enumerate(self._all_timestamps):
            if (timestep != agent_timestamps).all():
                no_value_sample = np.expand_dims(
                    np.array(no_value_sample_value),
                    c.FIRST_VALUE,
                )
                trajectory = np.insert(
                    trajectory,
                    i,
                    no_value_sample,
                    axis=c.FIRST_VALUE,
                )
        return trajectory


def format_gt(sample: dict, opts: dict) -> torch.Tensor:
    if opts["config_params"]["predict_target"]:
        ground_truth = get_targets(sample["y"], sample["index_target"])
    else:
        ground_truth = reduce_batch_dim(sample["y"])
    return ground_truth


def format_loss_masks(sample: dict, opts: dict) -> torch.Tensor:
    if opts["config_params"]["predict_target"]:
        loss_masks = get_target_loss_masks(sample["loss_masks"], sample["index_target"])
    else:
        loss_masks = reduce_batch_dim_loss_mask(sample["loss_masks"]).unsqueeze(
            c.LAST_VALUE,
        )
    return loss_masks


def adapt_config_to_store(config: dict, storage_path: pathlib.Path) -> dict:
    """change configs so that they can load model etc without further manual
    adapts so far this only stores the path of where the model is saved.

    :param config: configuration to adapt
    :param storage_path: path were model is stored
    :return: adapted config
    """
    if config["general"]["save_model"] and not config["general"]["restore_model"]:
        config["general"]["model_path"] = str(storage_path.resolve()) + "/model.pth"
    return config


def contains_nan(self, x):
    is_nan = torch.isnan(x)
    ok = torch.all(not is_nan)
    return not ok


def is_valid_vector_sample(sample: dict) -> bool:
    """Checks whether sample has correct interface.

    :param sample:
    :return: True if valid
    """
    if len(c.SAMPLE_KEYS_REQUIRED - set(sample.keys())) > 0:
        return False

    return True


def get_number_agents_in_sample(sample: dict) -> torch.Tensor:
    """Returns the number of agents in the (batched) sample(s).

    :param sample: scene prediction sample (see Readme)
    :return: int (or list of ints for batched samples)
    """
    return torch.sum(torch.any(sample["loss_masks"], dim=0), dim=-1).tolist()

def backtranslate_coordinates(inputs, reference_pose):
    current_translation = -1*reference_pose['translation']
    inputs[..., c.INDEX_X_COORDINATE] = inputs[..., c.INDEX_X_COORDINATE] \
                                                - current_translation[c.INDEX_X_COORDINATE]

    inputs[..., c.INDEX_Y_COORDINATE] = inputs[..., c.INDEX_Y_COORDINATE] \
                                                - current_translation[c.INDEX_Y_COORDINATE]
    return inputs


# ------------------------------------------------------------------------------
# Copyright:    Technical University Dortmund
# Project:      KISSaF
# Created by:   Institute of Control Theory and System Engineering
# ------------------------------------------------------------------------------

def transform_heading_in_ego_coordinates(heading_world_frame: np.array, reference_pose: dict):
    heading_ego_frame = transform_yaws(
        heading_world_frame,
        Quaternion(reference_pose["rotation"]),
    )

    return heading_ego_frame


