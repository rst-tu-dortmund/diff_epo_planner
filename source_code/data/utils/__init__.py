# ------------------------------------------------------------------------------
# Copyright:    ZF Friedrichshafen AG
# Project:      KISSaF
# Created by:   ZF AI LAB SBR
# ------------------------------------------------------------------------------
import logging
import pathlib
from pathlib import Path
import pickle
from abc import ABC, abstractmethod
from typing import Callable, Union, Optional

import data.constants as c
import data.utils.vector_utils as duv
import numpy as np

from data.utils.helpers import (
    Padder,
    get_targets,
    int_with_check,
    transform_in_ego_coordinates,
)
from torch.utils.data import Dataset

class VectorData(Dataset, ABC):
    """Abstract Class for VectorData."""

    _first_output: Optional[int] = None
    _last_input_timestep: int = None
    _all_frames: int = None
    _first_input: int = None
    _first_not_output: int = None
    input_format: Optional[int] = None
    additional_positional_info: Callable = None

    def __init__(
        self,
        cfgs: dict,
        dir: Optional[Union[str, Path]],
        fps: int,
        val: bool = False,
        *args,
        **kwargs,
    ):

        """
        :param cfgs: configuration
        :param dir: directory of data samples
        :param fps: frames per second in data
        :param val: eval mode or not
        """
        super(VectorData, self).__init__()
        self.cfgs = cfgs
        self.fps = fps
        c.DELTA_TIME_STEP = 1 / fps
        self._target_agent_number = c.NO_VALUE
        self.preprocessed: bool = cfgs["config_params"]["preprocess"]
        self._input_frames = (
            int_with_check(
                self.cfgs["hyperparameters"]["observation_length"] * self.fps,
            )
            - c.ONE
        )

        self.init_time_indices(self._input_frames)
        self.in_target_agent_frame = True
        self.data_path: Optional[Path] = None
        if dir:
            self.data_path = pathlib.Path(dir)

        self.padder = Padder(c.NO_VALUE_SAMPLE)
        self.epoch = 0
        self.pretrain_epochs = self.cfgs["hyperparameters"]["pretrain_epochs"]
        self.map_pretraining = False
        self.val = val
        if self.val:
            self.map_pretraining = False
        self.set_input_format()
        self.augmentations = self.init_data_augmentations()

    def init_time_indices(self, first_output: int) -> None:
        """Initialize time indices relevant for data extraction.

        :param first_output: time step of first output
        :return:  None
        """
        self._first_output = first_output
        # since x vectors ate constructed from x+1 timestep the correct way to
        # calulcate the last_input_timestep_index would be
        # x + 1 (match first output TIMESTEP index) - 1 (match LAST INPUT timestep) = x
        self._last_input_timestep = self._first_output
        self._all_frames = int_with_check(
            self._input_frames
            + self.cfgs["hyperparameters"]["prediction_length"] * self.fps,
        )
        self._first_input = self._first_output - self._input_frames
        self._first_not_output = (
            self._first_output - self._input_frames + self._all_frames
        )

    def init_data_augmentations(self):
        """Initialize data augmentation techniques from the configs in the
        order of their appearance.

        :return:
        """
        augmentations = []
        if "augmentation" not in self.cfgs:
            return augmentations

        for augmentation_type, kwargs in self.cfgs["augmentation"].items():
            if kwargs["active"] is True and (not self.val or kwargs["in_validation"]):
                augmentations.append(
                    VectorDataAugmentation.from_config(
                        augmentation_type,
                        self,
                        self.cfgs,
                    ),
                )

        return augmentations

    def apply_augmentations(self, sample):
        """Apply augmentations sequentially to sample.

        :param sample:
        :return:
        """
        for augmentation in self.augmentations:
            sample = augmentation(sample)

        return sample

    @property
    def all_timestamps(self) -> np.ndarray:
        return self._all_timestamps

    @all_timestamps.setter
    def all_timestamps(self, all_timestamps):
        self._all_timestamps = all_timestamps
        self.padder.all_timestamps = all_timestamps

    @all_timestamps.deleter
    def all_timestamps(self):
        del self.padder.all_timestamps
        del self._all_timestamps

    def set_input_format(self):
        self.input_format = self.cfgs["config_params"]["vn_input_format"]
        if self.cfgs["config_params"]["vn_input_format"] == c.START_AND_END_POINT:
            self.additional_positional_info = duv.prev_locations
        elif self.cfgs["config_params"]["vn_input_format"] == c.DELTA_AND_END_POINT:
            self.additional_positional_info = duv.calc_deltas
        else:
            raise ValueError(
                "{} not assigned to input format".format(
                    self.cfgs["config_params"]["vn_input_format"],
                ),
            )

    def process_trajectories(self, trajectories: np.ndarray) -> np.ndarray:
        """processes trajectories to required vectorformat.

        :param trajectories: trajectories in original format
        :return: trajectories in vectorized format (x and y coordintes of vector end
        points, additional spatial information (either coorinate delta or vecotr
        start point), timestamps of both)
        """
        nbr_agents = trajectories.shape[c.AGENT_DIM_WIP]
        additional_positional_params = self.additional_positional_info(
            trajectories[:, :, c.INDICES_COORDINATES],
        )
        tracking_states = get_tracking_states(trajectories)
        sample = np.concatenate(
            (
                trajectories[c.ONE :, :, c.INDICES_COORDINATES],
                additional_positional_params,
            ),
            axis=c.LAST_VALUE,
        )
        sample = duv.zero_padded_positions(sample, tracking_states)
        timestamp_features = self.get_relative_timestamp_features()
        timestamp_features = np.repeat(
            timestamp_features,
            nbr_agents,
            axis=c.AGENT_DIM_WIP,
        )
        sample = np.concatenate((sample, timestamp_features), axis=c.LAST_VALUE)
        return sample

    def reset_sample_dependent_values(self):
        """
        resets sample dependent mamber values
        :return:
        """
        self._target_agent_number = c.NO_VALUE

    def get_relative_timestamp_features(self):
        """
        Calulates timesteps relative to last input timestamp
        :return:
        """
        try:
            self.all_timestamps
        except NameError:
            print("No timestamps are assigned!")
        else:
            rel_timestamps = duv.relative_timestamps(
                self.all_timestamps,
                self._last_input_timestep,
            )
            rel_timestamps = np.expand_dims(rel_timestamps, axis=c.LAST_VALUE)
            features = np.concatenate(
                (rel_timestamps[c.ONE :, ...], rel_timestamps[: c.LAST_VALUE, ...]),
                axis=c.LAST_VALUE,
            )
            features = np.expand_dims(features, axis=c.AGENT_DIM_WIP)
            return features

    @abstractmethod
    def append_lane_graph(self, *args):
        pass

    def trajectory_to_sample(
        self,
        trajectories: np.ndarray,
        filter_with_roi: bool = True,
        yaws: Optional[np.ndarray] = None,
    ) -> dict:
        """converts the trajectory data to serve the defined interface between
        data and model map data is appended in an additional function.

        :param trajectories: original trajectory data (x, y, tracking_state)
        :param filter_with_roi: flag whether states should be filtered when outside region of interest
        :param yaws: yaw trajectories of all agents, shape: [time_steps, agents, 1]
        :return: formatted sample
        """
        ref_pose = duv.generate_reference_pose(
            trajectory=trajectories[:, self._target_agent_number, :],
            last_input_timestep=self._last_input_timestep,
            yaw=yaws[self._last_input_timestep, self._target_agent_number, :]
            if yaws is not None
            else None,
            method_flag=c.TWO,
        )
        trajectories = transform_in_ego_coordinates(trajectories, ref_pose)
        if yaws is not None:
            yaws -= yaws[self._last_input_timestep, self._target_agent_number, :]

        if filter_with_roi:
            trajectories = self.filter_with_roi(
                trajectories=trajectories,
                roi=self.cfgs["hyperparameters"]["region_of_interest"],
                input_frames=self._last_input_timestep,
            )
        vectors = self.process_trajectories(trajectories)
        sample = self.trajectories_and_vectors_to_input(trajectories, vectors)
        target = vectors[self._first_output : self._first_not_output, :, :]
        loss_masks = trajectories[
            self._first_output : self._first_not_output,
            :,
            c.INDICES_TRACKING_STATE_WIP,
        ]
        sample.update(
            {
                "y": target,
                "loss_masks": loss_masks,
                "reference_pose": ref_pose,
            },
        )
        if yaws is not None:
            sample["yaws"] = yaws[self._first_input : self._first_output, :, :]

        return sample

    def trajectories_to_inputs(
        self,
        trajectories: np.ndarray,
        filter_with_roi: bool = True,
    ) -> dict:
        """convert trajectories to input parts of the sample for online
        inference.

        :param trajectories: trajectory data (x, y, tracking_state)
        :param filter_with_roi: flag for filtering trajectory with region of interest
        :return: sample with trajectory input data in vectorformat (key: "x") and
        target index (key: "index_target")
        """
        if filter_with_roi:
            trajectories = self.filter_with_roi(
                trajectories=trajectories,
                roi=self.cfgs["hyperparameters"]["region_of_interest"],
                input_frames=self._last_input_timestep,
            )
        vectors = self.process_trajectories(trajectories)
        sample = self.trajectories_and_vectors_to_input(trajectories, vectors)
        return sample

    def trajectories_and_vectors_to_input(
        self,
        trajectories: np.ndarray,
        vectors: np.ndarray,
    ):
        """process trajectories and vectors to sample input format.

        :param trajectories: trajectory data (x, y, tracking_state)
        :param vectors: vectorized format (x and y coordintes of vector end
        points, additional spatial information (either coorinate delta or vecotr
        start point), timestamps of both)
        :return: sample with trajectory input data in vectorformat (key: "x") and
        target index (key: "index_target")
        """
        tracking_states = get_tracking_states(trajectories)
        tracking_states = tracking_states[self._first_input : self._first_output, :, :]
        sample_input = vectors[self._first_input : self._first_output, :, :]
        input = np.concatenate((sample_input, tracking_states), axis=c.LAST_VALUE)
        sample = {"x": input, "index_target": self._target_agent_number}
        return sample

    @staticmethod
    def _filter_with_roi(trajectories: np.ndarray, roi: int, input_frames: int):
        """filter data in trajectory format with roi.

        Args:
            trajectories: dynamic data in trajectory frame
            roi: region of interest within which agents are considered
            input_frames: number of input coordinates

        Returns:
        """
        last_input_sample = input_frames
        last_input_coordinates = trajectories[
            last_input_sample, :, c.INDICES_COORDINATES
        ]
        indices = duv.get_roi_indices_l2(last_input_coordinates, roi)
        trajectories = np.take(trajectories, indices=indices, axis=c.AGENT_DIM_INPUT)
        return trajectories, indices

    def filter_with_roi(
        self,
        trajectories: np.ndarray,
        roi: int,
        input_frames: int,
    ) -> np.ndarray:
        """filter agents based on their relative coordintates and region of
        interest.

        :param input_frames:
        :param trajectories: agents data with relative coordinates in first two entries
        of last dimension
        :param roi: region of interest radius
        :param last_input_sample: number of samples in input
        :return:
        """
        trajectories, indices = self._filter_with_roi(trajectories, roi, input_frames)
        self.update_target_agent_number(indices)
        return trajectories

    def update_target_agent_number(self, indices: np.ndarray):
        """Adapt target agent number with rejected agent indices.

        :param indices:
        :return:
        """
        new_target_agent_number = 0
        for index in indices:
            if index < self._target_agent_number:
                new_target_agent_number += 1
        self._target_agent_number = new_target_agent_number

    def format_dynamic_info(self, sample: dict) -> dict:
        """removes dynamic information in pretraining step to force learning
        from map data.

        :param sample:
        :return:
        """
        if (
            self.epoch < self.pretrain_epochs
            and self.map_pretraining == c.PRETRAIN_ON_MAP
        ):
            sample["x"] = np.zeros_like(sample["x"])
        elif (
            self.epoch < self.pretrain_epochs
            and self.map_pretraining == c.PRETRAIN_MAP_AND_TARGET
        ):
            sample = filter_target(sample)
        return sample

    @property
    @abstractmethod
    def all_files(self):
        """
        :return: list of files from all domains that can be used within a dataset
        """
        pass

    @staticmethod
    def get_intersection(data, valid_samples):
        return list(set(data) & set(valid_samples))

    def compute_file_list(self, binary_decision: Callable):
        """recompute files that are used whithin this dataset, if the files are
        filtered by domain affiliation.

        :param binary_decision:
        :return:
        """
        files = []
        for file in self.all_files:
            with open(file, "rb") as current_sample_handle:
                current_sample = pickle.load(current_sample_handle)
                if binary_decision(current_sample):
                    files.append(file)
        return files

    def list_domain_files(self, domain: str, search_saved_list: bool = True) -> list:
        """method that computes which files belong to a domain if this function
        was called before by default it will reuse the computed list, which ist
        stored in a pkl file.

        :param domain: data domain
        :param search_saved_list: if false the files will be recomputed, even if pkl
        with file names is available
        :return: paths to files
        """
        file_name = domain + ".pkl"
        path_data_list = self.data_path / file_name
        if path_data_list.exists() and search_saved_list:
            msg = f"data path {path_data_list} exists!"
            logging.info(msg)
            print(msg)
            with open(path_data_list, "rb") as handle:
                sample_paths = pickle.load(handle)
        else:
            msg = f"data path {path_data_list} doesnt exist!"
            logging.info(msg)
            print(msg)
            function = self.map_binary_domain_decisions(domain)
            sample_paths = self.compute_file_list(binary_decision=function)
            with open(path_data_list, "wb") as handle:
                pickle.dump(sample_paths, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return sample_paths

    @abstractmethod
    def map_binary_domain_decisions(self, key: str) -> Callable:
        """method that is used to map domain assignments to specific keys.

        :param key: domain name
        :return: function handle to decide if a sample is from a specific domain
        """
        pass


def filter_target(sample: dict) -> dict:
    """remove all dynamic data except target agent.

    :param sample:
    :return:
    """
    target = sample["index_target"]
    target_x = sample["x"][:, [target], :]
    target_y = sample["y"][:, [target], :]
    target_loss_mask = sample["loss_masks"][:, [target]]
    sample["index_target"] = 0
    sample["x"] = target_x
    sample["y"] = target_y
    sample["loss_masks"] = target_loss_mask
    return sample


def get_tracking_states(trajectories: np.ndarray) -> np.ndarray:
    return (
        trajectories[c.ONE :, :, [c.INDICES_TRACKING_STATE_WIP]]
        * trajectories[: c.LAST_VALUE, :, [c.INDICES_TRACKING_STATE_WIP]]
    )


def tail_in_strings(full_path: pathlib.Path, strings: list) -> bool:
    """compares if the last two levels of the path are in the strings list, as
    the samples in valid samples of kissaf data are specified by their two last
    levels.

    example string in strings from kissaf valid_samples.npy:
    KISSaF_2022_12_02_045659_31-45_rec_2023_09_02_02_41_09/2023_09_02_02_44_27_541.p

    :param full_path:
    :param strings:
    :return: interstection of strings that match
    """
    candidate = "/".join(pathlib.Path(full_path).parts[-2:])
    return candidate in strings
