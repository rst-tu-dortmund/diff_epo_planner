# ------------------------------------------------------------------------------
# Copyright:    Technical University Dortmund
# Project:      KISSaF
# Created by:   Institute of Control Theory and System Engineering

# Origin: This code is adapted from https://github.com/MCZhi/DIPP,
# Differentiable Integrated Motion Prediction and Planning with Learnable Cost Function for Autonomous Driving
# Zhiyu Huang, Haochen Liu, Jingda Wu, Chen Lv AutoMan Research Lab, Nanyang Technological University
# ------------------------------------------------------------------------------

import random
from pathlib import Path
import sys


from data.utils.helpers import transform_yaws, world_to_ego_quaternion

sys.path.append(str(Path(__file__).parent.parent))

import glob
import os
import traceback
import argparse
from pathlib import Path
from typing import List

from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pyquaternion import Quaternion
from waymo_open_dataset.protos import scenario_pb2
from multiprocessing import Pool
import tensorflow as tf
import matplotlib as mpl

import data.constants as c
from data.maps.maps_utils import format_lane_array
from data.utils import get_tracking_states, transform_in_ego_coordinates
from data.configs import load_configs
from data.utils.waymo import WaymoData
from data.utils.helpers import transform_heading_in_ego_coordinates
from external.DIPP.waymo_utils import *

# Disable all GPUS
tf.config.set_visible_devices([], 'GPU')


class WaymoPreprocessing(object):
    def __init__(self, files: List[str], config: dict):
        """
        Convert files from Waymo Open Dataset to VectorData format for Scene Prediction.
        :param files: list of waymo files
        :param config: scene prediction config
        """
        self.config = config
        self.fps = 1 / 10
        self.region_of_interest = 200
        self.search_extension_factor = 0.6 # factor for which depth first searched of reference lanes is performed -> factor*region_of_interest
        self.convert_crosswalks = False
        self.lane_distance_threshold = 150
        self.n_neighbors_max = 1
        self.n_lanes_max = 30
        self.interactive_split = True
        self.delta_steps = 50  # default value from Nils
        self.hist_len = 10
        self.future_len = 80
        self.data_files = files
        self.n_points_lane = c.POINT_PER_SUBGRAPH
        self._current_lanes = {}
        self._current_roads = {}
        self._current_stop_signs = {}
        self._current_crosswalks = {}
        self._current_speed_bumps = {}
        self._current_traffic_signals = {}
        self._current_driveways = {}
        self.max_coord_x = -10000
        self.min_coord_x = 10000
        self.max_coord_y = -10000
        self.min_coord_y = 10000

    def _parse_map(self, map_features, dynamic_map_states) -> None:
        """
        parse map from Waymo protobuf format
        :param map_features: map data in protobuf format
        :param dynamic_map_states: dynamic map data in protobuf format
        :return: None
        """
        self._current_lanes = {}
        self._current_roads = {}
        self._current_stop_signs = {}
        self._current_crosswalks = {}
        self._current_speed_bumps = {}

        # static map features
        for map in map_features:
            map_type = map.WhichOneof("feature_data")
            map_id = map.id
            try:
                map = getattr(map, map_type)
            except TypeError:
                continue

            if map_type == 'lane':
                self._current_lanes[map_id] = map
            elif map_type == 'road_line' or map_type == 'road_edge':
                self._current_roads[map_id] = map
            elif map_type == 'stop_sign':
                self._current_stop_signs[map_id] = map
            elif map_type == 'crosswalk':
                self._current_crosswalks[map_id] = map
            elif map_type == 'speed_bump':
                self._current_speed_bumps[map_id] = map
            elif map_type == 'driveway':
                self._current_driveways[map_id] = map
            else:
                # print(map_type)
                raise TypeError

        # dynamic map features
        self._current_traffic_signals = dynamic_map_states

    def _map_process(self,
                     traj: np.ndarray,
                     timestep: int,
                     other_trajs: List[np.ndarray] = None,
                     type=None,
                     ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract relevant map data in the surrounding of an agent's trajectory
        :param traj: trajectory for which map data is extracted
        :param timestep: initial time step of trajectory
        :param other_trajs: trajectories of other agents
        :param type: type of the agent
        :return: vectorized map and crosswalk
        """
        vectorized_map = np.zeros(shape=(self.n_lanes_max, self.n_points_lane, 17))
        agent_type = int(traj[-1][-1]) if type is None else type

        # get all lane polylines
        lane_polylines = get_polylines(self._current_lanes)

        # get all road lines and edges polylines
        road_polylines = get_polylines(self._current_roads)

        # find current lanes for the agent
        ref_lane_ids = find_reference_lanes(agent_type,
                                            traj,
                                            lane_polylines,
                                            self.search_extension_factor * self.region_of_interest,
                                            )
        if other_trajs is not None:
            for ii, other_traj in enumerate(other_trajs):
                agent_type_tmp = int(traj[-1][-1]) if type is None else type
                ref_lane_ids.update(find_reference_lanes(agent_type_tmp,
                                                         other_traj,
                                                         lane_polylines,
                                                         self.search_extension_factor * self.region_of_interest,
                                                         visited_lane_ids=ref_lane_ids
                                                         )
                                    )

        # find candidate lanes
        ref_lanes = []

        # get current lane's forward lanes
        for curr_lane, start in ref_lane_ids.items():
            candidate = depth_first_search(curr_lane, self._current_lanes,
                                           dist=lane_polylines[curr_lane][start:].shape[0],
                                           threshold=self.lane_distance_threshold)
            ref_lanes.extend(candidate)

        if agent_type != 2:
            # find current lanes' left and right lanes
            neighbor_lane_ids = find_neighbor_lanes(ref_lane_ids, traj, self._current_lanes, lane_polylines)

            # get neighbor lane's forward lanes
            for neighbor_lane, start in neighbor_lane_ids.items():
                candidate = depth_first_search(neighbor_lane, self._current_lanes,
                                               dist=lane_polylines[neighbor_lane][start:].shape[0],
                                               threshold=self.lane_distance_threshold)
                ref_lanes.extend(candidate)

            # update reference lane ids
            ref_lane_ids.update(neighbor_lane_ids)

        # get traffic light controlled lanes and stop sign controlled lanes
        traffic_light_lanes = {}
        stop_sign_lanes = []

        for signal in self._current_traffic_signals[timestep].lane_states:
            traffic_light_lanes[signal.lane] = (signal.state, signal.stop_point.x, signal.stop_point.y)
            for lane in self._current_lanes[signal.lane].entry_lanes:
                traffic_light_lanes[lane] = (signal.state, signal.stop_point.x, signal.stop_point.y)

        for i, sign in self._current_stop_signs.items():
            stop_sign_lanes.extend(sign.lane)

        # add lanes to the array
        added_lanes = 0
        for i, s_lane in enumerate(ref_lanes):
            added_points = 0
            if i >= self.n_lanes_max:
                break

            # create a data cache
            cache_lane = np.zeros(shape=(300, 17))

            for lane in s_lane:
                curr_index = ref_lane_ids[lane] if lane in ref_lane_ids else 0
                self_line = lane_polylines[lane][curr_index:]

                if added_points >= 300:  # max 150 meters (300 road points)
                    break

                # add info to the array
                for point in self_line:
                    # self_point and type
                    cache_lane[added_points, 0:3] = point
                    cache_lane[added_points, 10] = self._current_lanes[lane].type

                    # left_boundary_point and type
                    for left_boundary in self._current_lanes[lane].left_boundaries:
                        left_boundary_id = left_boundary.boundary_feature_id
                        left_start = left_boundary.lane_start_index
                        left_end = left_boundary.lane_end_index
                        left_boundary_type = left_boundary.boundary_type  # road line type
                        if left_boundary_type == 0:
                            left_boundary_type = self._current_roads[left_boundary_id].type + 8  # road edge type

                        if left_start <= curr_index <= left_end:
                            left_boundary_line = road_polylines[left_boundary_id]
                            nearest_point = find_neareast_point(point, left_boundary_line)
                            cache_lane[added_points, 3:6] = nearest_point
                            cache_lane[added_points, 11] = left_boundary_type

                    # right_boundary_point and type
                    for right_boundary in self._current_lanes[lane].right_boundaries:
                        right_boundary_id = right_boundary.boundary_feature_id
                        right_start = right_boundary.lane_start_index
                        right_end = right_boundary.lane_end_index
                        right_boundary_type = right_boundary.boundary_type  # road line type
                        if right_boundary_type == 0:
                            right_boundary_type = self._current_roads[right_boundary_id].type + 8  # road edge type

                        if right_start <= curr_index <= right_end:
                            right_boundary_line = road_polylines[right_boundary_id]
                            nearest_point = find_neareast_point(point, right_boundary_line)
                            cache_lane[added_points, 6:9] = nearest_point
                            cache_lane[added_points, 12] = right_boundary_type

                    # speed limit
                    cache_lane[added_points, 9] = self._current_lanes[lane].speed_limit_mph * c.MPH_TO_MPS

                    # interpolating
                    cache_lane[added_points, 15] = self._current_lanes[lane].interpolating

                    # traffic_light
                    if lane in traffic_light_lanes.keys():
                        cache_lane[added_points, 13] = traffic_light_lanes[lane][0]
                        if np.linalg.norm(traffic_light_lanes[lane][1:] - point[:2]) < 1:
                            cache_lane[added_points, 14] = True

                    # add stop sign
                    if lane in stop_sign_lanes:
                        cache_lane[added_points, 16] = True

                    # count

                    added_points += 1
                    curr_index += 1

                    if added_points >= 300:
                        break

            # scale the lane
            vectorized_map[i] = cache_lane[np.linspace(0, added_points, num=self.n_points_lane, endpoint=False, dtype=int)]

            # count
            added_lanes += 1

        self._added_lanes = added_lanes
        vectorized_crosswalks = None
        if self.convert_crosswalks:
            vectorized_crosswalks = self.add_crosswalks(traj).astype(np.float32)

        return vectorized_map.astype(np.float32), vectorized_crosswalks

    def _map_process_agent_specific_goals(self,
                                          traj: np.ndarray,
                                          timestep: int,
                                          other_trajs: List[np.ndarray] = None,
                                          type=None,
                                          ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract relevant map data in the surrounding of an agent's trajectory
        :param traj: trajectory for which map data is extracted
        :param timestep: initial time step of trajectory
        :param other_trajs: trajectories of other agents
        :param type: type of the agent
        :return: vectorized map and crosswalk
        """
        vectorized_map = np.zeros(shape=(self.n_lanes_max, self.n_points_lane, 17))
        vectorized_map_multiple_agents = np.zeros(shape=(1 + self.n_neighbors_max, self.n_lanes_max, self.n_points_lane, 17))
        agent_type = int(traj[-1][-1]) if type is None else type

        # get all lane polylines
        lane_polylines = get_polylines(self._current_lanes)

        # get all road lines and edges polylines
        road_polylines = get_polylines(self._current_roads)

        # find current lanes for the ego agent
        ref_lane_ids_traffic_participants = []  # list of lists
        ref_lane_ids_ego = find_reference_lanes(agent_type,
                                                traj,
                                                lane_polylines,
                                                self.search_extension_factor * self.region_of_interest,
                                                )
        ref_lane_ids_traffic_participants.append(ref_lane_ids_ego)
        # find current lanes for other agents
        if other_trajs is not None:
            for ii, other_traj in enumerate(other_trajs):
                agent_type_tmp = int(traj[-1][-1]) if type is None else type
                ref_lane_ids_traffic_participants.append(find_reference_lanes(agent_type_tmp,
                                                                              other_traj,
                                                                              lane_polylines,
                                                                              self.search_extension_factor * self.region_of_interest
                                                                              )
                                                         )
        added_lanes = 0
        for agent_count, ref_lane_ids in enumerate(ref_lane_ids_traffic_participants):
            # find candidate lanes
            ref_lanes = []

            # get current lane's forward lanes
            for curr_lane, start in ref_lane_ids.items():
                candidate = depth_first_search(curr_lane, self._current_lanes,
                                               dist=lane_polylines[curr_lane][start:].shape[0],
                                               threshold=self.lane_distance_threshold)
                ref_lanes.extend(candidate)

            if agent_type != 2:
                # find current lanes' left and right lanes
                neighbor_lane_ids = find_neighbor_lanes(ref_lane_ids, traj, self._current_lanes, lane_polylines)

                # get neighbor lane's forward lanes
                for neighbor_lane, start in neighbor_lane_ids.items():
                    candidate = depth_first_search(neighbor_lane, self._current_lanes,
                                                   dist=lane_polylines[neighbor_lane][start:].shape[0],
                                                   threshold=self.lane_distance_threshold)
                    ref_lanes.extend(candidate)

                # update reference lane ids
                ref_lane_ids.update(neighbor_lane_ids)

            # get traffic light controlled lanes and stop sign controlled lanes
            traffic_light_lanes = {}
            stop_sign_lanes = []

            for signal in self._current_traffic_signals[timestep].lane_states:
                traffic_light_lanes[signal.lane] = (signal.state, signal.stop_point.x, signal.stop_point.y)
                for lane in self._current_lanes[signal.lane].entry_lanes:
                    traffic_light_lanes[lane] = (signal.state, signal.stop_point.x, signal.stop_point.y)

            for i, sign in self._current_stop_signs.items():
                stop_sign_lanes.extend(sign.lane)

            # add lanes to the array
            for i, s_lane in enumerate(ref_lanes):
                added_points = 0
                if i >= self.n_lanes_max:
                    break

                # create a data cache
                cache_lane = np.zeros(shape=(300, 17))

                for lane in s_lane:
                    curr_index = ref_lane_ids[lane] if lane in ref_lane_ids else 0
                    self_line = lane_polylines[lane][curr_index:]

                    if added_points >= 300:  # max 150 meters (300 road points)
                        break

                    # add info to the array
                    for point in self_line:
                        # self_point and type
                        cache_lane[added_points, 0:3] = point
                        cache_lane[added_points, 10] = self._current_lanes[lane].type

                        # left_boundary_point and type
                        for left_boundary in self._current_lanes[lane].left_boundaries:
                            left_boundary_id = left_boundary.boundary_feature_id
                            left_start = left_boundary.lane_start_index
                            left_end = left_boundary.lane_end_index
                            left_boundary_type = left_boundary.boundary_type  # road line type
                            if left_boundary_type == 0:
                                left_boundary_type = self._current_roads[left_boundary_id].type + 8  # road edge type

                            if left_start <= curr_index <= left_end:
                                left_boundary_line = road_polylines[left_boundary_id]
                                nearest_point = find_neareast_point(point, left_boundary_line)
                                cache_lane[added_points, 3:6] = nearest_point
                                cache_lane[added_points, 11] = left_boundary_type

                        # right_boundary_point and type
                        for right_boundary in self._current_lanes[lane].right_boundaries:
                            right_boundary_id = right_boundary.boundary_feature_id
                            right_start = right_boundary.lane_start_index
                            right_end = right_boundary.lane_end_index
                            right_boundary_type = right_boundary.boundary_type  # road line type
                            if right_boundary_type == 0:
                                right_boundary_type = self._current_roads[right_boundary_id].type + 8  # road edge type

                            if right_start <= curr_index <= right_end:
                                right_boundary_line = road_polylines[right_boundary_id]
                                nearest_point = find_neareast_point(point, right_boundary_line)
                                cache_lane[added_points, 6:9] = nearest_point
                                cache_lane[added_points, 12] = right_boundary_type

                        # speed limit
                        cache_lane[added_points, 9] = self._current_lanes[lane].speed_limit_mph * c.MPH_TO_MPS

                        # interpolating
                        cache_lane[added_points, 15] = self._current_lanes[lane].interpolating

                        # traffic_light
                        if lane in traffic_light_lanes.keys():
                            cache_lane[added_points, 13] = traffic_light_lanes[lane][0]
                            if np.linalg.norm(traffic_light_lanes[lane][1:] - point[:2]) < 1:
                                cache_lane[added_points, 14] = True

                        # add stop sign
                        if lane in stop_sign_lanes:
                            cache_lane[added_points, 16] = True

                        # count

                        added_points += 1
                        curr_index += 1

                        if added_points >= 300:
                            break

                # scale the lane
                vectorized_map[i] = cache_lane[np.linspace(0, added_points, num=self.n_points_lane, endpoint=False, dtype=int)]

                # count
                added_lanes += 1

            vectorized_map_multiple_agents[agent_count] = vectorized_map.astype(np.float32)

        self._added_lanes = added_lanes
        vectorized_crosswalks = None
        if self.convert_crosswalks:
            vectorized_crosswalks = self.add_crosswalks(traj).astype(np.float32)

        return vectorized_map_multiple_agents, vectorized_crosswalks

    def add_crosswalks(self, traj: np.ndarray) -> np.ndarray:
        """
        find surrounding crosswalks and add them to the array
        :param traj: trajectory of an agent
        :return: vectorize crosswalks
        """
        vectorized_crosswalks = np.zeros(shape=(4, self.n_points_lane, 3))
        added_cross_walks = 0
        detection = Polygon([(0, -5), (50, -20), (50, 20), (0, 5)])
        detection = affine_transform(detection, [1, 0, 0, 1, traj[-1][0], traj[-1][1]])
        detection = rotate(detection, traj[-1][2], origin=(traj[-1][0], traj[-1][1]), use_radians=True)

        for _, crosswalk in self._current_crosswalks.items():
            polygon = Polygon([(point.x, point.y) for point in crosswalk.polygon])
            polyline = polygon_completion(crosswalk.polygon)
            polyline = polyline[np.linspace(0, polyline.shape[0], num=self.n_points_lane, endpoint=False, dtype=int)]

            if detection.intersects(polygon):
                vectorized_crosswalks[added_cross_walks, :polyline.shape[0]] = polyline
                added_cross_walks += 1

            if added_cross_walks >= 4:  # max 4 crosswalks
                break

        return vectorized_crosswalks.astype(np.float32)

    def _ego_process(self, sdc_id: int, timestep: int, tracks: "RepeatedCompositeContainers") -> np.ndarray:
        """
        Extract ego trajectory from parsed data.
        :param sdc_id: scenario ID
        :param timestep: intial time step of ego trajectory
        :param tracks: parsed data from waymo dataset
        :return: ego trajectory
        """
        ego_states = np.zeros(shape=(self.hist_len, 8), dtype=np.float32)
        sdc_states = tracks[sdc_id].states[timestep - self.hist_len:timestep]

        # add sdc states into the array
        for i, sdc_state in enumerate(sdc_states):
            ego_state = np.array([sdc_state.center_x,
                                  sdc_state.center_y,
                                  sdc_state.heading,
                                  sdc_state.velocity_x,
                                  sdc_state.velocity_y,
                                  sdc_state.length,
                                  sdc_state.width,
                                  sdc_state.height])
            ego_states[i] = ego_state
            if sdc_id not in self.agent_props:
                self.agent_props[sdc_id] = {"length": sdc_state.length,
                                            "width": sdc_state.width}

        # get the sdc current state
        self.current_xyh = ego_states[-1]
        return ego_states.astype(np.float32)

    def _extract_track_from_id(self, timestep: int, tracks: "RepeatedCompositeContainers") -> \
            Tuple[np.ndarray, List[int]]:
        """
        Returns neighbour trajectories as arrays.
        :param sdc_id: scenario ID
        :param timestep: initial time step of trajectories
        :param tracks: trajectory and agent data in protobuf format
        :return: trajectories with shape [self.num_neighbors, self.hist_len, NBR_STATES], with NBR_STATES length of:
                                        [center_x,
                                         center_y,
                                         heading,
                                         velocity_x,
                                         velocity_y,
                                         length,
                                         width,
                                         height,
                                         object_type
                                         ]
        """
        neighbors_states = np.zeros(shape=(self.n_neighbors_max, self.hist_len, 9))
        neighbors = {}
        self.neighbors_id = []

        # search for nearby agents

        track_states = tracks.states[timestep + 1 - self.hist_len:timestep + self.future_len: c.THREE]
        invalid = False
        for track_state in track_states:
            if not track_state.valid:
                invalid = True
                break
        if invalid:
            return 0
        track_states = [[track_state.center_x, track_state.center_y] for track_state in track_states]

        extracted_states = np.stack(track_states, axis=0)

        # add neighbor agents into the array
        added_num = 0
        neighbor_id = 0  # since we have passed a list to this function
        neighbor_states = tracks.states[timestep + 1 - self.hist_len:timestep + 1]
        neighbor_type = tracks.object_type

        if neighbor_type != 1:  # skip non-vehicle agents
            return 0
        any_valid = False

        for i, neighbor_state in enumerate(neighbor_states):
            if neighbor_state.valid:
                any_valid = True
                if neighbor_id not in self.agent_props:
                    self.agent_props[neighbor_id] = {"width": neighbor_state.width,
                                                     "length": neighbor_state.length,
                                                     }
                neighbors_states[added_num, i] = np.array(
                    [neighbor_state.center_x,
                     neighbor_state.center_y,
                     neighbor_state.heading,
                     neighbor_state.velocity_x,
                     neighbor_state.velocity_y,
                     neighbor_state.length,
                     neighbor_state.width,
                     neighbor_state.height,
                     neighbor_type,
                     ])


        return neighbors_states.astype(np.float32)

    def _neighbors_process(self, sdc_id: int, timestep: int, tracks: "RepeatedCompositeContainers") -> \
            Tuple[np.ndarray, List[int]]:
        """
        Returns neighbour trajectories as arrays.
        :param sdc_id: scenario ID
        :param timestep: initial time step of trajectories
        :param tracks: trajectory and agent data in protobuf format
        :return: trajectories with shape [self.num_neighbors, self.hist_len, NBR_STATES], with NBR_STATES length of:
                                        [center_x,
                                         center_y,
                                         heading,
                                         velocity_x,
                                         velocity_y,
                                         length,
                                         width,
                                         height,
                                         object_type
                                         ]
        """
        neighbors_states = np.zeros(shape=(self.n_neighbors_max, self.hist_len, 9))
        neighbors = {}
        self.neighbors_id = []

        # search for nearby agents
        for i, track in enumerate(tracks):
            track_states = track.states[timestep + 1 - self.hist_len:timestep + self.future_len: c.THREE]
            invalid = False
            for track_state in track_states:
                if not track_state.valid:
                    invalid = True
                    break
            if invalid:
                continue
            track_states = [[track_state.center_x,
                             track_state.center_y] for track_state in track_states]
            if i != sdc_id and track_states:
                neighbors[i] = np.stack(track_states, axis=0)
            elif i == sdc_id:
                ego_states = np.stack(track_states, axis=0)

        # sort the agents by distance
        min_distances = {n_id: np.min(np.linalg.norm(track_state[1] - ego_states, axis=-1, ord=1))
                         for n_id, track_state in neighbors.items()}
        min_distances = {n_id: dist for n_id, dist in min_distances.items() if dist < self.region_of_interest}
        sorted_neighbors = sorted(min_distances.items(), key=lambda item: item[1])

        # add neighbor agents into the array
        added_num = 0
        for neighbor in sorted_neighbors:
            neighbor_id = neighbor[0]
            neighbor_states = tracks[neighbor_id].states[timestep + 1 - self.hist_len:timestep + 1]
            neighbor_type = tracks[neighbor_id].object_type

            # Type: 0: unset, 1: vehicle, 2: pedestrian, 3: cyclist
            if neighbor_type != 1:  # skip non-vehicle agents
                continue
            any_valid = False
            for i, neighbor_state in enumerate(neighbor_states):
                if neighbor_state.valid:
                    any_valid = True
                    if neighbor_id not in self.agent_props:
                        self.agent_props[neighbor_id] = {"width": neighbor_state.width,
                                                         "length": neighbor_state.length,
                                                         }
                    neighbors_states[added_num, i] = np.array(
                        [neighbor_state.center_x,
                         neighbor_state.center_y,
                         neighbor_state.heading,
                         neighbor_state.velocity_x,
                         neighbor_state.velocity_y,
                         neighbor_state.length,
                         neighbor_state.width,
                         neighbor_state.height,
                         neighbor_type,
                         ])
            if any_valid:
                added_num += 1
                self.neighbors_id.append(neighbor_id)

            # only consider num_neighbor agents
            if added_num >= self.n_neighbors_max:
                break

        return neighbors_states.astype(np.float32), self.neighbors_id

    def _ground_truth_process(self, sdc_id: int, timestep: int, tracks: "RepeatedCompositeContainers") -> np.ndarray:
        """
        Return ground truth future trajectories.
        :param sdc_id: scenario ID
        :param timestep: initial time step of trajectories
        :param tracks: trajectory data in container with structure
        :return: trajectories with shape [1 + self.num_neighbors, self.future_len, NBR_STATES], with NBR_STATES length of:
                center_x
                center_y
                heading
                velocity_x
                velocity_y
        """
        ground_truth = np.zeros(shape=(1 + self.n_neighbors_max, self.future_len, 5))

        track_states = tracks[sdc_id].states[timestep + 1:timestep + self.future_len + 1]
        for i, track_state in enumerate(track_states):
            ground_truth[0, i] = np.stack([track_state.center_x,
                                           track_state.center_y,
                                           track_state.heading,
                                           track_state.velocity_x, track_state.velocity_y], axis=-1)
        if self.interactive_split:
            self.neighbors_id = [self.neighbors_id]
        for i, id in enumerate(self.neighbors_id):
            track_states = tracks[id].states[timestep + 1:timestep + self.future_len + 1]
            for j, track_state in enumerate(track_states):
                ground_truth[i + 1, j] = np.stack([track_state.center_x,
                                                   track_state.center_y,
                                                   track_state.heading,
                                                   track_state.velocity_x,
                                                   track_state.velocity_y], axis=-1)

        return ground_truth.astype(np.float32)

    def _route_process(self, sdc_id, timestep, cur_pos, tracks: "RepeatedCompositeContainers") -> Optional[np.ndarray]:
        """
        find reference paths according to the ground truth trajectory
        :param sdc_id: if of ego vehicle
        :param timestep: initial time step
        :param cur_pos: current position
        :param tracks: parsed data
        :return: reference path or None, if exception is raised during route search
        """
        gt_path = tracks[sdc_id].states

        # remove rare cases
        try:
            route = find_route(gt_path, timestep, cur_pos, self._current_lanes, self._current_crosswalks,
                               self._current_traffic_signals)
        except BaseException as e:
            traceback.print_exc()
            return None

        ref_path = np.array(route, dtype=np.float32)

        if ref_path.shape[0] < 1200:
            repeated_last_point = np.repeat(ref_path[np.newaxis, -1], 1200 - ref_path.shape[0], axis=0)
            ref_path = np.append(ref_path, repeated_last_point, axis=0)

        return ref_path

    def _normalize_data(self,
                        ego: np.ndarray,
                        neighbors: np.ndarray,
                        map_lanes: np.ndarray,
                        map_crosswalks: np.ndarray,
                        ref_line: np.ndarray,
                        ground_truth: np.ndarray,
                        viz=True,
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        transform data with respect to current ego state as new reference position
        :param ego: ego trajectory
        :param neighbors: neighbour trajectories
        :param map_lanes: vectorized maps
        :param map_crosswalks: vectorized crosswalks
        :param ref_line: reference path of ego vehicle
        :param ground_truth: ground truth trajectories
        :param viz: create visualization if true
        :return: ego, neighbors, map_lanes, map_crosswalks, ref_line, ground_truth
        """
        # get the center and heading (local view)
        center, angle = self.current_xyh[:2], self.current_xyh[2]

        # normalize agent trajectories
        ego[:, :5] = agent_norm(ego, center, angle)
        ground_truth[0] = agent_norm(ground_truth[0], center, angle)

        for i in range(neighbors.shape[0]):
            if neighbors[i, -1, 0] != 0:
                neighbors[i, :, :5] = agent_norm(neighbors[i], center, angle, impute=True)
                ground_truth[i + 1] = agent_norm(ground_truth[i + 1], center, angle)

        # normalize map points
        for i in range(map_lanes.shape[0]):
            lanes = map_lanes[i]
            crosswalks = map_crosswalks[i]

            for j in range(map_lanes.shape[1]):
                lane = lanes[j]
                if lane[0][0] != 0:
                    lane[:, :9] = map_norm(lane, center, angle)

            for k in range(map_crosswalks.shape[1]):
                crosswalk = crosswalks[k]
                if crosswalk[0][0] != 0:
                    crosswalk[:, :3] = map_norm(crosswalk, center, angle)

        # normalize ref line
        ref_line = ref_line_norm(ref_line, center, angle).astype(np.float32)

        if viz:
            rect = plt.Rectangle((ego[-1, 0] - ego[-1, 5] / 2, ego[-1, 1] - ego[-1, 6] / 2), ego[-1, 5], ego[-1, 6],
                                 linewidth=2, color='r', alpha=0.6, zorder=3,
                                 transform=mpl.transforms.Affine2D().rotate_around(*(ego[-1, 0], ego[-1, 1]),
                                                                                   ego[-1, 2]) + plt.gca().transData)
            plt.gca().add_patch(rect)

            plt.plot(ref_line[:, 0], ref_line[:, 1], 'y', linewidth=2, zorder=4)

            future = ground_truth[0][ground_truth[0][:, 0] != 0]
            plt.plot(future[:, 0], future[:, 1], 'r', linewidth=3, zorder=3)

            for i in range(neighbors.shape[0]):
                if neighbors[i, -1, 0] != 0:
                    rect = plt.Rectangle(
                        (neighbors[i, -1, 0] - neighbors[i, -1, 5] / 2, neighbors[i, -1, 1] - neighbors[i, -1, 6] / 2),
                        neighbors[i, -1, 5], neighbors[i, -1, 6], linewidth=2, color='m', alpha=0.6, zorder=3,
                        transform=mpl.transforms.Affine2D().rotate_around(*(neighbors[i, -1, 0], neighbors[i, -1, 1]),
                                                                          neighbors[i, -1, 2]) + plt.gca().transData)
                    plt.gca().add_patch(rect)
                    future = ground_truth[i + 1][ground_truth[i + 1][:, 0] != 0]
                    plt.plot(future[:, 0], future[:, 1], 'm', linewidth=3, zorder=3)

            for i in range(map_lanes.shape[0]):
                lanes = map_lanes[i]
                crosswalks = map_crosswalks[i]

                for j in range(map_lanes.shape[1]):
                    lane = lanes[j]
                    if lane[0][0] != 0:
                        centerline = lane[:, 0:2]
                        centerline = centerline[centerline[:, 0] != 0]
                        left = lane[:, 3:5]
                        left = left[left[:, 0] != 0]
                        right = lane[:, 6:8]
                        right = right[right[:, 0] != 0]
                        plt.plot(centerline[:, 0], centerline[:, 1], 'c', linewidth=3)  # plot centerline


                for k in range(map_crosswalks.shape[1]):
                    crosswalk = crosswalks[k]
                    if crosswalk[0][0] != 0:
                        crosswalk = crosswalk[crosswalk[:, 0] != 0]
                        plt.plot(crosswalk[:, 0], crosswalk[:, 1], 'b', linewidth=4)  # plot crosswalk

            plt.gca().set_aspect('equal')
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(2)
            plt.close()

        return ego, neighbors, map_lanes, map_crosswalks, ref_line, ground_truth

    def preprocess_files(self, save_path, existing_files=None, viz=True, delta_steps=1, separate_folders=False) \
            -> None:
        """
        Main function for parsing and preprocessing all files in save_path to a vectornet representation.
        :param save_path: directory for saving converted files
        :param existing_files: list of existing converted files that are not processed again
        :param viz: visualize converted scenarios
        :param delta_steps: number of time steps between two converted scenarios
        :return: None
        """
        existing_files = existing_files or []
        save_path_main = str(save_path)
        for data_file in self.data_files:
            dataset = tf.data.TFRecordDataset(data_file)
            try:
                self.pbar = tqdm(total=len(list(dataset)))
            except Exception as e:
                print(str(e))
                continue

            self.pbar.set_description(f"Processing {data_file.split('/')[-1]}")

            scenario_id = "UNKNOWN"
            for data in dataset:
                try:
                    parsed_data = scenario_pb2.Scenario()
                    parsed_data.ParseFromString(data.numpy())
                    self.agent_props = {}
                    scenario_id = parsed_data.scenario_id
                    if separate_folders:
                        save_path = Path(save_path_main, str(scenario_id))
                        os.makedirs(save_path, exist_ok=True)

                    sdc_id = parsed_data.sdc_track_index
                    time_len = len(parsed_data.tracks[sdc_id].states)
                    self._parse_map(parsed_data.map_features, parsed_data.dynamic_map_states)

                    # Interactive filtering based on objects of interest
                    object_of_interests = parsed_data.objects_of_interest
                    if len(object_of_interests) != 2:
                        continue

                    # Extract ego id
                    ego_indice = parsed_data.sdc_track_index
                    ego_id = parsed_data.tracks[ego_indice].id

                    # If ego id is within interactive agents -> filter the other participant track
                    for timestep in range(self.hist_len, time_len - self.future_len, delta_steps):
                        filename = str(Path(save_path, f"{scenario_id}_{timestep}.npz"))
                        if filename in existing_files: # check if file already exists
                            continue

                        # process data
                        if self.interactive_split:
                            id_interacting_track_1 = object_of_interests[0]
                            id_interacting_track_2 = object_of_interests[1]
                            # get interacting track form parsed_data
                            interacting_track_1 = next((track for track in parsed_data.tracks if track.id == id_interacting_track_1), None)
                            interacting_track_2 = next((track for track in parsed_data.tracks if track.id == id_interacting_track_2), None)
                            track_from_first_agent = self._extract_track_from_id(timestep, interacting_track_1)
                            track_from_second_agent = self._extract_track_from_id(timestep, interacting_track_2)

                            # Overwrite values from interacting tracks
                            ego = track_from_first_agent[0]
                            neighbors = track_from_second_agent
                            ref_line = np.zeros((10, 2))  # dummy
                            sdc_id = next((index for index, track in enumerate(parsed_data.tracks) if track.id == id_interacting_track_1), None)
                            self.neighbors_id = next((index for index, track in enumerate(parsed_data.tracks) if track.id == id_interacting_track_2), None)
                            self.agent_props = {}
                            self.agent_props[sdc_id] = {"length": interacting_track_1.states[0].length,
                                                        "width": interacting_track_1.states[0].width}  #
                            self.agent_props[self.neighbors_id] = {"length": interacting_track_2.states[0].length,
                                                                   "width": interacting_track_2.states[0].width}  #
                        else:
                            ego = self._ego_process(sdc_id, timestep, parsed_data.tracks)
                            ref_line = self._route_process(sdc_id, timestep, self.current_xyh, parsed_data.tracks)
                            if ref_line is None:
                                continue
                            neighbors, _ = self._neighbors_process(sdc_id, timestep, parsed_data.tracks)
                        if np.all(neighbors == 0):
                            # catch case, where only the ego agent is valid
                            continue

                        map_lanes = np.zeros(shape=(self.n_neighbors_max + 1, self.n_lanes_max, self.n_points_lane, 17), dtype=np.float32)
                        map_crosswalks = np.zeros(shape=(self.n_neighbors_max + 1, 4, self.n_points_lane, 3), dtype=np.float32)
                        other_trajs = [neighbors[i] for i in range(self.n_neighbors_max) if neighbors[i, -1, 0] != 0]
                        map_lanes, map_crosswalks[0] = self._map_process_agent_specific_goals(ego,
                                                                                                 timestep,
                                                                                                 other_trajs=other_trajs,
                                                                                                 type=1)
                        ground_truth = self._ground_truth_process(sdc_id, timestep, parsed_data.tracks)
                        if viz:
                            self.visualize(ego, neighbors, map_lanes, map_crosswalks, ref_line, ground_truth, "before_transform.png")
                        self._save_to_scene_prediction_format(ego,
                                                              neighbors,
                                                              map_lanes,
                                                              ground_truth,
                                                              ref_line,
                                                              map_crosswalks,
                                                              sdc_id,
                                                              filename,
                                                              viz
                                                              )
                except TypeError as e:
                    print(f"Type Error {e} for file {os.path.splitext(data_file)[-1]} with ID <{scenario_id}>")

                self.pbar.update(1)

            self.pbar.close()

    def visualize(self, ego, neighbors, map_lanes, map_crosswalks, ref_line, ground_truth, plot_name):
        rect = plt.Rectangle((ego[-1, 0] - ego[-1, 5] / 2, ego[-1, 1] - ego[-1, 6] / 2), ego[-1, 5], ego[-1, 6],
                             linewidth=2, color='r', alpha=0.6, zorder=3,
                             transform=mpl.transforms.Affine2D().rotate_around(*(ego[-1, 0], ego[-1, 1]),
                                                                               ego[-1, 2]) + plt.gca().transData)
        plt.gca().add_patch(rect)
        plt.plot(ref_line[:, 0], ref_line[:, 1], 'y', linewidth=2, zorder=4)
        future = ground_truth[0][ground_truth[0][:, 0] != 0]
        plt.plot(future[:, 0], future[:, 1], 'r', linewidth=3, zorder=3)
        for i in range(neighbors.shape[0]):
            if neighbors[i, -1, 0] != 0:
                rect = plt.Rectangle(
                    (neighbors[i, -1, 0] - neighbors[i, -1, 5] / 2, neighbors[i, -1, 1] - neighbors[i, -1, 6] / 2),
                    neighbors[i, -1, 5], neighbors[i, -1, 6], linewidth=2, color='m', alpha=0.6, zorder=3,
                    transform=mpl.transforms.Affine2D().rotate_around(*(neighbors[i, -1, 0], neighbors[i, -1, 1]),
                                                                      neighbors[i, -1, 2]) + plt.gca().transData)
                plt.gca().add_patch(rect)
                future = ground_truth[i + 1][ground_truth[i + 1][:, 0] != 0]
                plt.plot(future[:, 0], future[:, 1], 'm', linewidth=3, zorder=3)

        for i in range(map_lanes.shape[0]):
            lanes = map_lanes[i]
            crosswalks = map_crosswalks[i]

            for j in range(map_lanes.shape[1]):
                lane = lanes[j]
                if lane[0][0] != 0:
                    centerline = lane[:, 0:2]
                    centerline = centerline[centerline[:, 0] != 0]
                    left = lane[:, 3:5]
                    left = left[left[:, 0] != 0]
                    right = lane[:, 6:8]
                    right = right[right[:, 0] != 0]
                    plt.plot(centerline[:, 0], centerline[:, 1], 'c', linewidth=3)  # plot centerline
                    plt.plot(left[:, 0], left[:, 1], 'k', linewidth=3)  # plot left boundary
                    plt.plot(right[:, 0], right[:, 1], 'k', linewidth=3)  # plot right boundary

            for k in range(map_crosswalks.shape[1]):
                crosswalk = crosswalks[k]
                if crosswalk[0][0] != 0:
                    crosswalk = crosswalk[crosswalk[:, 0] != 0]
                    plt.plot(crosswalk[:, 0], crosswalk[:, 1], 'b', linewidth=4)  # plot crosswalk

        # apply to centerlines
        min_x = np.min(centerline[:,0])
        max_x = np.max(centerline[:,0])
        min_y = np.min(centerline[:,1])
        max_y = np.max(centerline[:,1])

        # apply to futures considering previous minimum values
        min_x = np.min([min_x, np.min(future[:,0])])
        max_x = np.max([max_x, np.max(future[:,0])])
        min_y = np.min([min_y, np.min(future[:,1])])
        max_y = np.max([max_y, np.max(future[:,1])])

        plt.xlim(min_x - 10, max_x + 10)
        plt.ylim(min_y - 10, max_y + 10)
        plt.gca().set_aspect('equal')


        plt.savefig(plot_name)
        plt.close()


    def _save_to_scene_prediction_format(self,
                                         ego: np.ndarray,
                                         neighbors: np.ndarray,
                                         map_lanes: np.ndarray,
                                         gt_future_states: np.ndarray,
                                         ref_line: np.ndarray,
                                         map_crosswalks,
                                         sdc_id,
                                         filename: str,
                                         viz,
                                         ) -> None:
        """
        Save preprocessed in Scene-Prediction's vectornet representation as .npz files.
        :param ego: trajectories of ego vehicle
        :param neighbors: trajectories of neighbouring agents
        :param map_lanes: vectorized map
        :param gt_future_states: gorund truth trajectories of other agents
        :param ref_line: reference path of ego vehicle
        :param map_crosswalks: (not supported yet)
        :param sdc_id: ID of current sample
        :param filename: filename for saving the sample
        :return: None
        """
        sample = self._convert_trajectory_data_to_scene_prediction_format(ego, neighbors, gt_future_states, sdc_id)
        if sample is None:
            return

        # Add map data and reachable centerline for goal positions
        sample['map'], sample[c.KEY_CENTERLINE] = self._convert_lanes_and_goals_data_to_scene_prediction_format(map_lanes, sample)

        # Add map coordinates.
        sample["map_max_coords"] = [-self.region_of_interest, self.region_of_interest, -self.region_of_interest, self.region_of_interest]

        # Extract init state and transform to ego coordinates
        ego_init_state = np.array((ego[-1, 0:3]))
        neighbors_init_state = neighbors[:, -1, 0:3]
        if (neighbors_init_state == ego_init_state).any():
            print("ERROR: Ego and Neighbors have same init state")
            return
        init_state = np.vstack([ego_init_state, neighbors_init_state])
        init_state = transform_in_ego_coordinates(init_state, sample["reference_pose"]) # transform the x,y position
        init_heading = init_state[:, 2]
        transformed_init_heading = transform_heading_in_ego_coordinates(np.array([[init_heading]]), sample["reference_pose"])[0][0] # transform the heading
        init_state[:, 2] = transformed_init_heading

        # Calculate and append velocity
        ego_v0 = np.array(np.sqrt(ego[-1, 3] ** 2 + ego[-1, 4] ** 2))
        neighbors_v0 = np.sqrt(neighbors[:, -1, 3] ** 2 + neighbors[:, -1, 4] ** 2)
        init_v0 = np.hstack([ego_v0, neighbors_v0]) * c.MPH_TO_MPS
        init_game = np.hstack([init_state, init_v0.reshape(-1, 1)])

        sample['init_state'] = init_game
        sample['reference_path'] = ref_line
        np.savez(file=str(filename), **sample)
        print("Saved sample to {}".format(filename))
        if viz:
            self._plot_vn_sample(sample, figure_name = "after_transform.png")

    def _plot_vn_sample(self, input_env, figure_name):
        map = input_env["map"]
        agent_colors = ["r", "g", "b", "y", "m", "c"]

        for lane in map:
            x_values_map = []
            y_values_map = []
            for node in lane:
                x_values_map.append(node[0])
                y_values_map.append(node[1])
            plt.plot(x_values_map, y_values_map, 'ko', alpha=0.8, markersize=0.6)

        for agent in range(input_env["init_state"].shape[0]):  # for each agent
            plt.plot(input_env['x'][:, agent, 0], input_env['x'][:, agent, 1], 'g-', linewidth=1)
            plt.plot(input_env['y'][:, agent, 0], input_env['y'][:, agent, 1], 'r-', linewidth=1)

            curr_init = input_env['init_state'][agent, 0:2]  # x, y
            plt.plot(curr_init[0], curr_init[1], 'o', color=agent_colors[agent], markersize=4)

            # Create the rectangle centered at curr_init
            rect = patches.Rectangle(
                (-input_env['agent_length'][agent] / 2, -input_env['agent_width'][agent] / 2),
                input_env['agent_length'][agent],
                input_env['agent_width'][agent],
                angle=0,  # set angle to 0 initially
                edgecolor=agent_colors[agent],
                facecolor='none'
            )

            # Apply transformation: Translate then Rotate
            t = patches.transforms.Affine2D().rotate_deg(input_env['init_state'][agent, 2] * 180 / np.pi).translate(curr_init[0], curr_init[1])
            rect.set_transform(t + plt.gca().transData)

            plt.gca().add_patch(rect)


        plt.axis('equal')
        plt.savefig(figure_name)
        plt.close()

    def _normalize_reference_path(self, reference_path, sample) -> np.ndarray:
        """
        Transform reference path with respect to reference pose from sample
        :param reference_path: reference path (i.e. center line of ego vehicle's lanes)
        :param sample: sample in scene prediction format
        :return: normalized reference_path
        """
        reference_pose = sample["reference_pose"]
        reference_path[:, :2] = transform_in_ego_coordinates(reference_path[:, :2], reference_pose)
        reference_path[:, 2] = transform_yaws(reference_path[:, 2], world_to_ego_quaternion(reference_pose))
        return reference_path

    def _get_lane_features(self, i_lane) -> List[int]:
        """
        Get road features for lane.
        :param i_lane: lane ID
        :return: [has_traffic_control, in_intersection, lane ID]
        """
        return [0, 0, i_lane]

    def get_halluc_lane_vec(
            self,
            centerlane: np.ndarray,
            city_name: str,
            use_deltas: bool = False,
            create_bounds=False,
    ) -> List[np.ndarray]:
        """
        return left & right lane based on centerline from https://github.com/xk-huang/yet-another-vectornet
        :param centerlane: center line of the lane
        :param city_name: name of city for lane width
        :returns:
            doubled_left_halluc_lane, doubled_right_halluc_lane, shaped in (N-1, 3)
        """
        if centerlane.shape[0] <= 1:
            raise ValueError('shape of centerlane error.')

        half_width = c.LANE_WIDTH[city_name] / 2
        rotate_quat = np.array([[0.0, -1.0], [1.0, 0.0]])
        end = centerlane[1:, :2]
        start = centerlane[:-1, :2]
        if create_bounds:
            dx_v = start - end
            norm_v = np.linalg.norm(dx_v, axis=-1)
            e1v = (dx_v @ rotate_quat.T) / np.repeat(norm_v[:, np.newaxis], 2, axis=c.LAST_VALUE)
            e2v = (dx_v @ rotate_quat) / np.repeat(norm_v[:, np.newaxis], 2, axis=c.LAST_VALUE)
            if use_deltas:
                params11 = end - start
                params21 = params11
            else:
                params11 = start + e1v * half_width
                params21 = start + e2v * half_width
            halluc_lane_1 = np.hstack((end + e1v * half_width, params11))
            halluc_lane_2 = np.hstack((end + e2v * half_width, params21))
            return [halluc_lane_1, halluc_lane_2]
        else:
            return [np.hstack((end, start))]

    def _convert_lanes_data_to_scene_prediction_format(self, map_lanes, sample) -> np.ndarray:
        """
        Convert lanes to Scene-Prediction's vector format.
        :param map_lanes: lanes
        :param sample: sample with reference pose
        :return: vectorized map in scene prediction format
        """

        def get_lane_coordinates(centerlane) -> list:
            """
            query lane marking coordinates with lane id
            :param id:
            :return:
            """
            centerlane = transform_in_ego_coordinates(centerlane, sample["reference_pose"])
            bounds = self.get_halluc_lane_vec(
                centerlane[:, c.INDICES_MAP_COORDINATES],
                "MIA",
                use_deltas=False,
                create_bounds=True,
            )
            return bounds

        lane_graph = []
        lane_graph_goal_points = []
        n_lanes = min(self._added_lanes, map_lanes.shape[1])
        for i_lane in range(n_lanes):
            centerline_points = map_lanes[0, i_lane, :, :2]
            centerline_points = transform_in_ego_coordinates(centerline_points, sample["reference_pose"])
            lane_bounds = get_lane_coordinates(centerline_points)
            for lane_bound in lane_bounds:
                nan_indices = np.argwhere(np.isnan(lane_bound[:, 0])).flatten()
                for i_point in nan_indices:
                    if 0 < i_point < lane_bound.shape[0] - 1:
                        lane_bound[i_point, :] = np.mean(lane_bound[[i_point - 1, i_point + 1], :], axis=0)
                    elif i_point == 0:
                        lane_bound[i_point, :] = lane_bound[i_point + 1, :]
                    else:
                        lane_bound[i_point, :] = lane_bound[i_point - 1, :]

            lane_features = self._get_lane_features(i_lane)
            lane_array = format_lane_array(lane_bounds, lane_features)
            lane_graph.append(lane_array)
            lane_graph_goal_points.append(centerline_points)

        lane_graph = np.concatenate(lane_graph, axis=c.SECOND_VALUE)
        lane_graph_goal_points = np.concatenate(lane_graph_goal_points,
                                                axis=c.FIRST_VALUE)
        assert not np.isnan(lane_graph_goal_points).any(), "NaN in lane graph centerpoints!"
        lane_graph[np.isnan(lane_graph)] = 0.0
        print(lane_graph_goal_points.shape)
        return lane_graph, lane_graph_goal_points

    def _convert_lanes_and_goals_data_to_scene_prediction_format(self, map_lanes, sample) -> np.ndarray:
        """
        Convert lanes to scene prediction vector format.
        :param map_lanes: lanes
        :param sample: sample with reference pose
        :return: vectorized map in scene prediction format
        """

        def get_lane_coordinates(centerlane) -> list:
            """
            query lane marking coordinates with lane id
            :param id:
            :return:
            """
            centerlane = transform_in_ego_coordinates(centerlane, sample["reference_pose"])
            bounds = self.get_halluc_lane_vec(
                centerlane[:, c.INDICES_MAP_COORDINATES],
                "MIA",
                use_deltas=False,
                create_bounds=True,
            )
            return bounds

        def get_lane_bound_coordinates(centerlane) -> list:
            """
            :param centerlane:
            :return: lane bounds
            """
            bounds = self.get_halluc_lane_vec(
                centerlane[:, c.INDICES_MAP_COORDINATES],
                "MIA",
                use_deltas=False,
                create_bounds=True,
            )
            return bounds

        lane_graph = []
        lane_graph_goal_points = []
        n_lanes = min(self._added_lanes, map_lanes.shape[1])

        # Create Map
        for agent_count in range(map_lanes.shape[0]):
            lane_graph_goal_points_per_agent = []
            for i_lane in range(n_lanes):
                centerline_points = map_lanes[agent_count, i_lane, :, :2]
                if np.all(centerline_points==0): # padded
                    lane_bounds = get_lane_coordinates(centerline_points)
                    if np.isnan(lane_bounds).any():
                        lane_bounds[0][np.isnan(lane_bounds[0])] = 0.0 #* check in left bound for nans
                        lane_bounds[1][np.isnan(lane_bounds[1])] = 0.0 #* check in right bound for nans
                else:
                    # mask nan into 0.0
                    centerline_points = transform_in_ego_coordinates(centerline_points, sample["reference_pose"])
                    lane_bounds = get_lane_bound_coordinates(centerline_points)
                for lane_bound in lane_bounds:
                    nan_indices = np.argwhere(np.isnan(lane_bound[:, 0])).flatten()
                    for i_point in nan_indices:
                        if 0 < i_point < lane_bound.shape[0] - 1:
                            lane_bound[i_point, :] = np.mean(lane_bound[[i_point - 1, i_point + 1], :], axis=0)
                        elif i_point == 0:
                            lane_bound[i_point, :] = lane_bound[i_point + 1, :]
                        else:
                            lane_bound[i_point, :] = lane_bound[i_point - 1, :]

                lane_features = self._get_lane_features(i_lane)
                lane_array = format_lane_array(lane_bounds, lane_features)
                lane_graph.append(lane_array)
                lane_graph_goal_points_per_agent.append(centerline_points)
            lane_graph_goal_points.append(lane_graph_goal_points_per_agent)

        # Create Goal Points
        lane_graph = np.concatenate(lane_graph, axis=c.SECOND_VALUE)

        # lane_graph_goal_points = [Agents, Lanes, Points, (x,y)]
        lane_graph_goal_points = np.stack(lane_graph_goal_points)

        assert not np.isnan(lane_graph_goal_points).any(), "NaN in lane graph centerpoints!"


        # Create a tensor where all points are stored, and then pad it with zeros to get the maximum number of goal points
        all_goal_points = np.zeros((self.n_neighbors_max+1,self.n_lanes_max,self.n_points_lane,2))
        all_goal_points[:, :lane_graph_goal_points.shape[1],:,:] = lane_graph_goal_points
        all_goal_points_feature = np.zeros((self.n_neighbors_max+1,self.n_lanes_max,self.n_points_lane,2+1))
        condition = all_goal_points[:,:,:,:] == 0 # check if point is zero
        condition = np.where(condition.any(-1),0,1)[...,None] # convert to binary numerics
        all_goal_points_feature = np.concatenate((all_goal_points, condition), axis=-1)
        lane_graph[np.isnan(lane_graph)] = 0.0
        return lane_graph, all_goal_points_feature


    def trajectory_to_sample(self,
                             trajectories: np.ndarray,
                             agent_properties: dict,
                             yaw: float,
                             data_loader: WaymoData,
                             ) -> Optional[dict]:
        """
        Collect all data in dict in scene-prediction format
        :param trajectories: trajectories of
        :param agent_properties: contains meta data of agents
        :param yaw: yaw of reference pose
        :param data_loader: data loader for waymo data
        :return: dict in scene-prediction format (see README-md for details about format)
        """
        reference_pose = dict()
        reference_pose['translation'] = trajectories[:, data_loader._target_agent_number, :][data_loader._input_frames, c.INDICES_COORDINATES]
        reference_pose['rotation'] = list(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]))
        trajectories = transform_in_ego_coordinates(trajectories, reference_pose)
        trajectories = data_loader.filter_with_roi(
            trajectories=trajectories,
            roi=data_loader.cfgs["hyperparameters"]["region_of_interest"],
            input_frames=data_loader._input_frames
        )
        trajectories_processed = data_loader.process_trajectories(trajectories)
        tracking_states = get_tracking_states(trajectories)
        sample_input = trajectories_processed[:data_loader._input_frames, :, :]
        input = np.concatenate(
            (sample_input, tracking_states[:data_loader._input_frames, ...]),
            axis=c.LAST_VALUE
        )
        ground_truth = trajectories_processed[data_loader._input_frames:data_loader._all_frames, :, :]
        loss_masks = trajectories[
                     (data_loader._input_frames + 1):(data_loader._all_frames + 1),
                     :,
                     c.INDICES_TRACKING_STATE_WIP]

        if np.max(np.abs(input)) > 3000:
            print("Input too large")
            return None

        sample = {'x': input,
                  'y': ground_truth,
                  'loss_masks': loss_masks,
                  'reference_pose': reference_pose,
                  "agent_length": agent_properties["length"],
                  "agent_width": agent_properties["width"],
                  'index_target': data_loader._target_agent_number
                  }
        return sample

    def _add_loss_mask(self, all_trajectories: np.ndarray) -> np.ndarray:
        """
        Add column with loss masks to trajectories, where trajectories are zero.
        :param all_trajectories:
        :return: trajectories with loss_mask column
        """
        loss_mask = np.logical_not(np.all(all_trajectories[:, :, :] == 0.0, axis=2)). \
            reshape(list(all_trajectories.shape[:2]) + [1])
        return np.append(all_trajectories, loss_mask, axis=2)

    def _convert_trajectory_data_to_scene_prediction_format(self, ego, neighbors, gt_future_states, sdc_id) -> dict:
        """
        Convert trajectory data of ego and other agents to scene prediction format.
        :param ego: ego trajectory
        :param neighbors: trajectories of neighbouring agents
        :param gt_future_states: ground truth data of all agents
        :param sdc_id: scenario ID
        :return: sample dict in scene prediction format (see README.md for details on vector format)
        """
        # aggregate trajectories into single array
        trajectories = neighbors[:len(self.neighbors_id), :, :2]
        all_histories = np.insert(trajectories, 0, ego[:, :2], axis=0)
        all_trajectories = np.concatenate([all_histories,
                                           gt_future_states[:len(self.neighbors_id) + 1, :, :2]],
                                          axis=1)
        all_trajectories = all_trajectories.swapaxes(0, 1)
        all_trajectories = self._add_loss_mask(all_trajectories)
        data_loader = WaymoData(self.config, dir=args.load_path, fps=c.FPS_WAYMO)
        dt = 1 / c.FPS_WAYMO
        data_loader.all_timestamps = np.arange(0, (self.hist_len + self.future_len) * dt, dt)
        data_loader._target_agent_number = 0
        agent_properties = self.create_agent_properties(sdc_id)
        sample = self.trajectory_to_sample(all_trajectories, agent_properties, yaw=ego[-1, 2], data_loader=data_loader)

        return sample

    def create_agent_properties(self, sdc_id) -> Dict[str, np.ndarray]:
        """
        Collect meta data of each agent.
        :param sdc_id: scneario ID
        :return: dictionary with keys "width", "length"
        """
        props = {}
        all_ids = [sdc_id] + self.neighbors_id
        props["width"] = np.array([self.agent_props[agent_id]["width"] for agent_id in all_ids])
        props["length"] = np.array([self.agent_props[agent_id]["length"] for agent_id in all_ids])
        return props


def multiprocessing(arg) -> None:
    """
    Use multi-processing for preprocessing.
    :param arg: tuple(data_files, existing_files, config)
    :return: None
    """
    data_files, existing_files, config = arg
    processor = WaymoPreprocessing([data_files, ], config)
    processor.preprocess_files(save_path, existing_files=existing_files, viz=False, delta_steps=processor.delta_steps, separate_folders=args.sequences > 0)


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='Data Processing')
    parser.add_argument('--load_path',
                        type=str, help='path to dataset files')
    parser.add_argument('--save_path',
                        type=str, help='path, where preprocessed data will be saved')
    parser.add_argument('--n_parallel', type=int, help='if use multiprocessing', default=0)
    parser.add_argument('--overwrite', action='store_true', help='path to save processed data')
    parser.add_argument('--sequences', type=int, default=-1, help='If > 0: Convert n continuous sequences for closed loop '
                                                                  'simulation. Saves files into folders for '
                                                                  'sequences separately.')
    parser.add_argument("--configs", help="path to config_MODELNAME_.json file")

    parser.set_defaults(feature=False)
    args = parser.parse_args()

    data_files = glob.glob(args.load_path + '/*')
    delta_steps = None
    if args.sequences > 0:
        data_files = random.sample(data_files, k=args.sequences)
        delta_steps = 1
    save_path = args.save_path

    os.makedirs(save_path, exist_ok=True)
    if args.overwrite:
        existing_files = []
        for f in glob.glob(args.save_path + '/**/*.npz', recursive=True):
            os.remove(f)
    else:
        existing_files = glob.glob(str(Path(args.save_path, '*.npz')), recursive=True)
    print(f"{len(existing_files)} preprocessed will be ignored")

    config_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), args.configs)
    config = load_configs(config_path)

    if args.n_parallel > 1:
        pool = Pool(args.n_parallel)
        arguments = zip(data_files, [existing_files] * len(data_files), [config] * len(data_files))
        with Pool(args.n_parallel) as p:
            pool.map(multiprocessing, arguments)
    else:
        processor = WaymoPreprocessing(data_files, config)
        processor.preprocess_files(save_path,
                                   existing_files=existing_files,
                                   delta_steps=processor.delta_steps,
                                   viz=False,
                                   separate_folders=args.sequences > 0,
                                   )

