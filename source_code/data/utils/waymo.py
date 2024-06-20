# ------------------------------------------------------------------------------
# Copyright:    Technical University Dortmund
# Project:      KISSaF
# Created by:   Institute of Control Theory and System Engineering
# ------------------------------------------------------------------------------

import glob
from functools import cached_property
from typing import Callable

import einops
from data.utils import VectorData
from data.utils.vector_utils import *
from data.utils.vector_utils import remove_lane_ids



class WaymoData(VectorData):
    """DataLoader to extract Vector based trajectories from the waymo open
    dataset."""

    def __init__(
        self,
        cfgs: dict,
        dir: Union[Path, str],
        fps: int = c.FPS_WAYMO,
        val: bool = False,
    ):
        """
        :param cfgs: configuration
        :param dir: directory of data samples
        :param fps: frames per second in data
        :param val: eval mode or not
        """
        super().__init__(cfgs, dir=dir, fps=fps, val=val)

    @cached_property
    def all_files(self) -> list:
        """
        :return: list of path to all data samples
        """
        return sorted(glob.glob(str(Path(self.data_path, Path("*.npz")))))

    @cached_property
    def file_list(self) -> list:
        """
        :return: list of path to all data samples
        """
        return self.all_files

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if self.cfgs["config_params"]["preprocess"]:
            sample = self._load_sample_from_npz(idx)
            # sample = self.apply_augmentations(sample)
            return sample
        else:
            raise ValueError(f"Only preprocessed data can be loaded using WaymoData!")

    def append_lane_graph(self, city_name: str, sample: dict) -> dict:
        """Append map information to the sample dict that is the return value
        of the getitem method.

        :param city_name: city name of current data
        :param sample: return value where map data will be appended
        :return:
        """
        raise ValueError(f"Only preprocessed data can be loaded using WaymoData!")

    def _load_sample_from_npz(self, idx: int) -> dict:
        """Load sample in scene prediction format from an *.npz file.
        :param idx: index in file list
        :return: data sample
        """
        data = np.load(self.file_list[idx], allow_pickle=True)
        game_goal = np.zeros((2,4))
        game_goal[:] = data["y"][-1, :, 0:4]  # extract last x and y position of all agent
        game_goal[..., 2:4] = 0.0 # game_goal is not used in the current version
        sample = {
            "x": data["x"],  # [time, agents, states]
            "y": data["y"],  # [time, agents, states]
            "loss_masks": data["loss_masks"],
            "reference_pose": data["reference_pose"].item(),
            "init_state": data["init_state"],
            "goal_state": game_goal,
            "index_target": data["index_target"].item(),
            "reference_path": data["reference_path"],
            "centerline": data["centerline"],
            "map": data["map"],
            "agent_length": data["agent_length"],
            "agent_width": data["agent_width"],
        }
        sample = self.format_dynamic_info(sample)
        sample = remove_lane_ids(sample)
        sample["map"][np.isnan(sample["map"])] = 0.0
        sample["file_path"] = self.file_list[idx]
        return sample

    def map_binary_domain_decisions(self, key: str) -> Callable:
        """implementation of abstract method from parent So far no
        implementation for this dataset needed.

        :param key:
        :return:
        """
        logging.error("not implemented for waymo")
        return -1


def collate_waymo_scenes(data: list) -> dict:
    '''
    collate vector based scenes
    :param data:
    :return:
    '''
    all_inputs = []
    all_targets = []
    all_loss_masks = []
    all_reference_poses = []
    all_lane_graphs = []
    all_target_indices = []
    all_init_states = []
    all_goal_states = []
    all_agent_lengths = []
    all_agent_widths = []
    all_agent_centerlines = []
    all_lanelet_maps = []
    all_filenames = []
    all_map_ref_vel = []

    for sample in data:
        all_inputs.append(np.expand_dims(sample['x'], axis=c.BATCH_DIM_SEQUENCE))
        all_targets.append(np.expand_dims(sample['y'], axis=c.BATCH_DIM_SEQUENCE))
        all_loss_masks.append(np.expand_dims(sample['loss_masks'], axis=c.BATCH_DIM_SEQUENCE))
        all_lane_graphs.append(np.expand_dims(sample['map'], axis=c.BATCH_DIM_SEQUENCE))
        all_reference_poses.append(sample['reference_pose'])
        all_target_indices.append(sample['index_target'])
        all_agent_lengths.append(sample['agent_length'])
        all_agent_widths.append(sample['agent_width'])
        # Agent Data
        all_init_states.append(sample[c.KEY_INIT_STATE])
        all_goal_states.append(sample[c.KEY_GOAL_STATE])

        # first ignore
        all_agent_centerlines.append(sample[c.KEY_CENTERLINE])
        all_filenames.append(sample['file_path'])
        if 'map_ref_vel' in sample:
            all_map_ref_vel.append(sample['map_ref_vel'])
        else:
            all_map_ref_vel.append(-1000)

    all_inputs = global_padding(all_inputs)
    all_targets = global_padding(all_targets)
    all_loss_masks = global_padding(all_loss_masks)
    all_lane_graphs = global_padding(all_lane_graphs, flag='lane_graph')
    all_agent_lengths = agent_geometry_padding(all_agent_lengths)
    all_agent_widths = agent_geometry_padding(all_agent_widths)

    all_init_states = np.stack(all_init_states)
    all_goal_states = np.stack(all_goal_states)

    all_agent_centerlines = np.stack(all_agent_centerlines)

    target_futures = torch.from_numpy(all_targets)
    batch = {'x': torch.from_numpy(all_inputs),
             'y': target_futures,
             'loss_masks': torch.from_numpy(all_loss_masks),
             'reference_pose': all_reference_poses,
             'map': torch.from_numpy(all_lane_graphs),
             'lanelet_map': all_lanelet_maps,
             'index_target': all_target_indices,
             'init_state': torch.from_numpy(all_init_states),
             'goal_state': torch.from_numpy(all_goal_states),
             'agent_centerlines': torch.from_numpy(all_agent_centerlines),
             'agent_lengths': torch.from_numpy(all_agent_lengths),
             'agent_widths': torch.from_numpy(all_agent_widths),
             'file_path': all_filenames
             }
    return batch


def agent_geometry_padding(inputs):
    max_size = [tensor.shape[c.FIRST_VALUE] for tensor in inputs]
    padded_tensors = [np.pad(curr_tensor, ((0, max(max_size) - curr_tensor.shape[0])), mode='constant', constant_values=0.0) for curr_tensor in inputs]
    return np.stack(padded_tensors)


def convert_tensormap_to_listmap(map):
    """
    :param map: input map from dataset (map elements x batch size x n_lanes_max (from waymo preprocessing) x vector features
    :return: lanelet_map: map in format for plotting
    """
    list_map = map
    map = einops.rearrange(map, 'w l f -> l w f')  # w:waypoint, l:lane, f:feature
    list_map = map.tolist()
    filtered_list_2 = [
        [
            [element for element in row if element != 0]  # filter out zeros from each row
            for row in inner_list
            if any(element != 0 for element in row)  # filter out rows containing only zeros
        ]
        for inner_list in list_map
        if any(any(element != 0 for element in row) for row in inner_list)  # filter out inner lists containing only zeros
    ]

    list_filtered_list = [filtered_list_2]



    return list_filtered_list


def remove_empty_lists(filtered_list):
    if isinstance(filtered_list, list):
        # Iterate over a copy of the list to avoid modifying it during iteration
        for sublist in filtered_list[:]:
            remove_empty_lists(sublist)
            if all(isinstance(item, list) and not item for item in sublist):
                filtered_list.remove(sublist)

def get_num_of_agents(inputs):
    num_of_agents = [int(curr_tensor.shape[2]) for curr_tensor in inputs]
    return num_of_agents