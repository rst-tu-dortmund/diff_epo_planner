# ------------------------------------------------------------------------------
# Copyright:    ZF Friedrichshafen AG
# Project:      KISSaF
# Created by:   ZF AI LAB SBR
# ------------------------------------------------------------------------------
import numpy as np
from data import constants as c


def repetative_padding(
    lane_array: list,
    nbr_subgraph_points: int,
    padding_dim: int = 0,
) -> list:
    """padding with last lane vector to reach default nbr.

    :param lane_array:
    :param nbr_subgraph_points: max number of samples in lane array
    :param padding_dim: dimension along which samples should be padded
    :return:
    """
    length_lane = len(lane_array)
    if length_lane < nbr_subgraph_points:
        nbr_repeats = nbr_subgraph_points - length_lane
        repeated_sample = np.repeat(lane_array[-1], nbr_repeats, axis=padding_dim)
        lane_array.append(repeated_sample)
    elif length_lane > nbr_subgraph_points:
        assert (
            "maximum number of datapoints per subgraph node needs to be increased "
            "to at least {}!".format(length_lane)
        )
    return lane_array


def format_lane_array(lane_coordinates: list, lane_features: list) -> np.array:
    """format a list of lane coordinates to numpy array.

    :param lane_coordinates: array with coordinates of lane vertices
    :param lane_features: list with additional features that should be added to lane array
    :return: formatted lane array
    """
    lane_array = []
    for lane in lane_coordinates:
        for lane_point in lane:
            lane_vector = np.append(lane_point, lane_features).reshape(1, 1, -1)
            lane_array.append(lane_vector)
    lane_array_rep = repetative_padding(lane_array, c.NBR_SUBGRAPH_POINTS)
    lane_array_concat = np.concatenate(lane_array_rep, axis=c.FIRST_VALUE)
    return lane_array_concat
