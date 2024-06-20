# ------------------------------------------------------------------------------
# Copyright:    ZF Friedrichshafen AG
# Project:      KISSaF
# Created by:   ZF AI LAB SBR
# ------------------------------------------------------------------------------
import os
from enum import IntEnum
from typing import Union

import numpy as np
import torch
import math

tensor_or_array = Union[torch.Tensor, np.ndarray]


class MapFuseType(IntEnum):
    INTERPOLATED = 1
    CNN = 2

class ModelType(IntEnum):
    V_Net = 5
    VECTOR_MLP = 6
    NAIVE_MLP = 7


# Set project root
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), "../.."))

# List Iteration
LAST_VALUE = -1
LAST = -1
SECOND_LAST_VALUE = -2
SECOND_LAST = -2
FIRST_VALUE = 0
FIRST = 0
SECOND_VALUE = 1
SECOND = 1
THIRD_VALUE = 2
THIRD = 2
FIRST_TWO_VALUES = slice(0, 2)

# Integers
ZERO = 0
ONE = 1
TWO = 2
THREE = 3
FOUR = 4

# Threshold values (unused, expect THRESHOLD_INDEPENDENT)
THRESHOLD_PED_PED = 1.66433170e-02
THRESHOLD_CAR_PED = 7.45614512e-01
THRESHOLD_CAR_CAR = 1.83362483e00
THRESHOLD_TWO_WHEELERS = 4.45917033e-01
THRESHOLD_INDEPENDENT = 0.15
THRESHOLD_SMALL_NUMERICAL_VALUE = 1e-6

# Indices and order of data in dataloader
INDEX_X_COORDINATE = 0
INDEX_Y_COORDINATE = 1
DELTA_X_INDEX = 2
DELTA_Y_INDEX = 3
INDEX_AGENT_CLASS = 7
INDEX_YAW = 5
INDEX_SIZE_START = 8
INDEX_SIZE_END = 10
INDEX_TRAFFIC_LIGHT = 4
INDEX_INTERSECTION = 5
INDEX_OBSERVATION = 6
INDICES_MAP_COORDINATES = slice(0, 2)
INDICES_VECTOR_END = slice(0, 2)
INDICES_VECTOR_START = slice(2, 4)
DELTAS = slice(2, 4)


# Parameters which can not be reset with new config

GENERAL_KEYS_OVERWRITE = ["model"]
CONFIG_PARAM_KEYS_OVERWRITE = [
    "direct_prediction",
    "interaction",
    "no_probs",
    "predict_target",
    "vn_dyn_encoder_type",
    "vn_input_format",
    "vn_map_encoder_type",
]

HYPERPARAMETER_KEYS_OVERWRITE = [
    "head_number",
    "norm_layer",
    "num_modes",
    "prediction_length",
    "vn_decoder_depth",
    "vn_decoder_width",
    "vn_dyn_encoder_depth",
    "vn_encoder_width",
    "vn_map_encoder_depth",
    "vn_global_graph_depth",
    "vn_global_graph_width",
]
OVERWRITE_KEYS = {
    "general": GENERAL_KEYS_OVERWRITE,
    "config_params": CONFIG_PARAM_KEYS_OVERWRITE,
    "hyperparameters": HYPERPARAMETER_KEYS_OVERWRITE,
}


INPUT_PARAM_LENGTH = 12
LABEL_CODE_ELSE = 11
PEDESTRIAN_LABEL = 9
NO_DELTA_SAMPLE = [[0, 0]]
NO_DELTA_YAW_SAMPLE = 0

# Indices for counting pedestrians
START_INDEX = 0
END_INDEX = 1

# Indices and dimensions for predictions
MODE_DIM_PREDICTION = 0
TIME_DIM_PREDICTION = 1
AGENT_DIM_PREDICTION = 2
PARAM_DIM_PREDICTION = 3
TIME_DIM_INPUT = 0
TIME_DIM_CLOSEST_MODE_CALCULATION = 2
AGENT_DIM_INPUT = 1
AGENT_DIM_WIP = 1
AGENT_DIM_INPUT_VEC = 2
BATCH_DIM_MAPS = 0
MODE_DIM_PROB_PREDICTOR = 1
AGENT_DIM_PROB_PREDICTOR = 0
BATCH_DIM_SEQUENCE = 1
TRACKING_DIM_INPUT = -1

# samples format
SAMPLE_KEYS_REQUIRED = {
    "x",
    "y",
    "loss_masks",
    "map",
    "reference_pose",
    "index_target",
}
SAMPLE_KEYS_OPTIONAL = {
    "file_path",
    "reference_path",
    "target_route",
}

# Input format vectors
START_AND_END_POINT = 1
DELTA_AND_END_POINT = 2

# feature engineering
FIRST_VECTOR_ONE_SIDE = 0
LAST_VECTOR_ONE_SIDE = 8

# Output parameters dependent on loss type
OUTPUT_PARAMS_COORDINATES = 2
OUTPUT_PARAMS_GAUSSIAN = 5

# Indices based on dataloader
INDICES_COORDINATES = slice(0, 2)
INDICES_DELTAS = slice(2, 4)
INDICES_GLOBAL_COORDINATES = slice(10, 12)
TIMESTEPS_FOR_LENGTH = slice(0, 9)
INDICES_TRACKING_STATE_WIP = 2

# Birds eye box
X_CORNERS = [-1, 1, 1, -1]
Y_CORNERS = [-1, -1, 1, 1]
Z_CORNERS = [0, 0, 0, 0]
TILE_CENTER_SHAPE = (4, 1)

SINGLE_LOSSMASK_SHAPE = (1, 1, 1)
SINGLE_VALUE_SHAPE = (1, 1)
SINGLE_TUPLE_SHAPE = (1, 2)

# Plots
LINESWIDTH_POLY = 1
FIG_SIZE = 15
TRANSPARENT_WHITE_IMAGE = (255, 255, 255, 0)
Z_AXIS = [0, 0, 1]
SCALE_LIMIT_PLOT = 1.1

# Metrics
MINIMUM_NON_ZERO_VALUE = 1
STANDARD_INVALID_VALUE = -1
NO_VALUE = -1
VECTOR_SIZE_CARTESIAN = 2
VECTOR_SIZE_GAUSSIAN = 5

# Map related
RESNET_IMAGE_SIZE = 224
RESNET_IMAGENET_MEAN = [0.485, 0.456, 0.406]
RESNET_IMAGENET_STD_DEVIATION = [0.229, 0.224, 0.225]
RESNET18_OUTPUT_FEATURE_DIMENSION = 128
RESNET34_OUTPUT_FEATURE_DIMENSION = 128
RESNET50_OUTPUT_FEATURE_DIMENSION = 512
NUMBER_OF_RESNET_LAYERS_TO_REMOVE = 4

# Oversampling
OVERSAMPLING = 1
BIN_BOUNDARIES_TURN_KISSAF = [-np.inf, -1000, -0.1, 0.1, 1000, np.inf]

NO_VALUE_SAMPLE = (0, 0, 0)
POINT_PER_SUBGRAPH = 18
NBR_MAX_TIMESTEPS = 50
NBR_SUBGRAPH_POINTS = 18

LANE_WIDTH = {"MIA": 3.84, "PIT": 3.97}



STANDARD_MAP_LAYERS = [
    "drivable_area",
    "road_segment",
    "lane",
    "ped_crossing",
    "walkway",
    "stop_line",
    "road_divider",
    "lane_divider",
]
INTEGER_MAP_LAYERS = len(STANDARD_MAP_LAYERS)
MAP_IMAGE_SIZE = (128, 128)

MAP_COLORS = {
    "drivable_area": (131, 161, 129, 255),
    "road_segment": (120, 120, 120, 255),
    "road_block": (178, 223, 138, 255),
    "lane": (109, 160, 105, 255),
    "ped_crossing": (251, 154, 153, 255),
    "walkway": (228, 137, 137, 255),
    "stop_line": (253, 191, 111, 255),
    "carpark_area": (255, 127, 0, 255),
    "road_divider": (202, 178, 214, 255),
    "lane_divider": (51, 255, 255, 255),
    "traffic_light": (126, 119, 46, 255),
}  

LABEL_VALUES = {
    "vehicle.car": 0,
    "vehicle.bicycle": 1,
    "vehicle.bus.bendy": 2,
    "vehicle.construction": 3,
    "vehicle.motorcycle": 4,
    "vehicle.trailer": 5,
    "vehicle.truck": 6,
    "vehicle.bus.rigid": 2,
    "vehicle.emergency.ambulance": 7,
    "vehicle.emergency.police": 8,
    "human.pedestrian": 9,
    "human.pedestrian.adult": 9,
    "human.pedestrian.child": 9,
    "human.pedestrian.construction_worker": 9,
    "human.pedestrian.personal_mobility": 9,
    "human.pedestrian.police_officer": 9,
    "human.pedestrian.stroller": 9,
    "human.pedestrian.wheelchair": 9,
    "animal": 10,
}

# Color values for plotting
COLOR_INPUT_AGENT = (1.0, 0.5, 0.0)
COLOR_SHAPE = (0.067, 0.475, 0.749)
COLOR_VRU = (1, 0.0, 0.5)
COLOR_MOT_VEH = [153 / 255, 153 / 255, 0 / 255]
COLOR_ELSE_AGENT = (0.5, 0.5, 0.5)
COLOR_TRAFFIC_LIGHT = [153 / 255, 0 / 255, 76 / 255]
COLOR_INTERSECTION = (0.5, 0, 1)
COLOR_DEFAULT_LANE = [0, 0, 0]
COLOR_TARGET_FUTURE = [0 / 255, 255 / 255, 0 / 255]
COLOR_GT_TRAJECTORY = "black"
COLOR_PRED = [0 / 255, 0 / 255, 255 / 255]
COLOR_PRED_TARGET = "red"
COLOR_PLAN_TARGET = "purple"
COLOR_REFERENCE_PATH_PLANNER = "gold"

VRU_CLASSES = [1, 9]
MOT_VEH_CLASSES = [0, 2, 3, 4, 5, 6, 7, 8]


# initialized automatically
DELTA_TIME_STEP = None

# Initialization of Device for GPU/CPU selection
DEVICE = None

# Will be initialized the first time a path is tried to be created in get_unique_path
TIMESTAMP = None
LOGGER_PATH_CANDIDATE = None
ID_STRING = None

# default training parameters
default_lr_scheduler_stepLR_gamma = 0.3
default_lr_scheduler_stepLR_step_size = 5
default_lr_scheduler_exponential_gamma = 0.97

# Net - Values are taken from paper
# (start_coord, end_coord, object type, timestamps, road feature type)
VN_INPUT_SIZE_DYN = 7
# (start_coord, end_coord, object type, timestamps, road feature type)
VN_INPUT_SIZE_MAP = 6
VN_HIDDEN_SIZE = 64  # works the best according to author
ROUTE_CONDITION_SIZE = 30
ROUTE_CONDITION_FEATURES = 4

# Plotting related constants for Net
MIN_PROB_VISUALIZE = 0.1
SCALE_TRANSPARENCY = 1.33
MIN_TRANSPARENCY = 0.2
PLOT_STEPS_REFERENCE_PATH = 20

# Dynamic Encoder Types for all net architectures if selectable
SUB_GRAPH_DYN = 1
LSTM_DYN = 2
TRANSFORMER_LAST = 4
TRANSFORMER_SUM = 5
TRANSFORMER_MAXPOOL = 6

# Map Encoder Types for all net architectures if selectable
SUB_GRAPH_MAP = 1
LINEAR_MAP = 2
FIXED_FEATURES = 3

NBR_MAP_FEATURES = 6

# Route conditioning
LANE_BOX_MATCHING_THRESHOLD = 8.0
LANE_POINT_MATCHING_THRESHOLD = 3.0
LANE_POINT_YAW_MATCHING_THRESHOLD_MAX = math.radians(12)
LANE_POINT_YAW_MATCHING_THRESHOLD = math.radians(5)

# Attention mask visualization
NBR_NODES_ATTENTION = 5

MIN_SHAPE_LANENODE = (1, 1, 6)

# State Indices for kissaf data
class StateIndices(IntEnum):
    HIGHWAY = 0
    CONSTRUCTION = 1
    EXIT_MERGE = 2
    HW_EXIT_LEFT = 3
    HW_EXIT_RIGHT = 4
    CROSSING_LEFT = 5
    CROSSING_RIGHT = 6
    RegnCod = 7
    TURN_INDICATOR = 8
    LANE_CHANGE = 9
    HERE_SPEED_LIMIT = 10
    HERE_ENVIRONMENT_TYPE = 11
    HERE_DRIVING_DIRECTIONS = 12
    HERE_ILLUMINATION = 13
    INGGreen_ILLUMINATION = 14
    INGGreen_ROAD_TYPE = 15
    INGGreen_WEATHER = 16


class RoadTypes(IntEnum):
    URBAN = 0
    RURAL = 1
    HIGHWAY = 2
    MOUNTAIN = 4


HIGHWAY_TYPE = 1

# Interaction Mechanisms
SELF_ATTENTION = 1
MHA = 2


TRAJ_OUTPUT_PARAMETER_SIZE = 2

EPSILON_SAMPLE_TIME = 0.24
EPSILON_STABILITY = 0.000001


INDEX_AGENT_DIM_INP = 1
INDEX_PARAMETER_DIM = -1
MATCH_SHAPE = -1
INDEX_WIDTH_IN_SIZE = 0
INDEX_LENGTH_IN_SIZE = 1

PREDICTION_SHAPE_LENGTH = 4
PAIR_SHAPE_LENGTH = 2
EMPTY_SAMPLE_LEN = 0


# UNIT CONVERSION
METER_TO_KM = 0.001
SECOND_TO_HOUR = 1.0 / 3600.0

# ------------------------------------------------------------------------------
# Copyright:    Technical University Dortmund
# Project:      KISSaF
# Created by:   Institute of Control Theory and System Engineering
# ------------------------------------------------------------------------------

class SceneModelType(IntEnum):
    EPO_NET_F = 10
    SCENE_CONTROL_NET = 11

NUM_OF_STATES = 4
NUM_OF_INPUTS = 2
OPT_INIT_F_GT = 1
OPT_INIT_F_ZEROS = 2

# Constants for readible index access
# States
X_IND = 0
Y_IND = 1
THETA_IND = 2
V_IND = 3
VX_IND = 3
VY_IND = 4

# Controls
OMEGA_IND = 0
ACC_IND = 1


KEY_CENTERLINE = 'centerline'
KEY_L_BOUNDARY = 'left_boundaries'
KEY_R_BOUNDARY = 'right_boundaries'
KEY_INIT_CNTRL = 'agent_init_control_acc'
KEY_WIDTH = 'agent_widths'
KEY_LENGTH = 'agent_lengths'
KEY_INIT_STATE = 'init_state'
KEY_GOAL_STATE = 'goal_state'
KEY_LANELETMAP = 'lanelet_map'
MAX_NUMBER_CENTERLINE_POINTS = 25  # maximum length of centerlinepoints (used only in carla)
TARGET_LANE_AGENT_IDENTIFIER = 8


ENC_NORM = 1

# Waymo
# mph to m/s
MPH_TO_MPS = 0.44704
WAYMO = 8
FPS_WAYMO = 10

# Plotting
WAYMO_AXIS = 50 # the width/height of the rectangle ROI for plotting in [m]
