# ------------------------------------------------------------------------------
# Copyright:    ZF Friedrichshafen AG
# Project:      KISSaF
# Created by:   ZF AI LAB SBR
# ------------------------------------------------------------------------------
from typing import Callable

from torch.utils.data import DataLoader, Dataset, RandomSampler

import data.constants as c
import data.utils.waymo as duw

from data.constants import ROOT_DIR
from data.utils.vector_utils import collate_vector_scenes


def create_data(opts: dict, train=True, val=False, test=False, args: dict = None) -> Dataset:
    """Create dataloader for dataset and location specified in config.
    :param args: training arguments that here can
       - set debugging mode for kissaf data
    :param opts: scene prediction configs
    :param val: use validation or training dataset
    :return: dataset
    """

    dataset = opts["general"]["dataset"]

    if opts["general"]["dataset"] == c.WAYMO:
        if opts["general"]["mini"]:
            if val:
                dir = opts["general"]["dataset_waymo"] + '/mini/val/data'
            elif test:
                val = True
                dir = opts["general"]["dataset_waymo"] + '/mini/test/data'
            elif train:
                dir = opts["general"]["dataset_waymo"] + '/mini/train/data'
            return duw.WaymoData(opts, dir=dir, val=val, fps=c.FPS_WAYMO)
        else:
            if val:
                dir = opts["general"]["dataset_waymo"] + '/val/data'
            elif test:
                val = True  # for processing inside of Waymo
                dir = opts["general"]["dataset_waymo"] + '/test/data'
            elif train:
                dir = opts["general"]["dataset_waymo"] + '/train/data'

        return duw.WaymoData(cfgs=opts, dir=dir, val=val, fps=c.FPS_WAYMO)
    else:
        raise ValueError(f"Unknown dataset {dataset}. Please check the config and set dataset to 8.")


