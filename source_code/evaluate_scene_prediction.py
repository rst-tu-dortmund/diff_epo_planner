# ------------------------------------------------------------------------------
# Copyright:    Technical University Dortmund
# Project:      KISSaF
# Created by:   Institute of Control Theory and System Engineering
# ------------------------------------------------------------------------------
import argparse
import logging
import random


logging.getLogger('shapely.geos').setLevel(logging.CRITICAL)
import wandb
import json

from torch.utils.data import DataLoader
import torch
import numpy as np
import data.constants as c
import data.utils.helpers as du
from data import create_data
from data.configs import load_configs
from model import create_scene_prediction_model
import data.utils.waymo as duw

import torchinfo

torch.multiprocessing.set_sharing_strategy('file_system')

def set_seed(seed_value: int):
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    g = torch.Generator()
    g.manual_seed(seed_value)


def setup_data(data_test, project_configs):

    collate_fn = duw.collate_waymo_scenes

    data_loader_test = DataLoader(
        dataset=data_test,
        shuffle=False,
        num_workers=project_configs["config_params"]["number_workers"],
        drop_last=True,
        batch_size=project_configs["hyperparameters"]["batch_size"],
        collate_fn=collate_fn)

    print("\n############################################")
    print("############### DATALOADER #################")
    print("############################################\n")

    print(f"Test  Samples {len(data_loader_test.dataset)} @ {len(data_loader_test)} Batches")
    return data_loader_test


def set_logging_parameters(configs: dict, release: bool):
    """_summary_
    sets parameters for logging
    """
    if release:
        logging_level = logging.ERROR
    else:
        logging_level = logging.DEBUG

    log_file_dir, _ = du.get_storage_path(configs)
    log_file_name = log_file_dir / 'Scene-Prediction.log'

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if configs["general"]["write_logs"]:
        logging.basicConfig(filename=log_file_name, format='%(asctime)s : %(message)s', level=logging_level)
    else:
        logging.basicConfig(format='%(asctime)s : %(message)s', level=logging_level)


def init_wandb(configs: dict):
    """
    initializes the wandb config
    """
    if configs["general"]["dataset"] == c.WAYMO:
        wandb_run = wandb.init(project="EPO_WAYMO_Eval", entity=configs["wandb"]["entity"], config=configs)
    else:
       NotImplementedError("Dataset not implemented. Please check the config settings.")

    return wandb_run


def test(configs: dict, overlap_rate: bool = False):

    # Logging and parameter updates
    wandb_run = init_wandb(configs)


    # Setup data loader
    data_test = create_data(configs, train=False, test=True)
    data_loader_test = setup_data(data_test, configs)
    nbr_batches = len(data_loader_test)

    # Load the trained_model
    model_path = configs["general"]["model_path"]
    trained_model = torch.load(configs["general"]["model_path"])
    print("Loaded model "+  configs["general"]["model_path"])

    # Overwrite current_config
    configs = trained_model["opts"]
    configs["metrics"]["overlap_rate"] = overlap_rate
    wandb.config.update(configs, allow_val_change=True)

    # Setup  model
    model = create_scene_prediction_model(configs)

    # Load weights
    model.net.load_state_dict(trained_model["model_state_dict"])
    torchinfo.summary(model.net, depth=2)


    print("\n############################################")
    print("############### Evaluation #################")
    print("############################################\n")

    model.net.eval()
    model.netmode = "test"

    metrics_mean = model.evaluate(data_loader=data_loader_test, global_step=1, netmode="test", epoch=0)
    print(metrics_mean)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--release', help="release flag", action="store_true", default=False)
    parser.add_argument("--configs", help="configs.json file", default='configs_scnet.json')
    parser.add_argument("--overlap_rate", help="overlap rate", default=False, type=bool)
    args = parser.parse_args()
    configs = load_configs(args.configs)
    set_logging_parameters(configs, args.release)

    set_seed(configs["general"]["seed_value"])
    with open(args.configs, 'rt') as f:
        json_args = argparse.Namespace()
        json_args.__dict__.update(json.load(f))
    if configs:
        du.set_device_type(configs)
        test(configs, args.overlap_rate)
    else:
        logging.error("Error loading config.json!")