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

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange
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

def setup_data(data_train: Dataset, data_val: Dataset, project_configs: dict):

    if project_configs["general"]["dataset"] == c.WAYMO:
        # waymo does not have an implemented collate function
        collate_fn = duw.collate_waymo_scenes
    else:
        raise NotImplementedError


    data_loader = DataLoader(
        dataset=data_train,
        shuffle=project_configs["config_params"]["shuffle"],
        num_workers=project_configs["config_params"]["number_workers"],
        drop_last=project_configs["hyperparameters"]["drop_last"],
        batch_size=project_configs["hyperparameters"]["batch_size"],
        collate_fn=collate_fn,
        pin_memory=True)
    data_loader_validation = DataLoader(
        dataset=data_val,
        shuffle=project_configs["config_params"]["shuffle"],
        num_workers=project_configs["config_params"]["number_workers"],
        drop_last=project_configs["hyperparameters"]["drop_last"],
        batch_size=project_configs["hyperparameters"]["batch_size"],
        collate_fn=collate_fn,
        pin_memory=True)
    print("\n############################################")
    print("############### DATALOADER #################")
    print("############################################\n")

    print(f"Train Samples {len(data_loader.dataset)} @ {len(data_loader)} Batches")
    print(f"Val   Samples {len(data_loader_validation.dataset)} @ {len(data_loader_validation)} Batches")
    return data_loader, data_loader_validation


def set_logging_parameters(configs: dict, release: bool):
    """_summary_
    Args:
        opts (dict): _description_
        release (bool): _description_
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


def init_wandb(configs: dict, curr_tags) -> wandb.run:
    if configs["general"]["dataset"] == c.WAYMO:
        wandb_run = wandb.init(project = configs["wandb"]["project"],
                               entity=configs["wandb"]["entity"],
                               tags=curr_tags)
    else:
        raise NotImplementedError

    return wandb_run


def train(configs: dict):

    # Logging and wandb settings updates
    curr_tags = []
    wandb_run = init_wandb(configs, curr_tags)
    configs["wandb"]["training_name"] = wandb_run.name
    wandb.config.update(configs, allow_val_change=True)



    # Setup data loader
    data_train = create_data(configs, train=True)
    data_val = create_data(configs, train=False, val=True)
    data_loader, data_loader_validation = setup_data(data_train, data_val, configs)
    nbr_batches = len(data_loader)

    # Setup neural network
    model = create_scene_prediction_model(configs)
    torchinfo.summary(model.net, depth=2)

    print("\n############################################")
    print("############## Start Training ################")
    print("############################################\n")

    for epoch in trange(configs["hyperparameters"]["epochs"], desc='Epoch: ', position=0):
        log_learning_rate(model, epoch, configs)

        for i, train_mini_batch in enumerate(tqdm(data_loader, desc='Batch: ', leave=False, position=1)):
            global_step = epoch * nbr_batches + i + 1


            loss_dict, eval_dict = model.predict_and_calculate_loss(train_mini_batch, netmode="train", global_step=global_step, epoch=epoch)
            loss = model.extract_loss(loss_dict)
            model.optimize_parameters(loss)

        model.lr_step(epoch=epoch)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.net.state_dict(),
            'optimizer_state_dict': model.optimizer.state_dict(),
            'opts': configs,
        }, configs["general"]["storage_path"] + "/" + wandb_run.name.replace('-', '_') + f"_E{epoch}.pth")


        # Validation
        model.netmode = "val"
        model.evaluate(data_loader = data_loader_validation, global_step=global_step, netmode="val", epoch=epoch)



def log_learning_rate(model, epoch, configs):
    if configs["config_params"]["lr_scheduler"] != 0:
        learning_rate = model.lr_scheduler.get_last_lr()[0]
    if configs["config_params"]["lr_scheduler"] != 0:
        wandb.log({'epoch': epoch, "Optimization/learning rate": learning_rate})
    else:
        wandb.log({'epoch': epoch, "Optimization/learning rate": model.optimizer.defaults["lr"]})

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--release', help="release flag", action="store_true", default=False)
    parser.add_argument("--configs", help="configs.json file", default='configs_scnet.json')

    args = parser.parse_args()
    configs = load_configs(args.configs)
    set_logging_parameters(configs, args.release)

    set_seed(configs["general"]["seed_value"])
    with open(args.configs, 'rt') as f:
        json_args = argparse.Namespace()
        json_args.__dict__.update(json.load(f))
    if configs:
        du.set_device_type(configs)
        train(configs)
    else:
        logging.error("Error loading config.json!")
