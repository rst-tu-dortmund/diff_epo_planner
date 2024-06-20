# ------------------------------------------------------------------------------
# Copyright:    ZF Friedrichshafen AG
# Project:      KISSaF
# Created by:   ZF AI LAB SBR
# ------------------------------------------------------------------------------
import argparse
import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Union, Optional

import data.constants as c

standard_config_path = os.path.join(c.ROOT_DIR, "source_code", "config.json")


@lru_cache(1)
def get_cli_config_arguments(default_config: Optional[str] = None):
    """Extract list of types for configuration arguments from nested dict. E.g.

    {"hyperparameters.learning_rate": 0.001} ->
    {"hyperparameters.learning_rate": float}

    :param default_config: path of config_json from which types are extracted.
    :return: dict of type {argument_name: argument_type}
    """

    def append_key(current_key: Optional[str], key: str):
        if current_key is None:
            return key
        else:
            return current_key + f".{key}"

    def get_cfg_types(cfgs_tmp: dict, current_key: Optional[str] = None):
        cfg_types = dict()
        for key, value in cfgs_tmp.items():
            if type(value) == dict:
                child_dict = get_cfg_types(
                    value,
                    current_key=append_key(current_key, key),
                )
                assert not set(child_dict.keys()).intersection(
                    set(cfg_types.keys()),
                ), f"keys {set(child_dict.keys()).intersection(set(cfg_types.keys()))} are duplicates in dict cfgs_tmp"
                cfg_types.update(child_dict)
            else:
                cfg_types[append_key(current_key, key)] = type(value)

        return cfg_types

    if default_config is None:
        default_config = Path(__file__).parent.parent / "config.json"

    try:
        cfgs = load_configs(default_config)
    except FileNotFoundError:
        cfgs = load_configs()

    return get_cfg_types(cfgs)


def load_configs(path: Union[Path, str] = None):
    configs = None
    if path is None:
        path = standard_config_path

    try:
        with open(path, "r") as config_file_object:
            configs = json.load(config_file_object)

        configs["hyperparameters"]["pixels_per_meter"] = (
            configs["hyperparameters"]["grid_size"]
            / configs["hyperparameters"]["region_of_interest"]
        )

    except FileNotFoundError as e:
        logging.error("%s not found!", path)
        raise e

    return configs


def overwrite_config_with_args(
    configs: dict,
    args: argparse.Namespace,
    parameter_names: Iterable[str],
) -> None:
    """Overwrite config parameters provided as command-line arguments in args
    (when arg is not None).

    :param configs: parameters loaded from config file.
    :param args: parsed command line arguments.
    :param parameter_names: list of attribute names in args that can overwrite config parameters,
        e.g. 'hyperparameters.epochs'.
    :return: None
    """
    for param in parameter_names:
        value = getattr(args, param)
        if value is None:
            continue
        key_path = param.split(".")
        cfg_tmp = configs
        for key in key_path[:-1]:
            cfg_tmp = cfg_tmp[key]
        cfg_tmp[key_path[-1]] = value


def parse_string_as_bool(string: str) -> bool:
    """
    :return: lambda for converting True/true strings as boolean values.
    """
    if str(string).lower() == "true":
        return True
    elif str(string).lower() == "false":
        return False
    else:
        raise ValueError(f"Expected on of (true, True, false, False), got {string}")
