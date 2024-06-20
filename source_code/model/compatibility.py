# ------------------------------------------------------------------------------
# Copyright:    ZF Friedrichshafen AG
# Project:      KISSaF
# Created by:   ZF AI LAB SBR
# ------------------------------------------------------------------------------
import logging
import data.constants as c
from data.utils.helpers import set_device_type
import torch


def merge_options(opts: dict, restored_opts: dict):
    """If we want to reload a trained model it is possible that newly set
    parameters in the configutration are not compatible with the original
    model. Here we check which parametes should be restored from the original.

    :param opts:
    :type opts:
    :param restored_opts:
    :type restored_opts:
    :return:
    :rtype:
    """
    for key in c.OVERWRITE_KEYS.keys():
        merge_section(opts[key], restored_opts[key], c.OVERWRITE_KEYS[key])
    return opts


def merge_section(
    section: dict,
    restored_section: dict,
    keys_to_overwrite: list,
) -> dict:
    """Overwrite keys of the subsection of options.

    :param section:
    :type section:
    :param restored_section:
    :type restored_section:
    :param keys_to_overwrite:
    :type keys_to_overwrite:
    :return: merged dict
    :rtype:
    """
    for key in keys_to_overwrite:
        if section[key] != restored_section[key]:
            section[key] = restored_section[key]
            logging.info(
                f"Overwrite key {key} with {restored_section[key]} from "
                f"restored file",
            )
    return section


def restore_opts(opts: dict) -> dict:
    """return adapted opts for reloading.

    :param opts:
    :type opts:
    :return:
    :rtype:
    """
    checkpoint = torch.load(
        opts["general"]["model_path"],
        map_location=set_device_type(opts=opts),
    )
    opts_ckpt = checkpoint["opts"]
    merged_opts = merge_options(opts, opts_ckpt)
    return merged_opts
