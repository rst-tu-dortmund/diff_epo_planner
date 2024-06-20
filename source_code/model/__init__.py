# ------------------------------------------------------------------------------
# Copyright:    ZF Friedrichshafen AG
# Project:      KISSaF
# Created by:   ZF AI LAB SBR
# ------------------------------------------------------------------------------
import logging

import data.constants as const
import model.models as md
import model.vectorizednets as mv
import model.scene_prediction_models as spm
from model.models import create_lr_scheduler, create_optimizer
from model.compatibility import restore_opts

def create_model(opts) -> md.BaseClass:
    """
    Initializes a model based on given opts parameters
    Args:
        opts: Options derived from config.json

    Returns: A handle for the specified model. If restore_model is set to true then a handle is returned to the
    stored model location

    """
    if opts["general"]["restore_model"]:
        logging.info("Restoring model from {}".format(opts["general"]["model_path"]))
        opts = restore_opts(opts)
    elif opts["general"]["model"] == const.ModelType.V_Net:
        logging.info("Creating Vectorized Net")
        model = mv.VNetClass(opts)
    else:
        logging.error("Unrecognized model code. Please check config.json!")
        model = None

    if opts["general"]["restore_model"]:
        model.restore_model(opts)

    return model

# ------------------------------------------------------------------------------
# Copyright:    Technical University Dortmund
# Project:      KISSaF
# Created by:   Institute of Control Theory and System Engineering
# ------------------------------------------------------------------------------
def create_scene_prediction_model(opts) -> spm.ScenePredictionBaseClass:
    """
    Initializes a model based on given opts parameters
    :param opts: options derived from config.json
    :return: model
    """
    if opts["general"]["restore_model"]:
        logging.info("Restoring model from {}".format(opts["general"]["model_path"]))
        opts = restore_opts(opts)
    elif opts["general"]["model"] == const.SceneModelType.EPO_NET_F:
        logging.info("Creating Energy-based Potential Game Net")
        model = spm.EPONetFClass(opts)
    elif opts["general"]["model"] == const.SceneModelType.SCENE_CONTROL_NET:
        logging.info("Creating SceneControlNet")
        model = spm.SceneControlNetClass(opts)
    else:
        raise NotImplementedError("Unrecognized model. Please check config.json!")
    return model

