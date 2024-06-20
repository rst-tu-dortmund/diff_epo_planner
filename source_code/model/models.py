# ------------------------------------------------------------------------------
# Copyright:    ZF Friedrichshafen AG
# Project:      KISSaF
# Created by:   ZF AI LAB SBR
# ------------------------------------------------------------------------------
import csv
import json
import logging
import statistics
import typing
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Optional
import numpy as np
import torch.cuda
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter

import data.constants as c
import data.utils.helpers as du
from model.interfaces.outputs import PredictionOutput, LSTMPredictionOutput
import model.losses as model_losses
import model.metrics
import model.modules
import model.multimode_modules as mmm
from data.utils.vector_utils import shorten_gt_to_prediction_length


class BaseClass(ABC):
    """Base class for prediction models."""

    def __init__(self, opts):
        self.opts = opts
        self.net: torch.nn.Module = None
        self.initialize_configurations(self.opts)
        self.initialize_hyperparameters(self.opts)
        self.initialize_metrics(self.opts)
        self.initialize_functions(self.opts)
        self.initialize_values(self.opts)

    def __del__(self):
        # Close the summary writer if it was opened
        if self.summary_writer_switch:
            self.summary_writer.close()

    def initialize_configurations(self, opts):
        self.device = c.DEVICE
        self.deltas = opts["config_params"]["deltas"]
        self.storage_path_logs, self.id_string = du.get_storage_path(opts)
        self.summary_writer_switch = opts["general"]["write_logs"]
        self.weigh_movement = opts["config_params"]["weigh_movement"]
        self.storage_path_logs, self.id_string = du.get_storage_path(opts)
        self.store_model_switch = opts["general"]["save_model"]
        if self.summary_writer_switch:
            self.summary_writer = SummaryWriter((self.storage_path_logs / "tb"))
            with open(self.storage_path_logs / "configs.json", "w") as fp:
                opts = du.adapt_config_to_store(opts, self.storage_path_logs)
                json.dump(opts, fp=fp, indent=4)
            # Set metrics log file name
            self.metrics_log_file = self.storage_path_logs / "metrics.csv"
        self.log_step =  opts["hyperparameters"]["log_step_training"]
        self.plot_step = opts["plotting"]["plot_step_training"]
        self.fps = du.get_fps_for_dataset(opts["general"]["dataset"])
        self.measure_time = opts["config_params"]["measure_time"]
        if self.opts["config_params"]["collision_loss"]:
            self.collision_loss = model_losses.ClassIndependentCollisionLoss(opts=opts)

    def initialize_hyperparameters(self, opts):
        self.batch_size = opts["hyperparameters"]["batch_size"]
        self.movement_threshold = opts["hyperparameters"]["movement_threshold"]
        self.movement_weight = opts["config_params"]["weigh_movement"]
        self.steps_per_interval = opts["config_params"]["steps_per_interval"]
        self.gradient_clip = opts["hyperparameters"]["gradient_clip"]
        self.decay_epoch = opts["hyperparameters"]["decay_epoch"]
        self.l1_regularization_weight = opts["hyperparameters"]["l1_regularization"]

    def initialize_metrics(self, opts):
        self.ade_switch = opts["metrics"]["average_displacement_error"]
        self.fde_switch = opts["metrics"]["final_displacement_error"]
        self.col_switch = opts["metrics"]["count_collisions"]

    def initialize_values(self, opts):
        self.metric_ade = c.NO_VALUE
        self.metric_ade_vru = c.NO_VALUE
        self.metric_ade_mot_veh = c.NO_VALUE
        self.metric_ade_closest = c.NO_VALUE
        self.metric_fde = c.NO_VALUE
        self.metric_fde_closest = c.NO_VALUE
        self.metric_fde_vru = c.NO_VALUE
        self.metric_fde_mot_veh = c.NO_VALUE
        self.brier_min_ade = c.NO_VALUE
        self.brier_min_fde = c.NO_VALUE
        self.collisions_count = c.NO_VALUE
        self.metric_ide = None
        self.cost_function_weights = c.NO_VALUE
        self.average_collisions = 0
        self.loss = 0
        self.optimizer: Optional[Optimizer] = None
        self.lr_scheduler = None
        self._epoch = 0
        self.gt = []
        if self.measure_time:
            self.times = []
            self.agents = []

    def initialize_functions(self, opts):
        self.loss_function = create_loss(opts)
        self.average_displacement_error = model.metrics.average_displacement_error
        self.final_displacement_error = model.metrics.final_displacement_error
        self.interval_displacement_error = model.metrics.interval_displacement_error
        self.collisions = model.metrics.collisions

    @staticmethod
    def initialize_average_metrics(
        compute_collision: bool,
        vectors: bool = False,
    ) -> dict:
        """Initialize dicts for collecting metrics.

        :param compute_collision: init collisin metrics
        :param vectors: initialize for vector base models
        :return: initialized dict
        """
        if vectors:
            metrics_init = {
                "ade": 0,
                "ade_closest": 0,
                "brier_ade_closest": 0,
                "fde": 0,
                "fde_closest": 0,
                "brier_fde_closest": 0,
                "ide": None,
            }

        else:
            metrics_init = {
                "ade": 0,
                "ade_vru": 0,
                "ade_mot_veh": 0,
                "ade_closest": 0,
                "cost_function_weights": 0,
                "fde": 0,
                "fde_vru": 0,
                "fde_mot_veh": 0,
                "fde_closest": 0,
                "ide": None,
            }
        if compute_collision:
            metrics_init["collisions"] = 0
        return metrics_init

    @staticmethod
    def cumulate_displacements(
        sample: dict,
        forward_pass: torch.Tensor,
        targets: bool,
        except_targets: bool = False,
    ) -> torch.Tensor:
        """function to calculate the positions relative to the reference point
        based on predicted displacements per timestep.

        :param sample:
        :param forward_pass:
        :param targets: True if forward pass contains targets only
        :param except_targets: True if targets are already in relative coordinates (due to a different target decoder)
        :return:
        """
        du.delta_to_relative(
            fw_pass=forward_pass,
            sample=sample,
            targets=targets,
            except_targets=except_targets,
        )
        return forward_pass

    def metrics_csv(
        self,
        epoch,
        ade,
        ade_vru,
        ade_mot_veh,
        fde,
        fde_vru,
        fde_mot_veh,
        collisions,
    ):
        if self.summary_writer_switch:
            with open(self.metrics_log_file, mode="a", newline="") as metrics_csv_file:
                csv_writer = csv.writer(metrics_csv_file, delimiter=",")
                if epoch == c.ZERO:  # Write header before the first row
                    csv_writer.writerow(
                        [
                            "epoch",
                            "ade",
                            "ade_vru",
                            "ade_mot_veh",
                            "fde",
                            "fde_vru",
                            "fde_mot_veh",
                            "collisions",
                        ],
                    )
                csv_writer.writerow(
                    [
                        epoch,
                        ade,
                        ade_vru,
                        ade_mot_veh,
                        fde,
                        fde_vru,
                        fde_mot_veh,
                        collisions,
                    ],
                )

    def get_metrics(self, training: bool = False) -> dict:
        """
        :param training: True if in training mode
        :return: Collected metrics in dict.
        """
        metric = dict()
        if self.ade_switch:
            metric["ade"] = self.metric_ade
            metric["ade_closest"] = self.metric_ade_closest
            metric["ade_vru"] = self.metric_ade_vru
            metric["ade_mot_veh"] = self.metric_ade_mot_veh
        if self.fde_switch:
            metric["fde"] = self.metric_fde
            metric["fde_vru"] = self.metric_fde_vru
            metric["fde_mot_veh"] = self.metric_fde_mot_veh
        if self.ide_switch:
            metric["ide"] = self.metric_ide
        if self.col_switch:
            metric["collisions"] = self.collisions_count
        return metric

    def update_logs(
        self,
        global_step: int,
        training: bool = True,
        metric: dict = None,
    ) -> typing.Optional[dict]:
        if self.summary_writer_switch:
            if metric is None:
                metric = self.get_metrics()
            if training:
                self.summary_writer.add_scalar("Training/Loss", self.loss, global_step)
                if global_step % self.log_step == 0:
                    if self.ade_switch and metric["ade"] != c.NO_VALUE:
                        self.summary_writer.add_scalar(
                            "Training/ADE/all_moving",
                            metric["ade"],
                            global_step,
                        )
                        if metric["ade_vru"] != c.NO_VALUE:
                            self.summary_writer.add_scalar(
                                "Training/ADE/VRUs",
                                metric["ade_vru"],
                                global_step,
                            )
                        if metric["ade_mot_veh"] != c.NO_VALUE:
                            self.summary_writer.add_scalar(
                                "Training/ADE/motorized_vehicles",
                                metric["ade_mot_veh"],
                                global_step,
                            )
            elif not training:
                if self.ade_switch and metric["ade"] != c.NO_VALUE:
                    self.summary_writer.add_scalar(
                        "Validation/ADE",
                        metric["ade"],
                        global_step,
                    )
                    if metric["ade_closest"] != c.NO_VALUE:
                        self.summary_writer.add_scalar(
                            "Validation/ADE/closest",
                            metric["ade_closest"],
                            global_step,
                        )
                    if metric["ade_vru"] != c.NO_VALUE:
                        self.summary_writer.add_scalar(
                            "Validation/ADE/VRUs",
                            metric["ade_vru"],
                            global_step,
                        )
                    if metric["ade_mot_veh"] != c.NO_VALUE:
                        self.summary_writer.add_scalar(
                            "Validation/ADE/motorized_vehicles",
                            metric["ade_mot_veh"],
                            global_step,
                        )
                if self.fde_switch and metric["fde"] != c.NO_VALUE:
                    self.summary_writer.add_scalar(
                        "Validation/FDE",
                        metric["fde"],
                        global_step,
                    )
                    if metric["fde_vru"] != c.NO_VALUE:
                        self.summary_writer.add_scalar(
                            "Validation/FDE/VRUs",
                            metric["fde_vru"],
                            global_step,
                        )
                    if metric["fde_mot_veh"] != c.NO_VALUE:
                        self.summary_writer.add_scalar(
                            "Validation/FDE/motorized_vehicles",
                            metric["fde_mot_veh"],
                            global_step,
                        )
                if self.col_switch and metric["collisions"] != c.NO_VALUE:
                    self.summary_writer.add_scalar(
                        "Validation/collsions",
                        metric["collisions"],
                        global_step,
                    )
                if self.ide_switch:
                    for i, curr_interval_metric in enumerate(metric["ide"]):
                        if curr_interval_metric != c.NO_VALUE:
                            self.summary_writer.add_scalar(
                                "Validation/IDE{}".format(i),
                                curr_interval_metric,
                                global_step,
                            )
        return metric

    def add_hparam_log(self):
        if self.summary_writer_switch:
            metrics = deepcopy(self.average_evaluation_metrics)
            metrics.pop("ide", None)
            params = {**self.opts["hyperparameters"], **self.opts["config_params"]}
            self.summary_writer.add_hparams(params, metrics)

    def optimize_parameters(self):

        self.optimizer.zero_grad()
        self.loss.backward()
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.gradient_clip)
        self.optimizer.step()
        self.loss = 0

    def lr_step(self, epoch):
        if (epoch >= self.decay_epoch) and (self.lr_scheduler is not None):
            self.lr_scheduler.step()
        self._epoch = epoch

    def store_model(self, epoch):
        if self.store_model_switch:
            model_storage_file_path = Path(self.storage_path_logs)
            model_storage_file_path.mkdir(parents=True, exist_ok=True)
            model_storage_file_path = model_storage_file_path / "model.pth"
            self.opts["general"]["model_path"] = model_storage_file_path
            torch.save(
                {
                    "epoch": epoch,
                    "fps": self.fps,
                    "model_state_dict": self.net.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "opts": self.opts,
                },
                model_storage_file_path,
            )

    def calculate_loss(self, sample, global_step):
        if self.opts["general"]["model"] == 3:
            assert self.deltas == False, (
                "Multi-modality is currently implemented for deltas set to false (true is "
                "deprecated)! "
            )

        out = self.forward(sample)
        fw_pass_output = out.prediction
        fw_pass_output_mode_prob = out.prediction_prob

        if self.deltas:
            self.loss += self.loss_function(
                pred=fw_pass_output,
                prob=fw_pass_output_mode_prob,
                sample=sample,
                delta=True,
            )
            fw_pass_output = self.cumulate_displacements(
                sample=sample,
                forward_pass=fw_pass_output,
                targets=False,
            )
        else:
            if not self.opts["config_params"]["direct_prediction"]:
                fw_pass_output = self.cumulate_displacements(
                    sample=sample,
                    forward_pass=fw_pass_output,
                    targets=False,
                )
            self.loss += self.loss_function(
                pred=fw_pass_output,
                prob=fw_pass_output_mode_prob,
                sample=sample,
            )
        if self.l1_regularization_weight != 0:
            self.loss += self.l1_regularization_weight * model_losses.l1_generalization(
                self.net,
            )

        if self.opts["config_params"]["collision_loss"]:
            self.loss += self.opts["hyperparameters"][
                "collision_loss_weight"
            ] * self.collision_loss(
                pred=fw_pass_output,
                prob=fw_pass_output_mode_prob,
                sample=sample,
            )
        if global_step % self.log_step == 0:
            fw_pass_output = mmm.get_highest_probability_mode(
                fw_pass_output,
                fw_pass_output_mode_prob,
                self.device,
            )
            self.calculate_trainingset_metrics(
                sample=sample,
                fw_pass_output=fw_pass_output,
                out=out,
            )

    def calculate_trainingset_metrics(
        self,
        sample,
        fw_pass_output,
        out: PredictionOutput,
    ) -> None:
        """Calculate metrics for current sample during training.

        :param sample: sample in vectorized format
        :param fw_pass_output: prediction output of model
        :param out: output of model
        :return: None
        """
        fw_pass_numpy = fw_pass_output.clone().detach().cpu().numpy()
        loss_masks = sample["loss_masks"].numpy()
        ground_truth = sample["y"].numpy()
        if self.weigh_movement:
            loss_masks = du.weigh_moving_agents(
                ground_truth=ground_truth,
                loss_masks=loss_masks,
                loss_mask_value=c.ZERO,
                movement_threshold=self.movement_threshold,
            )
        loss_masks[
            loss_masks > 0
        ] = 1  # negating possible weights used for loss for metric calculations
        loss_masks_vru = du.weigh_agents_class(
            ground_truth=ground_truth,
            loss_masks=loss_masks,
            agent_class_list=self.label_list_vru,
            zero_others=True,
        )
        loss_masks_mot_veh = du.weigh_agents_class(
            ground_truth=ground_truth,
            loss_masks=loss_masks,
            agent_class_list=self.label_list_mot_veh,
            zero_others=True,
        )
        if self.ade_switch:
            self.metric_ade = self.average_displacement_error(
                estimates=fw_pass_numpy[0, ...],
                ground_truth=ground_truth[..., 0:2],
                loss_mask=loss_masks,
            )
            self.metric_ade_vru = self.average_displacement_error(
                estimates=fw_pass_numpy[0, ...],
                ground_truth=ground_truth[..., 0:2],
                loss_mask=loss_masks_vru,
            )
            self.metric_ade_mot_veh = self.average_displacement_error(
                estimates=fw_pass_numpy[0, ...],
                ground_truth=ground_truth[..., 0:2],
                loss_mask=loss_masks_mot_veh,
            )
        if self.fde_switch:
            self.metric_fde = self.final_displacement_error(
                estimates=fw_pass_numpy[0, ...],
                ground_truth=ground_truth[..., 0:2],
                loss_mask=loss_masks,
            )
            self.metric_fde_vru = self.final_displacement_error(
                estimates=fw_pass_numpy[0, ...],
                ground_truth=ground_truth[..., 0:2],
                loss_mask=loss_masks_vru,
            )
            self.metric_fde_mot_veh = self.final_displacement_error(
                estimates=fw_pass_numpy[0, ...],
                ground_truth=ground_truth[..., 0:2],
                loss_mask=loss_masks_mot_veh,
            )
        if self.ide_switch:
            self.metric_ide = self.interval_displacement_error(
                estimates=fw_pass_numpy[0, ...],
                ground_truth=ground_truth[..., 0:2],
                loss_mask=loss_masks,
                steps_per_interval=self.steps_per_interval,
            )

    def calculate_mapped_metric(
        self,
        sample: dict,
        agent_flag: str = None,
        metric_flag: str = "ade_closest",
    ) -> dict:
        """
        So far only for batch size 1 and relative coordinates
        Args:
            sample:
            agent_flag:
            metric_flag:

        Returns:

        """
        current_metric = self.calculate_metric(
            sample=sample,
            agent_flag=agent_flag,
            metric_flag=metric_flag,
        )
        token = sample["sample_tokens"][c.FIRST_VALUE]
        results = dict()
        results[token] = current_metric
        return results

    def calculate_metric(
        self,
        sample: dict,
        agent_flag: str = None,
        metric_flag: str = "ade_closest",
        prediction: np.array = None,
    ) -> float:
        if prediction is None:
            trajectory = (
                self.select_mode(sample=sample, metric_flag=metric_flag)
                .detach()
                .cpu()
                .numpy()
            )
        else:
            trajectory = prediction

        metric_function = self.get_metric_function(metric_flag=metric_flag)
        loss_masks = self.get_lossmask(sample=sample, agent_flag=agent_flag)
        current_metric = metric_function(
            trajectory,
            sample["y"][..., c.INDICES_COORDINATES].numpy(),
            loss_masks,
        )
        return current_metric

    def select_mode(self, sample: dict, metric_flag: str):
        if self.measure_time and (self.device != "cpu"):
            start, end = du.init_events()
        out = self.forward(sample)
        relative_coordinates = self.cumulate_displacements(
            forward_pass=out.prediction,
            sample=sample,
            targets=self.opts["config_params"]["predict_target"],
        )
        if self.measure_time and (self.device != "cpu"):
            time = du.calculate_time(start=start, end=end)
            current_agent_number = sample["x"].shape[c.SECOND_VALUE]
            self.times.append(time)
            self.agents.append(current_agent_number)
        trajectory = None
        if "closest" in metric_flag:
            trajectory, _ = mmm.closest_pred_to_gt(
                multi_mode_y_pred=relative_coordinates,
                gt=sample["y"][..., c.INDICES_COORDINATES].to(self.device),
                loss_masks=sample["loss_masks"].to(self.device),
            )
        else:
            trajectory = mmm.get_highest_probability_mode(
                multi_mode_predictions=relative_coordinates,
                prob_distributions=out.prediction_prob,
                device=self.device,
            )
        return trajectory

    def get_metric_function(self, metric_flag: str) -> typing.Callable:
        if "ade" in metric_flag:
            return self.average_displacement_error
        elif "fde" in metric_flag:
            return self.final_displacement_error
        elif "ide" in metric_flag:
            return self.interval_displacement_error
        elif "collision" in metric_flag:
            return self.collisions
        else:
            logging.error("No metrics match your metric flag!")

    def get_lossmask(self, sample: dict, agent_flag: str = None) -> np.array:
        # only used in metric this movement is always filtered
        loss_masks = du.weigh_moving_agents(
            ground_truth=sample["y"].numpy(),
            loss_masks=sample["loss_masks"].numpy().copy(),
            movement_threshold=self.movement_threshold,
        )
        ground_truth = sample["y"].numpy()
        loss_masks[loss_masks > c.ZERO] = c.ONE
        if agent_flag == "vru":
            loss_masks = du.weigh_agents_class(
                ground_truth=ground_truth,
                loss_masks=loss_masks,
                agent_class_list=self.label_list_vru,
                zero_others=True,
            )
        elif agent_flag == "mot_veh":
            loss_masks = du.weigh_agents_class(
                ground_truth=ground_truth,
                loss_masks=loss_masks,
                agent_class_list=self.label_list_mot_veh,
                zero_others=True,
            )
        return loss_masks

    def calculate_all_metrics(self, sample: dict, collisions: bool = True) -> None:
        sample = sample
        self.net.eval()
        if self.measure_time and (self.device == "cpu"):
            logging.debug("currently time measurement is only supported on GPU")
        elif self.measure_time and (self.device != "cpu"):
            start, end = du.init_events()
        out = self.forward(sample)
        forward_pass = out.prediction
        forward_pass_mode_prob = out.prediction_prob
        if self.measure_time and (self.device != "cpu"):
            current_time = du.calculate_time(start=start, end=end)
            current_agent_number = sample["x"].shape[c.SECOND_VALUE]
            self.times.append(current_time)
            self.agents.append(current_agent_number)
            if len(self.times) > 1:
                logging.info(
                    "current time: {}; current agent number: {}; mean time: {}; mean agents: {}; std deviation time: "
                    "{}; std deviation agents: {}".format(
                        current_time,
                        current_agent_number,
                        statistics.mean(self.times),
                        statistics.mean(self.agents),
                        statistics.stdev(self.times),
                        statistics.stdev(self.agents),
                    ),
                )
        forward_pass = forward_pass.detach()
        if not self.opts["config_params"]["direct_prediction"]:
            forward_pass = self.cumulate_displacements(
                sample=sample,
                forward_pass=forward_pass,
                targets=False,
            )
        best_fw_pass, _ = mmm.closest_pred_to_gt(
            multi_mode_y_pred=forward_pass,
            gt=sample["y"][..., c.INDICES_COORDINATES].to(self.device),
            loss_masks=sample["loss_masks"].to(self.device),
            mode_prob=forward_pass_mode_prob,
        )
        best_fw_pass = best_fw_pass.unsqueeze(c.FIRST_VALUE).cpu().numpy()
        # handle multi-modality transparently
        most_prob_fw_pass = mmm.get_highest_probability_mode(
            forward_pass,
            forward_pass_mode_prob,
            self.device,
        )
        most_prob_fw_pass = most_prob_fw_pass.cpu().numpy()
        loss_masks = sample["loss_masks"].numpy()
        ground_truth = sample["y"].numpy()
        if self.weigh_movement:
            loss_masks = du.weigh_moving_agents(
                ground_truth=ground_truth,
                loss_masks=loss_masks,
                loss_mask_value=0,
                movement_threshold=self.movement_threshold,
            )
        loss_masks[loss_masks > 0] = 1
        loss_masks_vru = du.weigh_agents_class(
            ground_truth=ground_truth,
            loss_masks=loss_masks,
            agent_class_list=self.label_list_vru,
            zero_others=True,
        )
        loss_masks_mot_veh = du.weigh_agents_class(
            ground_truth=ground_truth,
            loss_masks=loss_masks,
            agent_class_list=self.label_list_mot_veh,
            zero_others=True,
        )
        if self.ade_switch:
            self.metric_ade = self.average_displacement_error(
                estimates=most_prob_fw_pass[0, ...],
                ground_truth=ground_truth[..., 0:2],
                loss_mask=loss_masks,
            )
            self.metric_ade_closest = self.average_displacement_error(
                estimates=best_fw_pass,
                ground_truth=ground_truth[..., c.INDICES_COORDINATES],
                loss_mask=loss_masks,
            )
            self.metric_ade_vru = self.average_displacement_error(
                estimates=most_prob_fw_pass[0, ...],
                ground_truth=ground_truth[..., 0:2],
                loss_mask=loss_masks_vru,
            )
            self.metric_ade_mot_veh = self.average_displacement_error(
                estimates=most_prob_fw_pass[0, ...],
                ground_truth=ground_truth[..., 0:2],
                loss_mask=loss_masks_mot_veh,
            )

        if self.fde_switch:
            self.metric_fde = self.final_displacement_error(
                estimates=most_prob_fw_pass[0, ...],
                ground_truth=ground_truth[..., 0:2],
                loss_mask=loss_masks,
            )
            self.metric_fde_vru = self.final_displacement_error(
                estimates=most_prob_fw_pass[0, ...],
                ground_truth=ground_truth[..., 0:2],
                loss_mask=loss_masks_vru,
            )
            self.metric_fde_mot_veh = self.final_displacement_error(
                estimates=most_prob_fw_pass[0, ...],
                ground_truth=ground_truth[..., 0:2],
                loss_mask=loss_masks_mot_veh,
            )
        if self.ide_switch:
            self.metric_ide = self.interval_displacement_error(
                estimates=most_prob_fw_pass[0, ...],
                ground_truth=ground_truth[..., 0:2],
                loss_mask=loss_masks,
                steps_per_interval=self.steps_per_interval,
            )
        if self.col_switch and collisions:
            self.collisions_count = self.collisions(
                prediction=most_prob_fw_pass[0, ...],
                ground_truth=ground_truth,
            )
        self.net.train()

    def evaluate(self, data_loader_validation, epoch):
        compute_collisions = (
            (epoch + 1) % self.frequency_collision_calculation == 0
        ) and self.col_switch
        average = self.initialize_average_metrics(self.col_switch)
        counter = self.initialize_average_metrics(self.col_switch)
        for i, mini_batch in enumerate(data_loader_validation):
            mini_batch = shorten_gt_to_prediction_length(
                mini_batch,
                self.opts["hyperparameters"]["prediction_length"],
                self.fps,
            )
            self.calculate_all_metrics(mini_batch, collisions=compute_collisions)
            if i % self.opts["config_params"]["print_step"] == 0:
                logging.info(
                    f"Calculated current metric of validation sample {i}: "
                    f"ADE: {self.metric_ade}; "
                    f"FDE: {self.metric_fde}; "
                    f"Coll.: {self.collisions_count}",
                )
            if self.metric_ade != c.NO_VALUE:
                counter["ade"] += 1
                average["ade"] += self.metric_ade
            if self.metric_ade_closest != c.NO_VALUE:
                counter["ade_closest"] += 1
                average["ade_closest"] += self.metric_ade_closest
            if self.metric_ade_vru != c.NO_VALUE:
                counter["ade_vru"] += 1
                average["ade_vru"] += self.metric_ade_vru
            if self.metric_ade_mot_veh != c.NO_VALUE:
                counter["ade_mot_veh"] += 1
                average["ade_mot_veh"] += self.metric_ade_mot_veh
            if self.metric_fde != c.NO_VALUE:
                counter["fde"] += 1
                average["fde"] += self.metric_fde
            if self.metric_fde_vru != c.NO_VALUE:
                counter["fde_vru"] += 1
                average["fde_vru"] += self.metric_fde_vru
            if self.metric_fde_mot_veh != c.NO_VALUE:
                counter["fde_mot_veh"] += 1
                average["fde_mot_veh"] += self.metric_fde_mot_veh
            if self.metric_ide is not None:
                if counter["ide"] is None:
                    counter["ide"] = np.zeros_like(self.metric_ide)
                    average["ide"] = np.zeros_like(self.metric_ide)
                for j in range(len(counter["ide"])):
                    if self.metric_ide[j] != c.NO_VALUE:
                        counter["ide"][j] += 1
                        average["ide"][j] += self.metric_ide[j]
            if self.collisions_count != c.NO_VALUE and compute_collisions:
                counter["collisions"] += 1
                average["collisions"] += self.collisions_count
        if self.ade_switch and (counter["ade"] > 0):
            average["ade"] = average["ade"] / counter["ade"]
            if counter["ade_closest"] > 0:
                average["ade_closest"] = average["ade_closest"] / counter["ade_closest"]
            if counter["ade_vru"] > 0:
                average["ade_vru"] = average["ade_vru"] / counter["ade_vru"]
            if counter["ade_mot_veh"] > 0:
                average["ade_mot_veh"] = average["ade_mot_veh"] / counter["ade_mot_veh"]
        if self.fde_switch and (counter["fde"] > 0):
            average["fde"] = average["fde"] / counter["fde"]
            if counter["fde_vru"] > 0:
                average["fde_vru"] = average["fde_vru"] / counter["fde_vru"]
            if counter["fde_mot_veh"] > 0:
                average["fde_mot_veh"] = average["fde_mot_veh"] / counter["fde_mot_veh"]
        if self.ide_switch and (np.sum(counter["ide"]) > 0):
            average["ide"] = average["ide"] / counter["ide"]
        if compute_collisions:
            average["collisions"] = average["collisions"] / counter["collisions"]
        self.update_logs(global_step=epoch, training=False, metric=average)
        # self.metrics_csv(epoch, average_ade, average_ade_vru, average_ade_mot_veh, average_fde, average_fde_vru,
        #                  average_fde_mot_veh, self.average_collisions)

        logging.info(
            "Metric on whole validation dataset in epoch {} : ADE: {}; FDE: {}"
            "".format(epoch, average["ade"], average["fde"]),
        )
        logging.info(
            "{} samples with VRUs and {} samples with motorized vehicles considered for FDE calculations(VRU {}; MV "
            "{}".format(
                counter["fde_vru"],
                counter["fde_mot_veh"],
                average["fde_vru"],
                average["fde_mot_veh"],
            ),
        )

    def restore_model(self, cfgs: dict) -> None:
        """
        Restores parameters from pretrained model.
        :return: None
        :param cfgs: configs with relevant config parameters merged from pretrained model.
        """
        net_state_dict, optimizer_state_dict, epoch, fps = restore_model(cfgs)
        if self.net:
            self.net.load_state_dict(net_state_dict)

        if (
            "finetune_restored_model" in cfgs["general"]
            and cfgs["general"]["finetune_restored_model"]
        ):
            # start new training initialized with restored model
            epoch = 0
            self.optimizer = create_optimizer(self, cfgs)
            if cfgs["config_params"]["lr_scheduler"] != 0:
                self.lr_scheduler = create_lr_scheduler(self, cfgs)
        else:
            # continue training of restored model
            self.optimizer.load_state_dict(optimizer_state_dict)
            self.optimizer.param_groups[0][
                "capturable"
            ] = True  # prevents errors when continuing training of restored models (might not be required in future
            # pytorch release)

        self._epoch = epoch
        if fps is not None:
            self.fps = fps
        else:
            print(
                f"fps not stored in model checkpoint. Use data-specific value {self.fps}",
            )

    @abstractmethod
    def forward(self, sample: dict) -> PredictionOutput:
        """Predicts trajectories of agents given the data in sample.

        :param sample: see README.md -> "Data formats" for details on the format.
        :return: predictions
        """
        pass

    @abstractmethod
    def draw_model(self, sample):
        pass

    def get_calibration_parameters(self):
        raise NotImplementedError(
            "Derived models are calibratible only if get_calibration_parameters is implemented",
        )



def create_optimizer(base_model_class, opts):
    if opts["general"]["optimizer"] == 1:
        return torch.optim.Adam(
            base_model_class.net.parameters(),
            lr=opts["hyperparameters"]["learning_rate"],
            weight_decay=opts["hyperparameters"]["weight_decay"],
        )
    else:
        raise NotImplementedError

def create_loss(opts):
    if opts["general"]["loss"] == 1:
        return model.losses.Gaussian_Nll(opts)
    elif opts["general"]["loss"] == 2:
        return model.losses.NegativeMultiLogLikelihood()
    elif opts["general"]["loss"] == 3:
        return model.losses.VariatyRegressionAndClassification(opts=opts)
    else:
        raise ValueError(f"Unknown loss {opts['general']['loss']}")


def restore_model(opts: dict) -> tuple:
    """reloads trained model weigths and optimizer state.

    :param opts: Training configurations with model specifications
    :type opts:
    :return: returns state dicts of model and optimizer
    :rtype:tuple
    """
    checkpoint = torch.load(
        opts["general"]["model_path"],
        map_location=du.set_device_type(opts=opts),
    )
    model_state_dict = checkpoint["model_state_dict"]
    optimizer_state_dict = checkpoint["optimizer_state_dict"]
    epoch = checkpoint["epoch"] + 1
    if "fps" in checkpoint:
        fps = checkpoint["fps"]
    else:
        fps = None

    return model_state_dict, optimizer_state_dict, epoch, fps


def create_lr_scheduler(base_model_class, opts):
    if opts["config_params"]["lr_scheduler"] == 1:
        if "lr_scheduler_exponential_gamma" in opts["config_params"]:
            lr_scheduler_exponential_gamma = opts["config_params"][
                "lr_scheduler_exponential_gamma"
            ]
        else:
            lr_scheduler_exponential_gamma = c.default_lr_scheduler_exponential_gamma

        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer=base_model_class.optimizer,
            gamma=lr_scheduler_exponential_gamma,
        )
    elif opts["config_params"]["lr_scheduler"] == 2:
        if "lr_scheduler_stepLR_step_size" in opts["config_params"]:
            lr_scheduler_stepLR_step_size = opts["config_params"][
                "lr_scheduler_stepLR_step_size"
            ]
        else:
            lr_scheduler_stepLR_step_size = c.default_lr_scheduler_stepLR_step_size

        if "lr_scheduler_stepLR_gamma" in opts["config_params"]:
            lr_scheduler_stepLR_gamma = opts["config_params"][
                "lr_scheduler_stepLR_gamma"
            ]
        else:
            lr_scheduler_stepLR_gamma = c.default_lr_scheduler_stepLR_gamma

        return torch.optim.lr_scheduler.StepLR(
            optimizer=base_model_class.optimizer,
            step_size=lr_scheduler_stepLR_step_size,
            gamma=lr_scheduler_stepLR_gamma,
        )