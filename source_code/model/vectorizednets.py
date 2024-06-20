# ------------------------------------------------------------------------------
# Copyright:    ZF Friedrichshafen AG
# Project:      KISSaF
# Created by:   ZF AI LAB SBR
# ------------------------------------------------------------------------------
import logging
import statistics
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Tuple, Optional, Union, Type

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import data.constants as c
import data.utils.helpers as du
import data.utils.vector_utils as duv
import model.losses as model_losses
import model.metrics
import model.modules
import model.multimode_modules as mmm
from model.interfaces.outputs import (
    PredictionOutput,
    AttentionPredictionOutput,
    LSTMAttentionPredictionOutput,
)
import torch


class VectorBaseClass(model.models.BaseClass, ABC):
    """net output has to have format (modes, sequence, batch, agents, 2)"""

    net_class: model.modules.VectorBase = None

    def __init__(self, opts):
        super(VectorBaseClass, self).__init__(opts)
        self.configs = opts
        self.env_step = 0
        if self.net_class is not None:
            self.net = self.net_class(opts=self.opts, device=self.device)

    def _calculate_loss(
        self,
        out: PredictionOutput,
        sample: dict,
        cum_fw_pass_output: torch.Tensor,
        cumred_fw_pass_output: torch.Tensor,
        fw_pass_output: torch.Tensor,
        fw_pass_output_mode_prob: torch.Tensor,
        global_step: int,
    ):
        """

        :param sample:
        :param fw_pass_output:
        :param fw_pass_output_mode_prob:
        :param global_step:
        :return:
        """
        if self.deltas:
            self.loss += self.loss_function(
                pred=fw_pass_output,
                prob=fw_pass_output_mode_prob,
                sample=sample,
                delta=True,
            )
        else:
            self.loss += self.loss_function(
                pred=cumred_fw_pass_output,
                prob=fw_pass_output_mode_prob,
                sample=sample,
            )

        if self.l1_regularization_weight != 0:
            self.loss += self.l1_regularization_weight * model_losses.l1_generalization(
                self.net,
            )
        if global_step % self.log_step == 0:
            fw_pass_output = mmm.get_highest_probability_mode(
                cumred_fw_pass_output,
                fw_pass_output_mode_prob,
                self.device,
            )
            self.calculate_trainingset_metrics(
                sample=sample,
                fw_pass_output=fw_pass_output,
                out=out,
            )

    def calculate_loss(self, sample: dict, global_step: int) -> None:
        """Calculate loss for sample.

        :param sample: vector sample
        :param global_step: total steps
        :return: None
        """
        out = self.forward(sample)
        (
            cum_forward_pass,
            cumred_forward_pass,
            red_forward_pass,
            red_forward_pass_mode_prob,
        ) = self._reduce_batch_and_cumulate_deltas(out, sample)
        self._calculate_loss(
            out,
            sample,
            cum_forward_pass,
            cumred_forward_pass,
            red_forward_pass,
            red_forward_pass_mode_prob,
            global_step,
        )

    def calculate_trainingset_metrics(
        self,
        sample,
        fw_pass_output,
        out: PredictionOutput,
    ) -> None:
        """Calculate metrics for current sample during training.

        :param sample: sample in Vector Format
        :param fw_pass_output: prediction output of model
        :param out: output of model
        :return: None
        """
        fw_pass_numpy = fw_pass_output.clone().detach().cpu().numpy()
        ground_truth = du.format_gt(sample, self.opts).numpy()
        loss_masks = du.format_loss_masks(sample, self.opts).numpy()
        if self.ade_switch:
            self.metric_ade = self.average_displacement_error(
                estimates=fw_pass_numpy[0, ...],
                ground_truth=ground_truth[..., 0:2],
                loss_mask=loss_masks,
            )
        if self.fde_switch:
            self.metric_fde = self.final_displacement_error(
                estimates=fw_pass_numpy[0, ...],
                ground_truth=ground_truth[..., 0:2],
                loss_mask=loss_masks,
            )

    def get_metrics(self, training: bool = False) -> dict:
        """
        :param training: True if in training mode
        :return: Collected metrics in dict.
        """
        metric = {
            "ade": c.NO_VALUE,
            "ade_closest": c.NO_VALUE,
            "brier_ade_closest": c.NO_VALUE,
            "fde": c.NO_VALUE,
            "fde_closest": c.NO_VALUE,
            "brier_fde_closest": c.NO_VALUE,
            "ide": None,
        }
        if self.ade_switch:
            metric["ade"] = self.metric_ade
            metric["ade_closest"] = self.metric_ade_closest
            metric["brier_ade_closest"] = self.brier_min_ade
        if self.fde_switch:
            metric["fde"] = self.metric_fde
            metric["fde_closest"] = self.metric_fde_closest
            metric["brier_fde_closest"] = self.brier_min_fde
        return metric

    def update_logs(
        self,
        global_step: int,
        training: bool = True,
        metric: dict = None,
    ) -> Optional[dict]:
        if self.summary_writer_switch:
            if metric is None:
                metric = self.get_metrics(training=training)

            self.log_wandb(
                global_step,
                training,
                metric,
            )
            if training:
                self.summary_writer.add_scalar("Training/Loss", self.loss, global_step)
                if global_step % self.log_step == 0:
                    if self.ade_switch and metric["ade"] != c.NO_VALUE:
                        self.summary_writer.add_scalar(
                            "Training/ADE/all_moving",
                            metric["ade"],
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
                    if metric["brier_ade_closest"] != c.NO_VALUE:
                        self.summary_writer.add_scalar(
                            "Validation/brierADE/closest",
                            metric["brier_ade_closest"],
                            global_step,
                        )
                if self.fde_switch and metric["fde"] != c.NO_VALUE:
                    self.summary_writer.add_scalar(
                        "Validation/FDE",
                        metric["fde"],
                        global_step,
                    )
                    if metric["fde_closest"] != c.NO_VALUE:
                        self.summary_writer.add_scalar(
                            "Validation/FDE/closest",
                            metric["fde_closest"],
                            global_step,
                        )
                    if metric["brier_fde_closest"] != c.NO_VALUE:
                        self.summary_writer.add_scalar(
                            "Validation/brierFDE/closest",
                            metric["brier_fde_closest"],
                            global_step,
                        )
        return metric

    def log_wandb(
        self,
        global_step: int,
        training: bool = True,
        metric: dict = None,
    ) -> None:
        """

        :return:
        """
        if not self.configs["wandb"]["write_logs"]:
            return

        if training:
            caption = "training"
            step_name = "batch"
        else:
            caption = "validation"
            step_name = "epoch"

        log_dict = {}
        if training:
            self.summary_writer.add_scalar("Training/Loss", self.loss, global_step)
            if global_step % self.log_step == 0:
                log_dict[f"{caption}/loss"] = self.loss
                if self.ade_switch and metric["ade"] != c.NO_VALUE:
                    log_dict[f"{caption}/ade"] = metric["ade"]

        elif not training:
            if self.ade_switch and metric["ade"] != c.NO_VALUE:
                log_dict[f"{caption}/ade"] = metric["ade"]
                if metric["ade_closest"] != c.NO_VALUE:
                    log_dict[f"{caption}/ade_closest"] = metric["ade_closest"]
                if metric["brier_ade_closest"] != c.NO_VALUE:
                    log_dict[f"{caption}/brier_ade_closest"] = metric[
                        "brier_ade_closest"
                    ]
            if self.fde_switch and metric["fde"] != c.NO_VALUE:
                log_dict[f"{caption}/ade_closest"] = metric["ade_closest"]
                if metric["fde_closest"] != c.NO_VALUE:
                    log_dict[f"{caption}/fde_closest"] = metric["fde_closest"]
                if metric["brier_fde_closest"] != c.NO_VALUE:
                    log_dict[f"{caption}/ade_closest"] = metric["ade_closest"]

        log_dict[step_name] = global_step

    def select_mode(self, sample: dict, metric_flag: str):
        """NOT IMPLEMENTED SINCE FIRST IMPL ONLY ONE MODE.

        :param sample:
        :param metric_flag:
        :return:
        """
        print("Mulitmodality for vector based approaches pending")
        pass

    def get_lossmask(self, sample: dict, agent_flag: str = None) -> np.array:
        """get lossmask to assess for which predicted point a gt is available
        :param sample:
        :param agent_flag:
        :return:
        """
        loss_masks = sample["loss_masks"].numpy().copy()
        shape = loss_masks.shape
        loss_masks = du.reduce_batch_dim(loss_masks)
        loss_masks = loss_masks.reshape(shape)
        return loss_masks

    def _reduce_batch_and_cumulate_deltas(
        self,
        out: PredictionOutput,
        sample: dict,
        except_targets: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reduce batch dimension and cumulated predicted deltas when not using
        'direct_prediction' option.

        :param out: output of forward pass
        :param sample: sample in vector format
        :param except_targets: don't cumulate trajectories of target when True
        :return: cumulated and reduced dimension:cum_forward_pass, cumred_forward_pass
                 reduced dimension only: red_forward_pass, red_forward_pass_mode_prob,
        """
        red_forward_pass = du.reduce_batch_dim(out.prediction)
        red_forward_pass_mode_prob = du.reduce_batch_dim(out.prediction_prob)
        cum_forward_pass = out.prediction
        cumred_forward_pass = red_forward_pass
        if not self.opts["config_params"]["direct_prediction"]:
            forward_pass_shape = out.prediction.shape
            cumred_forward_pass = self.cumulate_displacements(
                forward_pass=red_forward_pass.clone(),
                sample=sample,
                targets=self.opts["config_params"]["predict_target"],
                except_targets=except_targets,
            )
            cum_forward_pass = cumred_forward_pass.view(forward_pass_shape)

        return (
            cum_forward_pass,
            cumred_forward_pass,
            red_forward_pass,
            red_forward_pass_mode_prob,
        )

    def _postprocess_for_metrics(
        self,
        sample,
        forward_pass_reduced,
        forward_pass_mode_prob,
    ):
        """Postprocess forward pass before computing metrics.

        :param sample: sample for which prediction was computed
        :param forward_pass_reduced: forward pass
        :param forward_pass_mode_prob:
        :return: best_fw_pass, best_mode_prob, ground_truth, loss_mask, most_prob_fw_pass
        """
        forward_pass = forward_pass_reduced.detach()
        forward_pass_mode_prob = forward_pass_mode_prob.detach()
        ground_truth = du.format_gt(sample, self.opts)
        loss_mask = du.format_loss_masks(sample, self.opts)
        best_fw_pass, best_mode_prob = mmm.closest_pred_to_gt(
            multi_mode_y_pred=forward_pass,
            gt=ground_truth.to(self.device)[..., c.INDICES_COORDINATES],
            loss_masks=loss_mask.to(self.device),
            mode_prob=forward_pass_mode_prob,
        )
        best_fw_pass = best_fw_pass.unsqueeze(c.FIRST_VALUE).cpu().numpy()[0, ...]
        best_mode_prob = best_mode_prob.cpu().detach().numpy()
        ground_truth = ground_truth.cpu().numpy()
        loss_mask = loss_mask.cpu().numpy()
        # handle multi-modality transparently
        most_prob_fw_pass = (
            mmm.get_highest_probability_mode(
                forward_pass,
                forward_pass_mode_prob,
                self.device,
            )
            .cpu()
            .detach()
            .numpy()[0, ...]
        )
        return best_fw_pass, best_mode_prob, ground_truth, loss_mask, most_prob_fw_pass

    def _compute_metrics(
        self,
        best_fw_pass: np.ndarray,
        best_mode_prob: np.ndarray,
        most_prob_fw_pass: np.ndarray,
        ground_truth: np.ndarray,
        loss_mask: np.ndarray,
        out: PredictionOutput,
        sample: dict,
        compute_brier: bool = True,
    ) -> None:
        """Evaluate the predicted trajectories of all agents.

        :param best_fw_pass: predictions of best (closest to ground truth) mode for each agent
        :param best_mode_prob: corresponding probabilities of best_fw_pass
        :param most_prob_fw_pass: predictions of modes with highest predicted probabilities
        :param ground_truth: ground truth of predictions
        :param loss_mask: masks states of predictions that should be considered in metrics
        :param out: output of prediction
        :param sample: current sample
        :param compute_brier: compute brier ADE and FDE
        :return: None
        """
        if self.ade_switch:
            self.metric_ade = self.average_displacement_error(
                estimates=most_prob_fw_pass,
                ground_truth=ground_truth[..., c.INDICES_COORDINATES],
                loss_mask=loss_mask,
            )
            self.metric_ade_closest = self.average_displacement_error(
                estimates=best_fw_pass,
                ground_truth=ground_truth[..., c.INDICES_COORDINATES],
                loss_mask=loss_mask,
            )
            if compute_brier:
                self.brier_min_ade = model.metrics.brier_ade(
                    estimates=best_fw_pass,
                    ground_truth=ground_truth[..., c.INDICES_COORDINATES],
                    loss_mask=loss_mask,
                    probabilities=best_mode_prob,
                )

        if self.fde_switch:
            self.metric_fde = self.final_displacement_error(
                estimates=most_prob_fw_pass,
                ground_truth=ground_truth[..., 0:2],
                loss_mask=loss_mask,
            )
            self.metric_fde_closest = self.final_displacement_error(
                estimates=best_fw_pass,
                ground_truth=ground_truth[..., c.INDICES_COORDINATES],
                loss_mask=loss_mask,
            )
            if compute_brier:
                self.brier_min_fde = model.metrics.brier_fde(
                    estimates=best_fw_pass,
                    ground_truth=ground_truth[..., c.INDICES_COORDINATES],
                    loss_mask=loss_mask,
                    probabilities=best_mode_prob,
                )

    def measure_time_start(
        self,
    ) -> Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]]:
        """Start measuring computation time if option is activated.

        :return: None
        """
        if self.measure_time and (self.device == "cpu"):
            logging.info("currently time measurement is only supported on GPU")
        elif self.measure_time and (self.device != "cpu"):
            start, end = du.init_events()
            return start, end

        return None, None

    def log_measure_time_end(
        self,
        start: Optional[torch.cuda.Event],
        end: Optional[torch.cuda.Event],
        sample: dict,
    ) -> None:
        """Stop measuring computation time and write to log if option is
        activated.

        :return: None
        """

        if self.measure_time and (self.device != "cpu"):
            current_time = du.calculate_time(start=start, end=end)
            current_agent_number = sample["x"].shape[c.SECOND_VALUE]
            self.times.append(current_time)
            self.agents.append(current_agent_number)
            if len(self.times) > 1:
                logging.info(
                    "current time: {}; current agent number: {}; mean time: {};"
                    " mean agents: {}; std deviation time: {};"
                    " std deviation agents: {}".format(
                        current_time,
                        current_agent_number,
                        statistics.mean(self.times),
                        statistics.mean(self.agents),
                        statistics.stdev(self.times),
                        statistics.stdev(self.agents),
                    ),
                )

    def calculate_all_metrics(
        self,
        sample: dict,
        collisions: bool = False,
    ) -> None:
        """Compute metrics for a sample.

        :param sample: vector sample
        :param collisions: should collisions be evaluated or not
        :return: None
        """
        self.net.eval()
        start, end = self.measure_time_start()
        out = self.forward(sample)
        (
            _,
            cumred_forward_pass,
            red_forward_pass,
            red_forward_pass_mode_prob,
        ) = self._reduce_batch_and_cumulate_deltas(out, sample)
        self.log_measure_time_end(start, end, sample)
        (
            best_fw_pass,
            best_mode_prob,
            ground_truth,
            loss_mask,
            most_prob_fw_pass,
        ) = self._postprocess_for_metrics(
            sample,
            cumred_forward_pass,
            red_forward_pass_mode_prob,
        )
        self._compute_metrics(
            best_fw_pass,
            best_mode_prob,
            most_prob_fw_pass,
            ground_truth,
            loss_mask,
            out=out,
            sample=sample,
        )
        self.net.train()

    def _update_evaluation_stats(
        self,
        metrics: dict,
        average: dict,
        counter: dict,
    ) -> None:
        """Update average and counters of metrics statistics based on new
        metrics.

        :param metrics: new values for metrics
        :param average: sum of metrics from previous samples
        :param counter: counter of metrics from previous samples
        :return: None
        """

        for key in (
            "ade",
            "ade_closest",
            "brier_ade_closest",
            "fde",
            "fde_closest",
            "brier_fde_closest",
        ):
            self._update_metric_entries(key, metrics, average, counter)

        if metrics["ide"] is not None:
            if counter["ide"] != c.NO_VALUE:
                counter["ide"] = np.zeros_like(metrics["ide"])
                average["ide"] = np.zeros_like(metrics["ide"])
            for j in range(len(counter["ide"])):
                if metrics["ide"][j] != c.NO_VALUE:
                    counter["ide"][j] += 1
                    average["ide"][j] += metrics["ide"][j]

    def evaluate(self, data_loader_validation: DataLoader, epoch: int) -> None:
        """Evaluate current state of model on validation set.

        :param data_loader_validation:
        :param epoch: epoch that is evaluated
        :return: None
        """
        average = self.initialize_average_metrics(self.col_switch, vectors=True)
        counter = self.initialize_average_metrics(self.col_switch, vectors=True)
        for i, mini_batch in enumerate(
            tqdm(
                data_loader_validation,
                desc="Validation - Batch: ",
                leave=False,
                position=1,
            ),
        ):
            mini_batch = duv.shorten_gt_to_prediction_length(
                mini_batch,
                self.opts["hyperparameters"]["prediction_length"],
                self.fps,
            )
            if duv.skip_sample(mini_batch):
                continue
            self.calculate_all_metrics(mini_batch)
            metrics = self.get_metrics()
            self._update_evaluation_stats(metrics, average, counter)
            if i % self.opts["config_params"]["print_step"] == 0:
                logging.info(
                    f"Calculated current metric of validation sample {str(i).zfill(6)}: "
                    f"ADE: {metrics['ade']:.3f}; "
                    f"FDE: {metrics['fde']:.3f}; ",
                )
        for key in counter.keys():
            if key != "ide" and counter[key] > 0:
                average[key] = average[key] / counter[key]
        self.update_logs(global_step=epoch, training=False, metric=average)
        self.average_evaluation_metrics = average
        logging.info(
            "Metric on whole validation dataset in epoch {}: ADE: {}; FDE: {}"
            "".format(epoch, average["ade"], average["fde"]),
        )

    def generate_trajectory_log(self, trajectories, sample):
        input = sample["x"].cpu().numpy()
        trajectories = trajectories.detach().cpu().numpy()
        nbr_agents = trajectories.shape[2]
        nbr_modes = trajectories.shape[0]
        for agent in range(nbr_agents):
            logging.debug("trajectory input (target)")
            target = sample["index_target"][agent]
            logging.debug("{}".format(input[:, agent, target, c.INDICES_COORDINATES]))
            for mode in range(nbr_modes):
                trajectory_mode_agent = trajectories[mode, :, agent, :]
                logging.debug(
                    "timestep {}; agent {}; mode {}".format(self.env_step, agent, mode),
                )
                logging.debug("    {}".format(trajectory_mode_agent))

    @staticmethod
    def _update_metric_entries(
        key: str,
        metrics: dict,
        average: dict,
        counter: dict,
    ) -> None:
        """Update dicts that collect average and counters of metrics.

        :param key: key which should be updated in counter and average dicts
        :param metrics: dict with ["metric_name": results]
        :param average: sum over all samples so far for each metric
        :param counter: number of all samples so far for each metric
        :return: None
        """
        if metrics[key] != c.NO_VALUE:
            counter[key] += 1
            average[key] += metrics[key]

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


class VNetClass(VectorBaseClass):
    net_class: Type[model.modules.VNet] = model.modules.VNet
    net: model.modules.VNet

    def __init__(self, opts):
        super().__init__(opts)
        self.optimizer = model.models.create_optimizer(self, opts)
        self.use_self_attention = not model.modules.is_multihead(self.opts)
        if opts["config_params"]["lr_scheduler"] != 0:
            self.lr_scheduler = model.models.create_lr_scheduler(self, opts)

    def forward(
        self,
        sample: dict,
    ) -> Union[AttentionPredictionOutput, LSTMAttentionPredictionOutput]:
        """Predicts trajectories of agents given the data in sample.

        :param sample: see README.md -> "Data formats" for details on the format.
        :return: predictions
        """
        self.net.sample = sample
        agent_data = sample["x"].float()
        targets_indices = sample["index_target"]
        lane_graph = sample["map"].float()
        out = self.net(
            agent_data,
            targets_indices,
            lane_graph,
        )
        return out

    def draw_model(self, sample):
        # Not yet implemented
        pass

    def get_attention_mask(self, sample):
        agent_data = sample["x"].float()
        targets_indices = sample["index_target"]
        lane_graph = sample["map"].float()
        attention_mask = self.net.get_attention_mask(
            agent_data,
            targets_indices,
            lane_graph,
        )
        return attention_mask

    def get_scores_for_targets(self, sample: dict) -> torch.Tensor:
        """compute scores for target of first batch.

        :param sample:
        :return: attention scored w.r.t. input nodes
        """
        attention_mask = self.get_attention_mask(sample=sample)
        target_attention_mask = du.get_targets(attention_mask, sample["index_target"])
        return target_attention_mask
