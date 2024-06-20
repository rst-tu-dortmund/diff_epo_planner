# ------------------------------------------------------------------------------
# Copyright:    ZF Friedrichshafen AG
# Project:      KISSaF
# Created by:   ZF AI LAB SBR
# ------------------------------------------------------------------------------
import itertools
import logging
import math

import data.utils.helpers as du
import model.multimode_modules as mmm
import model.modules as mm
import numpy as np
import torch
from data import constants as c
from torch.nn import L1Loss



def get_coefficients_gaussian2d_likelihood(outputs: np.ndarray):
    """
    Extracts the mean, standard deviation and correlation
    Args:
        outputs: DL algo output

    Returns: Gaussian coefficients
    """

    mux, muy, sx, sy, corr = (
        outputs[:, :, 0],
        outputs[:, :, 1],
        outputs[:, :, 2],
        outputs[:, :, 3],
        outputs[:, :, 4],
    )
    sx = torch.exp(sx)
    sy = torch.exp(sy)
    corr = torch.tanh(corr)
    return mux, muy, sx, sy, corr


def gaussian2d_likelihood(estimates, targets):
    """
    Computes the likelihood of the ground truth based on predicted gaussian 2d
    distributions
    Args:
        estimates: predictions from DL algo
        targets: ground truth

    Returns: likelihood based on gaussian 2d distribution

    """
    # seq_length = estimates.shape[0]

    # Extract mean, std devs and correlation
    mux, muy, sx, sy, corr = get_coefficients_gaussian2d_likelihood(estimates)

    # Compute factors
    normx = targets[:, :, 0] - mux
    normy = targets[:, :, 1] - muy
    sxsy = sx * sy

    z = (normx / sx) ** 2 + (normy / sy) ** 2 - 2 * ((corr * normx * normy) / sxsy)
    negRho = 1 - corr ** 2

    # Numerator
    result = torch.exp(-z / (2 * negRho))
    # Normalization factor
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

    # Final PDF calculation
    result = result / denom

    # Numerical stability
    epsilon = 1e-20

    result = -torch.log(torch.clamp(result, min=epsilon))

    loss = torch.sum(result)
    return loss


def l1_distance(estimates, ground_truth, loss_masks):
    """
    Args:
        estimates: Estimated trajectory of shape (batch size, sequence length, 2)
        ground_truth: Available trajectory of shape (batch size, sequence length, 2)
        loss_masks: A tensor derived from the weights of the moving agents for a scene

    Returns:
        Sum of l1 distances between respective points
    """

    assert estimates.shape == ground_truth.shape, (
        "for l1 loss estimates(shape:{}) and ground_truth(shape{}) need to "
        "be of same shape".format(estimates.shape, ground_truth.shape)
    )
    difference = torch.abs(ground_truth - estimates)
    if loss_masks != "":
        difference = torch.sum(difference, dim=-1, keepdim=True) * loss_masks

    return torch.sum(difference)


def l2_distance(estimates, ground_truth, loss_masks):
    """
    calculates l2 distance between predictions and ground truth and masks out
    unobserved references (e.g. due to occlusion)
    Args:
        estimates: Estimated trajectory of shape (batch size, sequence length, 2)
        ground_truth: Available trajectory of shape (batch size, sequence length, 2)
        loss_masks: A tensor derived from the weights of the moving agents for a scene

    Returns:
        Square root of the sum of l2 distances between respective points
    """

    assert estimates.shape == ground_truth.shape, (
        "for l1 loss estimates(shape:{}) and ground_truth(shape{}) need to "
        "be of same shape".format(estimates.shape, ground_truth.shape)
    )

    l2_distances = torch.norm((ground_truth - estimates), p=2, dim=-1, keepdim=True)
    if logging.root.isEnabledFor(logging.DEBUG):
        logging.debug("-------output l2 loss---------")
        logging.debug(estimates.clone().detach().cpu().numpy())
        logging.debug(l2_distances.clone().detach().cpu().numpy())
        logging.debug("------------------------------")

    if loss_masks != "":
        l2_distances = l2_distances * loss_masks

    return torch.sum(l2_distances)


def l1_generalization(model):
    """
    Returns the l1 regularization loss for a given model
    Args:
        model: The network that needs to be regularized

    Returns:
        l1 regularization loss
    """
    l1_regularization_loss = 0
    l1_criterion = L1Loss(reduction="sum")

    for param in model.parameters():
        l1_regularization_loss += l1_criterion(
            param,
            target=torch.zeros_like(param).to(model.device),
        )

    return l1_regularization_loss


class NegativeMultiLogLikelihood(torch.nn.Module):
    def __init__(self):
        super(NegativeMultiLogLikelihood, self).__init__()

    def forward(
            self,
            sample: dict,
            pred: torch.Tensor,
            prob: torch.Tensor,
    ) -> torch.Tensor:
        """
            sample: sample received from dataloader
            pred Tensor of shape (modes x time x agetns x 2D coords)
            prob Tensor of shape (agetns x modes) with a confidence for each mode in each sample
        Returns:
            Tensor: summed negative log-likelihood for the whole batch
        """
        assert (
                len(pred.shape) == c.PREDICTION_SHAPE_LENGTH
        ), f"expected 4D (MxTxAxC) array for pred, got {pred.shape}"
        num_modes, future_len, num_agents, num_coords = pred.shape
        prob = prob.transpose(c.MODE_DIM_PROB_PREDICTOR, c.AGENT_DIM_PROB_PREDICTOR)
        device = pred.device

        assert sample["y"].shape[c.FIRST_TWO_VALUES] == (
            future_len,
            num_agents,
        ), f"expected (Time x Coords) array for gt, got {sample['y'].shape}"
        assert prob.shape == (
            num_modes,
            num_agents,
        ), f"expected (Modesx Agents) array for gt, got {prob.shape}"
        assert sample["loss_masks"].shape[c.FIRST_TWO_VALUES] == (
            future_len,
            num_agents,
        ), f"expected (Time x Agents) array for gt, got {sample['loss_masks'].shape}"
        # assert all data are valid
        assert torch.isfinite(pred).all(), "invalid value found in pred"
        assert torch.isfinite(sample["y"]).all(), "invalid value found in gt"
        assert torch.isfinite(prob).all(), "invalid value found in prob"
        assert torch.isfinite(
            sample["loss_masks"],
        ).all(), "invalid value found in loss_masks"

        gt = torch.stack(
            num_modes * [sample["y"][:, :, c.FIRST_TWO_VALUES]],
            dim=c.FIRST_VALUE,
        ).to(device)
        loss_masks = torch.stack(
            num_modes
            * [
                torch.stack(
                    num_coords * [sample["loss_masks"][:, :, 0]],
                    dim=c.LAST_VALUE,
                ),
            ],
            dim=c.FIRST_VALUE,
        ).to(device)
        squared_displacement_error = torch.sum(
            ((gt - pred) * loss_masks) ** 2,
            dim=c.LAST_VALUE,
        )  # reduce coordinate dim

        with np.errstate(
                divide="ignore",
        ):  # when confidence is 0 log goes to -inf, but we're fine with it
            negative_multi_log_likelihood = torch.log(prob) - 0.5 * torch.sum(
                squared_displacement_error,
                dim=c.TIME_DIM_PREDICTION,
            )  # reduce time

        # use max aggregator on modes for numerical stability
        max_value = (
            negative_multi_log_likelihood.max()
        )  # error are negative at this point, so max() gives the minimum one
        return_value = (
                -torch.log(torch.sum(torch.exp(negative_multi_log_likelihood - max_value)))
                - max_value
        )  # reduces modes and agents
        return return_value


class VariatyRegressionAndClassification(torch.nn.Module):
    """Loss for multimodal trajectory prediction.

    Combines a regression loss for the trajectory values and a
    classification loss to detect best fitting mode w.r.t to the
    generated candidates
    """

    def __init__(self, opts: dict):
        """Assigns probability and regression losses possibility to extend this
        loss for probability caliration So far implemented losses:

        probability:
        - NLL

        regression:
        - L2-Distance

        :param opts: project configrations
        :type opts: dict
        """
        super(VariatyRegressionAndClassification, self).__init__()
        self.prob_loss = mmm.CustomNllLoss()
        self.regression_loss = l2_distance
        self.opts = opts
        self.classification_only_loss = False
        if "mode_calibration_method" in opts["config_params"]:
            if opts["config_params"]["mode_calibration_method"] == 1:
                self.classification_only_loss = True
                logging.debug(
                    "Calibration mode: regression part of VariatyRegressionAndClassification is disabled",
                )

    def compute_regression_loss(
            self,
            pred: torch.Tensor,
            ground_truth: torch.Tensor,
            loss_masks: torch.Tensor,
            delta: bool,
    ) -> float:
        """calculates the regression loss by using the defined regression_loss
        function.

        Args:
            pred: predicted trajectories with shape (1, timesteps, agents, 2D coords)
            ground_truth: ground_truth trajectories with shape (timesteps, agents,
            out_features)
            loss_masks: loss_masks with shape (timesteps, agents, 1)
            delta: boolean whether to use deltas or coords

        Returns:
        """
        if delta:
            loss = self.regression_loss(
                pred[c.FIRST_VALUE, ...],
                ground_truth[..., c.INDICES_DELTAS],
                loss_masks,
            )
        else:
            loss = self.regression_loss(
                pred[c.FIRST_VALUE, ...],
                ground_truth[..., c.INDICES_COORDINATES],
                loss_masks,
            )
        logging.debug("-------output reg loss---------")
        logging.debug(loss.clone().detach().cpu().numpy())
        logging.debug("-----------------")

        return loss

    def forward(
            self,
            pred: torch.Tensor,
            prob: torch.Tensor,
            sample: dict,
            delta: bool = False,
    ) -> torch.Tensor:
        """negative log likelihood summed with the l2 distance of the closest
        mode.

        :param pred: array of shape (modes x time x agetns x 2D coords)
        :param prob: Tensor of shape (agetns x modes) with a confidence for each mode in
         each sample
        :param sample:  sample received from dataloader
        :param delta: boll to switch between loss calculation based on coordinates in a
        realtive system or deltas
        :return: sum of nll between the predicted probability and the predicted mode as
        well as distance between
            gt traj and closest prediction
        """
        loss = 0
        device = pred.device
        logging.debug("-------output preds---------")
        logging.debug(pred.clone().detach().cpu().numpy())
        logging.debug("-----------------")
        ground_truth = du.format_gt(sample, self.opts)
        loss_masks = du.format_loss_masks(sample, self.opts)
        if pred.shape[c.FIRST_VALUE] > c.ONE:

            # handle multi-modality
            if delta:
                closest_mode_to_gt, best_mode_prob = mmm.closest_pred_to_gt(
                    multi_mode_y_pred=pred,
                    gt=ground_truth[..., c.INDICES_DELTAS].to(device),
                    loss_masks=loss_masks.to(device),
                    mode_prob=prob,
                )
            else:
                closest_mode_to_gt, best_mode_prob = mmm.closest_pred_to_gt(
                    multi_mode_y_pred=pred,
                    gt=ground_truth[..., c.INDICES_COORDINATES].to(device),
                    loss_masks=loss_masks.to(device),
                    mode_prob=prob,
                )
            # the loss of the closest to gt will be minimized below
            fw_pass_output = closest_mode_to_gt
            fw_pass_output = torch.unsqueeze(fw_pass_output, c.FIRST_VALUE)

            # maximize the prob of best mode for each agent!
            if self.opts["config_params"]["no_probs"] is False:
                loss += self.prob_loss(best_mode_prob.to(device))
        elif pred.shape[c.FIRST_VALUE] == c.ONE:
            fw_pass_output = pred
        if not self.classification_only_loss:
            loss += self.compute_regression_loss(
                fw_pass_output.to(device),
                ground_truth.to(device),
                loss_masks.to(device),
                delta,
            )

        return loss


class ClassIndependentCollisionLoss(torch.nn.Module):
    def __init__(self, opts: dict):
        super(ClassIndependentCollisionLoss, self).__init__()
        self.opts = opts

    def forward(
            self,
            pred: torch.Tensor,
            prob: torch.Tensor,
            sample: dict,
    ) -> torch.Tensor:
        if pred.shape[c.FIRST_VALUE] > c.ONE:
            most_probable_pred = mmm.get_highest_probability_mode(
                multi_mode_predictions=pred,
                prob_distributions=prob,
                device=pred.device,
            )
        elif pred.shape[c.FIRST_VALUE] == c.ONE:
            most_probable_pred = pred
        most_probable_pred = most_probable_pred.permute(
            c.MODE_DIM_PREDICTION,
            c.AGENT_DIM_PREDICTION,
            c.TIME_DIM_PREDICTION,
            c.PARAM_DIM_PREDICTION,
        )
        most_probable_pred = most_probable_pred.squeeze(c.MODE_DIM_PREDICTION)
        collision_loss = c.ZERO
        for index_pair in sample["scene_indices"]:
            most_probable_pred_per_scene = most_probable_pred[
                                           index_pair[c.FIRST_VALUE]: index_pair[c.LAST_VALUE], ...
                                           ]
            for agent_sample_1, agent_sample_2 in itertools.combinations(
                    most_probable_pred_per_scene,
                    c.TWO,
            ):
                distances = torch.norm(
                    (agent_sample_1 - agent_sample_2),
                    dim=c.LAST_VALUE,
                    p=2,
                )  # L2 distance
                distances = distances - c.THRESHOLD_INDEPENDENT
                distances[distances > c.ZERO] = c.ZERO
                collision_loss += torch.sum(torch.abs(distances))
        return collision_loss


class Gaussian_Nll(torch.nn.Module):
    def __init__(self, opts: dict):
        super(Gaussian_Nll, self).__init__()
        self.opts = opts

    def forward(
            self,
            pred: torch.Tensor,
            prob: torch.Tensor,
            sample: dict,
    ) -> torch.Tensor:
        loss = 0
        device = pred.device
        ground_truth = du.format_gt(sample, self.opts).to(device)
        loss_masks = du.format_loss_masks(sample, self.opts).to(device)
        for mode in pred:
            loss += nll_loss_per_mode(mode, ground_truth, loss_masks)
        return loss


def nll_loss_per_mode(
        pred: torch.Tensor,
        data: torch.Tensor,
        mask: torch.Tensor,
) -> torch.Tensor:
    """NLL averages across steps and dimensions, but not samples (agents)."""
    x_mean = pred[:, :, c.ZERO]
    y_mean = pred[:, :, c.ONE]
    x_sigma = pred[:, :, c.TWO]
    y_sigma = pred[:, :, c.THREE]
    rho = pred[:, :, c.FOUR]
    ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)  # type: ignore
    x = data[:, :, c.ZERO]
    y = data[:, :, c.ONE]
    results = (
            0.5
            * torch.pow(ohr, 2)
            * (
                    torch.pow(x_sigma, 2) * torch.pow(x - x_mean, 2)
                    + torch.pow(y_sigma, 2) * torch.pow(y - y_mean, 2)
                    - 2
                    * rho
                    * torch.pow(x_sigma, 1)
                    * torch.pow(y_sigma, 1)
                    * (x - x_mean)
                    * (y - y_mean)
            )
            - torch.log(x_sigma * y_sigma * ohr)
            + math.log(2 * math.pi)
    )
    results = results * mask[..., c.ZERO]  # nSteps by nBatch
    loss = torch.sum(results) / (torch.sum(mask[..., c.ZERO]) + c.EPSILON_STABILITY)
    if torch.isnan(loss):
        raise ValueError("Got nan in loss")
    return loss


