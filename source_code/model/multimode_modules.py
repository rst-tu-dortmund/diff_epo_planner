# ------------------------------------------------------------------------------
# Copyright:    ZF Friedrichshafen AG
# Project:      KISSaF
# Created by:   ZF AI LAB SBR
# ------------------------------------------------------------------------------
import logging
from typing import Tuple
import torch

import data.utils.helpers as du
from data import constants as c


def closest_pred_to_gt(multi_mode_y_pred, gt, loss_masks, mode_prob=None):
    # change the dimension order given by the current implementation
    multi_mode_y_pred = multi_mode_y_pred.permute(2, 0, 1, 3)[
        ...,
        c.INDICES_COORDINATES,
    ]
    # new shape: [num_agents,num_modes,future_length_in_frames,output_size]
    gt = gt.permute(
        1,
        0,
        2,
    )  # new shape: [num_agents,future_length_in_frames,output_size]
    num_agents = multi_mode_y_pred.shape[0]
    num_modes = multi_mode_y_pred.shape[1]
    future_length_in_frames = multi_mode_y_pred.shape[2]
    output_size = multi_mode_y_pred.shape[3]

    # reshape gt
    gt = gt.view(-1, 1, gt.shape[1], gt.shape[2])  # dim added
    gt = gt.repeat(
        1,
        num_modes,
        1,
        1,
    )  # vector repeated so that subtraction is possible

    # diff
    diff = (
        multi_mode_y_pred - gt
    )  # should be [num_agents,num_modes,future_length_in_frames,output_size]
    # dim will disappear (operator is applied to). Result's shape [num_agents,num_modes]
    dist_to_gt = torch.norm(diff[:, :, :, :], dim=[3], p=2) # - > (batches x agents) x modes x time
    dist_to_gt = dist_to_gt * (loss_masks.permute(1, 2, 0).repeat(1, num_modes, 1)) # apply loss_mask
    dist_to_gt = torch.norm(dist_to_gt, dim=c.TIME_DIM_CLOSEST_MODE_CALCULATION, p=1) # -> mean over time

    # 1nn
    # Largest is False. This is done along (applied to) dim so shape of result is num_agents
    nn = dist_to_gt.topk(1, dim=1, largest=False) # get indices of best modes
    idx = nn.indices.flatten().view(-1)  # nn is a named tuple (nn.indices, nn.value)

    # return best mode for each element in the batch (so for each agent)
    y_pred_best = torch.zeros(num_agents, future_length_in_frames, output_size).to(
        gt.device,
    )
    mode_prob_best = -1 * torch.ones(num_agents).to(gt.device)  # invalid probability
    for b in range(num_agents):
        y_pred_best[b, :, :] = multi_mode_y_pred[b, idx[b], :, :]
        if mode_prob is not None:
            mode_prob_best[b] = mode_prob[b, idx[b]]

    # reorder dims to match the current code interface
    y_pred_best = y_pred_best.permute(1, 0, 2)

    return y_pred_best, mode_prob_best


def closest_pred_to_gt_batched(
    predictions: torch.Tensor,
    ground_truth: torch.Tensor,
) -> Tuple[torch.tensor, torch.Tensor]:
    """Get the closest mode of each agent to ground truth.

    :param predictions: shape=[nbr_modes, nbr_time_steps, nbr_batches, nbr_agents, 2]
    :param ground_truth: shape=[1, nbr_time_steps, nbr_batches, nbr_agents, 2]
    :return: best predictions, shape=[nbr_time_steps, nbr_batches, nbr_agents, 2]
    """
    mode_index = 0
    dist = torch.sum(
        torch.sum(
            torch.square(predictions[..., :2] - ground_truth[..., :2]),
            dim=-1,
            keepdim=True,
        ),
        dim=1,
        keepdim=True,
    )
    best_modes = torch.argmin(dist, dim=mode_index, keepdim=True)
    return torch.take_along_dim(predictions, best_modes, dim=0).squeeze(0), best_modes


"""
Input: - a prediction of shape (num_modes, future_length_in_frames, num_agents, output_size)
       - corresponding distributions of shape (num_agents, num_modes)
Returns the pred of highest prob. Its shape is (1, future_length_in_frames, num_agents, output_size)
"""


def get_highest_probability_mode(multi_mode_predictions, prob_distributions, device):
    assert prob_distributions.shape[0] == multi_mode_predictions.shape[2]
    num_agents = prob_distributions.shape[0]

    num_modes = prob_distributions.shape[1]
    if num_modes == 1:
        return multi_mode_predictions

    # find the mode of highest probability for each agent
    top = prob_distributions.topk(
        1,
        dim=1,
        largest=True,
    )  # largest is True. Applied to dim 1, so result's shape is #agents
    idx = top.indices.flatten().view(-1)
    assert idx.shape == (num_agents,)  # top is a named tuple (nn.indices, nn.value)

    best_mode_pred = torch.zeros(
        1,
        multi_mode_predictions.shape[1],
        multi_mode_predictions.shape[2],
        multi_mode_predictions.shape[3],
    ).to(device)
    for i in range(num_agents):
        best_mode_pred[0, :, i, :] = multi_mode_predictions[idx[i], :, i, :]
    return best_mode_pred


def get_closest_to_gt_characteristic(
    characteristic_matrix: torch.Tensor,
    q_gt: torch.Tensor,
    device: torch.device,
) -> tuple:
    """get the characteristics that are closest to the given ground_truth
    characteristics and the indices belonging to that characteristics for all
    agents.

    Args:
        characteristic_matrix:
        shape -> (characteristic_size^characteristic_size, characteristic_size)
        q_gt: shape(agents, characteristic_size)
        device:

    Returns: characteristics and belonging indices
    """
    total_num_agents = q_gt.shape[c.FIRST_VALUE]
    characteristics = []
    idcs = torch.zeros(total_num_agents).to(device)
    for agent_idx in range(total_num_agents):
        q_gt_agent = q_gt[agent_idx]
        idx_mode = du.select_characteristic(q_gt_agent, characteristic_matrix)
        characteristics.append(characteristic_matrix[idx_mode])
        idcs[agent_idx] = idx_mode
    characteristics = torch.stack(characteristics, dim=c.ZERO).to(device)
    return characteristics, idcs


def get_dummy_prob_distribution(
    sample: torch.Tensor,
    opts: dict,
    device: torch.device,
    vectors: bool = False,
) -> torch.Tensor:
    num_modes = opts["hyperparameters"]["num_modes"]
    predict_targets = opts["config_params"]["predict_target"]
    if not vectors:
        num_agents = sample.shape[c.ONE]
        p = torch.ones(num_agents, num_modes) / num_modes
    else:
        num_batches = sample.shape[c.ONE]
        if predict_targets:
            num_agents = c.ONE
        else:
            num_agents = sample.shape[c.TWO]
        p = torch.ones(num_batches, num_agents, num_modes) / num_modes
    return p.to(device)


def get_dummy_hidden_state(sample, opts, device):
    num_agents = sample["x"].shape[1]
    h = torch.zeros(
        opts["hyperparameters"]["num_modes"],
        num_agents,
        opts["hyperparameters"]["rnn_size"],
    )
    return h.to(device)


class CustomNllLoss(torch.nn.Module):
    """calculates Negative Log Likelihood loss based on Softmax output of
    target class Clamping needs to be added as suggested in
    https://discuss.pytorch.org/t/custom-loss-function-causes-loss-to-go-nan-
    after-certain-epochs/102714/5."""

    def __init__(self, reduction="mean"):
        super(CustomNllLoss, self).__init__()
        self.reduction = reduction

    def forward(
        self,
        prob,
    ):  # expected input's shape is a scalar (probability) per batch, so [batch_size]
        if self.reduction == "mean":
            logging.debug("-------output closest probs---------")
            logging.debug(prob.clone().detach().cpu().numpy())
            logging.debug("-----------------")
            loss = -torch.mean(torch.log(torch.clamp(prob, min=1e-4)))
            logging.debug("-------output all probs---------")
            logging.debug(loss.clone().detach().cpu().numpy())
            logging.debug("-----------------")
            return loss
        else:
            raise Exception("unknown reduction type")
