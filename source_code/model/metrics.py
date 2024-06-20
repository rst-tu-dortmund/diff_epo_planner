# ------------------------------------------------------------------------------
# Copyright:    ZF Friedrichshafen AG
# Project:      KISSaF
# Created by:   ZF AI LAB SBR
# ------------------------------------------------------------------------------
import logging

import data.constants as const
import numpy as np
import torch


def mean_squared_error(y_gt, y_pred):
    l2_distances = torch.sqrt(
        torch.pow(y_gt[:, :, 0] - y_pred[:, :, 0], 2)
        + torch.pow(y_gt[:, :, 1] - y_pred[:, :, 1], 2),
    )
    return torch.sum(l2_distances / y_pred.shape[0], dim=1)


def displacement_error_per_timestep(
    estimates: np.ndarray,
    ground_truth: np.ndarray,
    loss_mask: np.ndarray,
) -> np.ndarray:
    """
    calculates displacements error for each timestep
    assumption: at least one ground truth per timestep
    :param estimates:
    :param ground_truth:
    :param loss_mask:
    :return:
    """
    displacements = calc_displacements(estimates, ground_truth)

    if len(loss_mask) != const.ZERO:
        displacements *= loss_mask
        displacements = np.sum(
            displacements.squeeze(const.LAST_VALUE),
            axis=const.LAST_VALUE,
        )
        loss_mask = np.sum(loss_mask.squeeze(const.LAST_VALUE), axis=const.LAST_VALUE)
        displacements_per_timestep = displacements / loss_mask
    else:
        displacements_per_timestep = np.sum(
            displacements.squeeze(const.LAST_VALUE),
            axis=const.LAST_VALUE,
        )
    return displacements_per_timestep


def brier_ade(
    estimates: np.ndarray,
    ground_truth: np.ndarray,
    loss_mask: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    """calculation of metric brier ade
    :param estimates:
    :param ground_truth:
    :param loss_mask:
    :param probabilities:
    :return:
    """
    ade = average_displacement_error(estimates, ground_truth, loss_mask)
    probability_term = np.mean(np.square(1 - probabilities))
    brier_ade = ade + probability_term
    return brier_ade


def calc_displacements(estimates: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    """calculates displacement for each coordinate pair.

    :param estimates:
    :param ground_truth:
    :return:
    """
    if estimates.shape[const.LAST_VALUE] == const.VECTOR_SIZE_CARTESIAN:
        # For the case of direct coordinate prediction
        displacements = np.absolute(estimates - ground_truth)
    elif estimates.shape[const.LAST_VALUE] == const.VECTOR_SIZE_GAUSSIAN:
        # For the case of modeling the out put as 2d gaussian, the means
        # are taken for metric calc
        displacements = np.absolute(
            (estimates[..., const.INDICES_COORDINATES] - ground_truth),
        )
    displacements = np.sqrt(
        np.sum(np.square(displacements), keepdims=True, axis=const.LAST_VALUE),
    )
    return displacements


def average_displacement_error(
    estimates: np.ndarray,
    ground_truth: np.ndarray,
    loss_mask: np.ndarray,
) -> np.ndarray:
    """
    Method to calculate the average displacement error against the ground truth
    Args:
        estimates: Array of estimated displacements
        ground_truth: Array of available ground truth
        loss_mask: A tensor derived from the weights of the moving agents for a scene
         (time, agents)

    Returns:
        Average displacement error value

    """
    if np.sum(loss_mask) == const.ZERO:
        ade = const.STANDARD_INVALID_VALUE
        logging.debug("No futures in sample check data pipeline")
    else:
        displacements = displacement_error_per_timestep(
            estimates,
            ground_truth,
            loss_mask,
        )
        ade = np.mean(displacements)
    return ade


def final_displacement_error(
    estimates: np.ndarray,
    ground_truth: np.ndarray,
    loss_mask: np.ndarray,
) -> np.ndarray:
    """calculates final displacement error.

    :param estimates:
    :param ground_truth:
    :param loss_mask:
    :return:
    """
    displacements_per_timestep = displacement_error_per_timestep(
        estimates,
        ground_truth,
        loss_mask,
    )
    if valid_end(loss_mask):
        fde = displacements_per_timestep[const.LAST_VALUE, ...]
    else:
        fde = const.STANDARD_INVALID_VALUE

    return fde


def brier_fde(
    estimates: np.ndarray,
    ground_truth: np.ndarray,
    loss_mask: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    """calculation of brier fde
    :param estimates:
    :param ground_truth:
    :param loss_mask:
    :param probabilities:
    :return:
    """
    fde = final_displacement_error(estimates, ground_truth, loss_mask)
    probability_term = np.mean(np.square(1 - probabilities))
    brier_fde = fde + probability_term
    return brier_fde


def interval_displacement_error(
    estimates,
    ground_truth,
    loss_mask,
    steps_per_interval=4,
):
    """

    Args:
        estimates: Array of estimated displacements
        ground_truth: Array of available ground truth
        loss_mask: A tensor derived from the weights of the moving agents for a scene
        steps_per_interval: Number of steps per chunk

    Returns:
        An array of interval displacement errors

    """
    estimates_chunked = [
        estimates[i * steps_per_interval : (i + 1) * steps_per_interval, ...]
        for i in range((len(estimates) + steps_per_interval - 1) // steps_per_interval)
    ]
    ground_truth_chunked = [
        ground_truth[i * steps_per_interval : (i + 1) * steps_per_interval, ...]
        for i in range((len(estimates) + steps_per_interval - 1) // steps_per_interval)
    ]
    loss_masks_chunked = [
        loss_mask[i * steps_per_interval : (i + 1) * steps_per_interval, ...]
        for i in range((len(estimates) + steps_per_interval - 1) // steps_per_interval)
    ]
    interval_displacements = []
    for estimate_chunk, ground_truth_chunk, loss_masks_chunk in zip(
        estimates_chunked,
        ground_truth_chunked,
        loss_masks_chunked,
    ):
        interval_displacements.append(
            average_displacement_error(
                estimates=estimate_chunk,
                ground_truth=ground_truth_chunk,
                loss_mask=loss_masks_chunk,
            ),
        )
    return np.asarray(interval_displacements)


def collisions(prediction, ground_truth):
    """
    Args:
        prediction: Array of predicted positions
        ground_truth: Array of available ground truth

    Returns:
        The number of collisions in the current scene

    """
    collision_matrix = np.zeros((prediction.shape[1], prediction.shape[1]))

    for prediction_time_step, ground_truth_time_step in zip(prediction, ground_truth):
        for i, (agent_prediction, agent_ground_truth) in enumerate(
            zip(prediction_time_step, ground_truth_time_step),
        ):
            current_agent_box = BirdsEyeBox(
                center=agent_prediction,
                size=agent_ground_truth[..., 6:8],
                yaw=agent_ground_truth[..., 5],
            )
            current_corners = current_agent_box.corners()
            current_polygon = Polygon(current_corners)

            for j, (
                agent_prediction_for_comparison,
                agent_ground_truth_for_comparison,
            ) in enumerate(zip(prediction_time_step, ground_truth_time_step)):
                if j == i:
                    break
                current_agent_box_comparison = BirdsEyeBox(
                    center=agent_prediction_for_comparison,
                    size=agent_ground_truth_for_comparison[..., 6:8],
                    yaw=agent_ground_truth_for_comparison[..., 5],
                )
                current_corners_comparison = current_agent_box_comparison.corners()
                current_polygon_comparison = Polygon(current_corners_comparison)

                if (i != j) and current_polygon.intersects(current_polygon_comparison):
                    collision_matrix[i, j] = 1

    number_of_collisions = np.sum(collision_matrix)

    return number_of_collisions


def valid_end(loss_masks):
    if np.sum(loss_masks[const.LAST_VALUE, ...]) > 0:
        check = True
    else:
        check = False
    return check


# ------------------------------------------------------------------------------
# Copyright:    Technical University Dortmund
# Project:      KISSaF
# Created by:   Institute of Control Theory and System Engineering
# ------------------------------------------------------------------------------


import torch.nn as nn
import data.constants as c
import einops
from shapely.geometry import Polygon

class Metrics(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.device = device

    @staticmethod
    def minADE(y, yhat):
        """
        Calculates the minimum average displacement error
        :param y: ground truth joint trajectory
        :param yhat: multimodal predicted trajectories
        """

        b, m, p, t, s = yhat.shape

        ade_loss_storage = torch.zeros((b, m, p)).to("cuda")

        for mode in range(m):
            l2_distances = torch.sqrt(
                torch.pow(y[..., c.X_IND] - yhat[:, mode, ..., c.X_IND], 2)
                +
                torch.pow(y[..., c.Y_IND] - yhat[:, mode, ..., c.Y_IND], 2))
            mode_ade = torch.sum(l2_distances / t, dim=2)  # mean over time
            ade_loss_storage[:, mode, :] = mode_ade

        # take the minimum over the modes
        min_mode_loss, min_mode_ind = torch.min(ade_loss_storage, dim=1)
        min_mode_loss = torch.mean(min_mode_loss, dim=1)  # mean over agents

        return torch.mean(min_mode_loss), min_mode_ind, min_mode_loss

    @staticmethod
    def minSADE(y, yhat):
        """
        Calculates the minimum scene average displacement error
        :param y: ground truth joint trajectory
        :param yhat: multimodal predicted trajectories
        """
        b, m, p, t, s = yhat.shape
        sade_loss_storage = torch.zeros((b, m, p)).to("cuda")
        for mode in range(m):
            l2_distances = torch.sqrt(
                torch.pow(y[..., c.X_IND] - yhat[:, mode, ..., c.X_IND], 2)
                +
                torch.pow(y[..., c.Y_IND] - yhat[:, mode, ..., c.Y_IND], 2))
            mode_ade = torch.sum(l2_distances / t, dim=2)  # sum over time
            sade_loss_storage[:, mode, :] = mode_ade

        sade_loss_storage = torch.mean(sade_loss_storage, dim=2)  # mean over agents

        min_mode_loss, min_mode_ind = torch.min(sade_loss_storage, dim=1)
        return torch.mean(min_mode_loss), min_mode_ind, min_mode_loss


    @staticmethod
    def minSFDE(y, yhat):
        """
        Calculates the minimum scene final displacement error
        :param y: ground truth joint trajectory
        :param yhat: multimodal predicted trajectories
        """
        b, m, p, t, s = yhat.shape

        sfde_loss_storage = torch.zeros((b, m)).to("cuda")
        for mode in range(m):
            l2_distances = torch.sqrt(
                torch.pow(y[..., -1, c.X_IND] - yhat[:, mode, ..., -1, c.X_IND], 2)
                +
                torch.pow(y[..., -1, c.Y_IND] - yhat[:, mode, ..., -1, c.Y_IND], 2))

            sfde_loss_storage[:, mode] = l2_distances.mean(dim=1)  # mean over agents

        min_mode_loss, min_mode_ind = torch.min(sfde_loss_storage, dim=1)
        return torch.mean(min_mode_loss), min_mode_ind, min_mode_loss

    @staticmethod
    def minFDE(y, yhat):
        """
        Calculates the minimum final displacement error
        :param y: ground truth joint trajectory
        :param yhat: multimodal predicted trajectories
        """
        b, m, p, t, s = yhat.shape
        fde_loss_storage = torch.zeros((b, p, m)).to("cuda")
        for player in range(p):

            for mode in range(m):
                l2_distances = torch.sqrt(
                    torch.pow(y[:, player, -1, c.X_IND] - yhat[:, mode, player, -1, c.X_IND], 2)
                    +
                    torch.pow(y[:, player, -1, c.Y_IND] - yhat[:, mode, player, -1, c.Y_IND], 2))
                mode_fde = l2_distances
                fde_loss_storage[:, player, mode] = mode_fde

        # take for each sample the mode with the lowest loss
        min_mode_loss, min_mode_ind = torch.min(fde_loss_storage, dim=2)
        minFDE = min_mode_loss.mean(dim=1)  # mean over player

        # * mean over batch
        return torch.mean(minFDE), min_mode_ind, minFDE

    def overlap_rate(self, yhat, scene_probs, dynamics_model):
        """
        counts the number of overlaps between rectangular shapes of the most likely joint scene predcition
        :param yhat: predicted joint trajectories
        :param scene_probs: scene probabilities
        :param dynamics_model: dynamics model
        """
        # set init values
        b, modes, p, t, s = yhat.shape

        car_lengths = dynamics_model.car_lengths
        car_widths = dynamics_model.car_widths

        overlaps_list = []
        car_lengths = einops.rearrange(car_lengths, '(b m) p -> b m p', b=b)[:, 0]
        car_widths = einops.rearrange(car_widths, '(b m) p -> b m p', b=b)[:, 0]

        # extract most probable mode
        most_prob_mode = torch.argmax(scene_probs, dim=1)
        yhat = torch.gather(yhat, dim=1, index=most_prob_mode[:, None, None, None, None].repeat(1, 1, p, t, 3))[:, 0]

        for i in range(0, p - 1):

            for j in range(i + 1, p, 1):

                rect_agent_i = get_rotated_bbox(
                    yhat[:, i, :, c.X_IND],
                    yhat[:, i, :, c.Y_IND],
                    car_lengths[:, i, None],
                    car_widths[:, i, None],
                    yhat[:, i, :, c.THETA_IND])

                rect_agent_j = get_rotated_bbox(
                    yhat[:, j, :, c.X_IND],
                    yhat[:, j, :, c.Y_IND],
                    car_lengths[:, j, None],
                    car_widths[:, j, None],
                    yhat[:, j, :, c.THETA_IND])

                polygons_i_list = []
                polygons_j_list = []

                for k in range(rect_agent_i.shape[0]):  # Batch

                    polygon_i_batch = []
                    polygon_j_batch = []

                    for l in range(rect_agent_i.shape[1]):  # Time

                        polygon_i_edges = []
                        polygon_j_edges = []

                        for m in range(rect_agent_i.shape[2]):  # Edge

                            edge_i = (rect_agent_i[k, l, m, 0], rect_agent_i[k, l, m, 1])
                            edge_j = (rect_agent_j[k, l, m, 0], rect_agent_j[k, l, m, 1])

                            polygon_i_edges.append(edge_i)
                            polygon_j_edges.append(edge_j)

                        polygon_i = Polygon(polygon_i_edges)
                        polygon_j = Polygon(polygon_j_edges)

                        polygon_i_batch.append(polygon_i)
                        polygon_j_batch.append(polygon_j)

                    polygons_i_list.append(polygon_i_batch)
                    polygons_j_list.append(polygon_j_batch)

                intersections = torch.zeros((rect_agent_i.shape[0], rect_agent_i.shape[1]))
                for k in range(rect_agent_i.shape[0]):  # for every batch
                    for l in range(rect_agent_i.shape[1]):  # for every timestep
                        intersection = polygons_i_list[k][l].intersection(polygons_j_list[k][l])
                        intersections[k, l] = intersection.area

                intersections[intersections != 0.0] = 1.0
                overlaps_list.append(intersections)

        overlaps = torch.stack(overlaps_list)
        overlaps = einops.rearrange(overlaps, 'd b t -> b t d').cuda()

        num_of_overlaps_per_distance = torch.any(overlaps, dim=1)
        num_of_overlaps_per_batch = num_of_overlaps_per_distance.sum(dim=1) / p
        overlap_rate_avergaged_over_batch = num_of_overlaps_per_batch.sum() / b
        return overlap_rate_avergaged_over_batch, num_of_overlaps_per_batch

    def predict_and_calc_scene_prob_loss(self, mini_batch, net, yhat):
        """
        Calculates the scene probability loss
        :param mini_batch: mini_batch dict
        :param net: neural network
        :param yhat: predicted joint trajectories
        """
        groundtruth = einops.rearrange(mini_batch['y'], 't b a s -> b a t s')[..., 0:2].to("cuda:0")

        # get index of best scene mode
        _, best_mode_ind, _ = self.minSADE(groundtruth, yhat)
        best_mode_ind = best_mode_ind.to("cuda:0")

        # reshape and predict
        yhat = einops.rearrange(yhat, 'b m p t s -> b (p m t s)')
        inter_out = einops.rearrange(net.inter_out.clone(), 'b a f -> b (a f)') # clone
        prob_head_input = torch.cat([inter_out, yhat.clone().type(torch.float32).detach()], dim=1)

        # forward pass of scene_prob_decoder
        scene_probs = net.scene_prob_predictor(prob_head_input)

        # extract best mode from scene_prob
        best_mode_probs = torch.gather(scene_probs, 1, best_mode_ind.unsqueeze(1))
        best_mode_probs[best_mode_probs<1e-21] = 1e-21 # accounts for numerical stability

        # nll loss
        scene_prob_loss = torch.mean(-1 * torch.log(best_mode_probs))
        return scene_probs, scene_prob_loss

    def calc_target_loss(self, mini_batch, game_goal):
        """
        Calculates the target loss
        :param mini_batch: mini_batch dict
        :param game_goal: predicted game_goal
        """
        goal_trajectory = einops.repeat(game_goal, 'b m p 1 s -> b m p t s', t=len(mini_batch["y"]))  # t dimension contrains allways the same goal
        gt_trajectory = einops.rearrange(mini_batch["y"], 't b p s -> b p t s').to(self.device)
        endpoint_loss, _, _ = self.minSFDE(gt_trajectory, goal_trajectory)

        return endpoint_loss

def get_rotated_bbox(x_center, y_center, length, width, heading):
    """
    Calculate the four corners of a rotated bounding box given the center, length, width and heading.
    :param x_center: center x coordinate
    :param y_center: center y coordinate
    :param length: length of the bounding box
    :param width: width of the bounding box
    :param heading: heading of the bounding box
    """
    centroids = torch.stack([x_center, y_center], dim=1)

    # Precalculate all components needed for the corner calculation
    l = length / 2
    w = width / 2
    c = torch.cos(heading)
    s = torch.sin(heading)

    lc = l * c
    ls = l * s
    wc = w * c
    ws = w * s

    # Calculate all four rotated bbox corner positions assuming the object is located at the origin.
    # To do so, rotate the corners at [+/- length/2, +/- width/2] as given by the orientation.

    # Compute the remaining corners and concatenate them along the last dimension
    front_right_corner = torch.cat([(lc - ws)[:, :, None], (ls + wc)[:, :, None]], dim=2)
    rear_right_corner = torch.cat([(-lc - ws)[:, :, None], (-ls + wc)[:, :, None]], dim=2)
    rear_left_corner = torch.cat([(-lc + ws)[:, :, None], (-ls - wc)[:, :, None]], dim=2)
    front_left_corner = torch.cat([(lc + ws)[:, :, None], (ls - wc)[:, :, None]], dim=2)
    rotated_bbox_vertices = torch.cat(
        [front_right_corner[:, :, None, :], rear_right_corner[:, :, None, :], rear_left_corner[:, :, None, :], front_left_corner[:, :, None, :]], dim=2)

    # Move corners of rotated bounding box from the origin to the object's location
    rotated_bbox_vertices = rotated_bbox_vertices + centroids.swapaxes(1, 2)[:, :, None, :]
    return rotated_bbox_vertices
