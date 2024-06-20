# ------------------------------------------------------------------------------
# Copyright:    Technical University Dortmund
# Project:      KISSaF
# Created by:   Institute of Control Theory and System Engineering
# ------------------------------------------------------------------------------

import torch as torch
from .dynamical_system import DynSystem


class UnicycleModel(DynSystem):
    def __init__(self):
        super().__init__()
        self.num_of_states = 4  #  x, y, theta, v
        self.num_of_controls = 2  #  omega, acc

    def state_transition(self, control, current_state, dt):

        x_0 = current_state[:, :, :, 0] # x position
        y_0 = current_state[:, :, :, 1] # y position
        theta_0 = current_state[:, :, :, 2] # heading angle
        v_0 = current_state[:, :, :, 3] # velocity

        omega = control[:, :, :, 0] # angular velocity
        acc = control[:, :, :, 1] # acceleration

        v = v_0 + torch.cumsum(acc * dt, dim=-1)
        theta = theta_0 + torch.cumsum(omega * dt, dim=-1)
        theta = torch.fmod(theta, 2 * torch.pi)

        x = x_0 + torch.cumsum(v * torch.cos(theta) * dt, dim=-1)
        y = y_0 + torch.cumsum(v * torch.sin(theta) * dt, dim=-1)

        traj = torch.stack([x, y, theta, v], dim=-1)
        return traj



