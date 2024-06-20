# ------------------------------------------------------------------------------
# Copyright:    Technical University Dortmund
# Project:      KISSaF
# Created by:   Institute of Control Theory and System Engineering
# ------------------------------------------------------------------------------

import torch as torch
from .dynamical_system import DynSystem


class SingleIntegratorModel(DynSystem):
    def __init__(self):
        super().__init__()
        self.num_of_states = 4  #  x, y, dummy, dummy (for consistency with other models)
        self.num_of_controls = 2  #  v_x, v_y
        self.car_lengths = torch.empty((1, 1))
        self.car_widths = torch.empty((1, 1))

    def state_transition(self, control, current_state, dt):
        x_0 = current_state[:, :, :, 0]
        y_0 = current_state[:, :, :, 1]

        v_x = control[:, :, :, 0]
        v_y = control[:, :, :, 1]

        x = x_0 + torch.cumsum(v_x * dt, dim=-1)
        y = y_0 + torch.cumsum(v_y * dt, dim=-1)

        dummy_v = torch.zeros_like(x)
        dummy_theta = torch.zeros_like(x)

        # output Trajectory
        traj = torch.stack([x, y, dummy_theta, dummy_v], dim=-1)
        return traj



